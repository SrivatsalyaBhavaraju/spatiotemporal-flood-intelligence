"""
Task 2.8 -- graph-snapshot Dataset/data-loader class over task 2.7's model
input schema (X [T,N,F], edge_index [2,E], node_order). One "snapshot" is
one flood phase (pre_event/rising/peak/receding): the whole line graph's
node feature matrix at that phase, paired with the SAME static edge_index
every phase shares (topology doesn't change across phases -- task 2.2
builds one graph, not one per phase; task 3.5 later freezes it for good).

Two consumption modes, matching the two candidate architectures already
smoke-tested end-to-end in toy_gnn_prototype.py (task 1.12) -- reproducing
those EXACT shape conventions here so a later training script (task 4.1)
doesn't have to re-derive them (see that file's README note on how easily
MPNN-LSTM's tiling goes wrong without a shape error, silently mixing the
wrong nodes' features):
  - GraphSnapshotDataset[t]  -- one Data(x, edge_index, y) per phase, for
    generic inspection/iteration/the rule-based baseline (task 3.6).
  - .to_a3tgcn_input()        -- (N, F, T) tensor + static edge_index.
  - .to_mpnn_lstm_input()     -- (T*N, F) tensor + edge_index tiled/offset
    per timestep + unit edge_weight.

Labels (Y_{t+1}) -- task 3.4 (ground truth fusion) hasn't produced these
yet. GraphSnapshotDataset works fully without them (every Data.y is None,
transition_pairs() raises until labels exist); call .attach_labels(df)
once task 3.4's output exists to fill them in, validated against task
2.7's schema.json y_t1_contract (segment_id, phase_id join key; binary;
must cover every node for every phase that has a "next" phase).

Usage (as a library, imported by future baseline/training code -- also
runnable directly as a smoke test against real committed data):
    from src.models.gnn.build_graph_snapshot_dataset import load_dataset
    ds = load_dataset()   # reads task 2.7's outputs from data/processed/model_input/
    ds[0]                  # Data for phase 0 (pre_event)
    X, edge_index = ds.to_a3tgcn_input()
    X, edge_index, edge_weight = ds.to_mpnn_lstm_input()

    python src/models/gnn/build_graph_snapshot_dataset.py

Required inputs (task 2.7):
    data/processed/model_input/node_order.json
    data/processed/model_input/edge_index.npy
    data/processed/model_input/X.npy
    data/processed/model_input/schema.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_INPUT_DIR = REPO_ROOT / "data" / "processed" / "model_input"


class Phase2DatasetError(FileNotFoundError):
    """Raised when a required task 2.7 artifact is missing or malformed.
    Matches the *ArtifactError/*SchemaError convention used across
    src/graph, src/features, and build_model_input_schema.py."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase2DatasetError(
            f"Required artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first."
        )
    return path


class GraphSnapshotDataset(Dataset):
    """One item per flood phase: Data(x=[N,F], edge_index=[2,E], y=[N,1] or
    None). edge_index is the SAME static line-graph adjacency for every
    phase.
    """

    def __init__(self, X: np.ndarray, edge_index: np.ndarray, node_order: list, phases: list, schema: dict):
        if X.ndim != 3:
            raise Phase2DatasetError(f"X must be (T, N, F), got shape {X.shape}.")
        T, N, F = X.shape
        if len(node_order) != N:
            raise Phase2DatasetError(f"node_order has {len(node_order)} entries but X has N={N}.")
        if len(phases) != T:
            raise Phase2DatasetError(f"{len(phases)} phase(s) given but X has T={T}.")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise Phase2DatasetError(f"edge_index must be shape (2, E), got {edge_index.shape}.")
        max_idx = int(edge_index.max()) if edge_index.size else -1
        if max_idx >= N:
            raise Phase2DatasetError(f"edge_index references node position {max_idx}, but only {N} nodes exist.")

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        self.node_order = list(node_order)
        self.phases = list(phases)  # [{"phase_id": int, "phase_name": str}, ...], same order as X's T axis
        self.schema = schema
        self._labels = None  # set via attach_labels(): dict {phase_id: FloatTensor[N,1]}

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, t: int) -> Data:
        phase = self.phases[t]
        y = self._labels.get(phase["phase_id"]) if self._labels else None
        return Data(x=self.X[t], edge_index=self.edge_index, y=y, phase_id=phase["phase_id"], phase_name=phase["phase_name"])

    @property
    def num_nodes(self) -> int:
        return self.X.shape[1]

    @property
    def num_features(self) -> int:
        return self.X.shape[2]

    @property
    def num_periods(self) -> int:
        return self.X.shape[0]

    # ----------------------------------------------------------------
    # Labels -- task 3.4 attaches these once fused ground truth exists.
    # ----------------------------------------------------------------

    def attach_labels(self, labels_df) -> None:
        """labels_df: DataFrame with columns segment_id, phase_id,
        flood_label (binary 0/1) -- task 2.7's schema.json y_t1_contract
        join key. Validates full node coverage for every "usable
        transition" target phase_id in the contract before accepting; a
        partial/malformed label set raises rather than attaching
        silently-incomplete data.
        """
        required_cols = {"segment_id", "phase_id", "flood_label"}
        missing_cols = required_cols - set(labels_df.columns)
        if missing_cols:
            raise Phase2DatasetError(f"labels_df is missing required column(s) {sorted(missing_cols)}.")

        needed_phase_ids = {t["y_t1_phase_id"] for t in self.schema["y_t1_contract"]["usable_transitions"]}

        labels = {}
        for phase_id in needed_phase_ids:
            rows = labels_df[labels_df["phase_id"] == phase_id].set_index("segment_id")
            missing_segments = [s for s in self.node_order if s not in rows.index]
            if missing_segments:
                raise Phase2DatasetError(
                    f"phase_id {phase_id}: labels_df is missing {len(missing_segments)} segment(s) "
                    f"(e.g. {missing_segments[:5]}) -- every node needs a label per the y_t1_contract."
                )
            values = rows.loc[self.node_order, "flood_label"].to_numpy(copy=True)
            if not np.isin(values, [0, 1]).all():
                raise Phase2DatasetError(f"phase_id {phase_id}: flood_label must be binary (0/1).")
            labels[phase_id] = torch.as_tensor(values, dtype=torch.float32).unsqueeze(1)

        self._labels = labels

    def transition_pairs(self):
        """Yield (X_t snapshot, Y_{t+1} tensor) for every usable transition
        in schema.json's y_t1_contract. Raises if labels haven't been
        attached yet -- see attach_labels()."""
        if self._labels is None:
            raise Phase2DatasetError("No labels attached -- call attach_labels() first (task 3.4's output).")
        phase_pos = {p["phase_id"]: t for t, p in enumerate(self.phases)}
        for transition in self.schema["y_t1_contract"]["usable_transitions"]:
            t = phase_pos[transition["x_t_phase_id"]]
            yield self[t], self._labels[transition["y_t1_phase_id"]]

    # ----------------------------------------------------------------
    # Architecture-specific views -- exact conventions from
    # toy_gnn_prototype.py (task 1.12), reproduced here so a real training
    # script doesn't have to re-derive them.
    # ----------------------------------------------------------------

    def to_a3tgcn_input(self):
        """(N, F, T) tensor + static edge_index -- A3TGCN's
        forward(X, edge_index) signature."""
        X_nft = self.X.permute(1, 2, 0).contiguous()  # (T,N,F) -> (N,F,T)
        return X_nft, self.edge_index

    def to_mpnn_lstm_input(self):
        """(T*N, F) tensor (timesteps stacked row-wise, t-major) + edge_index
        tiled once per timestep with node ids offset by t*N + unit
        edge_weight -- MPNNLSTM's forward(X, edge_index, edge_weight)
        signature. See toy_gnn_prototype.py's test_mpnn_lstm() docstring for
        why getting this tiling wrong silently mixes the wrong nodes'
        features rather than raising a shape error.
        """
        T, N, F = self.X.shape
        X_tnf = self.X.reshape(T * N, F)
        edge_index_batched = torch.cat([self.edge_index + t * N for t in range(T)], dim=1)
        edge_weight = torch.ones(edge_index_batched.size(1), dtype=torch.float32)
        return X_tnf, edge_index_batched, edge_weight


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_dataset(model_input_dir: Path = MODEL_INPUT_DIR) -> GraphSnapshotDataset:
    """Load task 2.7's outputs from `model_input_dir` and build a
    GraphSnapshotDataset over them."""
    node_order_path = model_input_dir / "node_order.json"
    edge_index_path = model_input_dir / "edge_index.npy"
    X_path = model_input_dir / "X.npy"
    schema_path = model_input_dir / "schema.json"

    produced_by = "src/models/gnn/build_model_input_schema.py (task 2.7)"
    _require_file(node_order_path, produced_by)
    _require_file(edge_index_path, produced_by)
    _require_file(X_path, produced_by)
    _require_file(schema_path, produced_by)

    node_order = json.loads(node_order_path.read_text())
    edge_index = np.load(edge_index_path)
    X = np.load(X_path)
    schema = json.loads(schema_path.read_text())

    return GraphSnapshotDataset(X=X, edge_index=edge_index, node_order=node_order, phases=schema["phases"], schema=schema)


# --------------------------------------------------------------------------
# CLI entry point (smoke test against real data)
# --------------------------------------------------------------------------

def main():
    print(f"Loading task 2.7 model input schema <- {MODEL_INPUT_DIR}")
    try:
        ds = load_dataset()
    except Phase2DatasetError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> {len(ds)} phase snapshots, {ds.num_nodes} nodes, {ds.num_features} features")

    print("\nPer-phase snapshot check:")
    for t in range(len(ds)):
        d = ds[t]
        print(f"  phase {d.phase_id} ({d.phase_name}): x={tuple(d.x.shape)}  edge_index={tuple(d.edge_index.shape)}  y={d.y}")

    X_a3tgcn, ei = ds.to_a3tgcn_input()
    print(f"\nA3TGCN input:    X={tuple(X_a3tgcn.shape)} (N,F,T)  edge_index={tuple(ei.shape)}")

    X_mpnn, ei_batched, ew = ds.to_mpnn_lstm_input()
    print(f"MPNN-LSTM input: X={tuple(X_mpnn.shape)} (T*N,F)  edge_index={tuple(ei_batched.shape)}  edge_weight={tuple(ew.shape)}")

    print("\nNo labels attached yet -- call attach_labels() once task 3.4's fused ground truth exists.")
    print("Done. Next: task 3.6 (baseline model) and task 3.7 (training harness) consume this class;")
    print("task 3.4 must produce a labels_df satisfying schema.json's y_t1_contract for attach_labels().")


if __name__ == "__main__":
    main()
