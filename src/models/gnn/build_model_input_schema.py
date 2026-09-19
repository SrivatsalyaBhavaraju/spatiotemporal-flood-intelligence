"""
Task 2.7 -- define and materialize the model input schema (working.md SS1.4):
X_t [N x F] per phase, edge_index [2 x E], and the Y_{t+1} [N x 1] contract
that task 3.4's fused labels must satisfy. This is the handoff point from
Phase 2 (graph + features) to Phase 3/4 (ground truth, training).

Node ordering / edge_index: reused directly from task 2.2's build_line_graph.py
node_order()/edge_index_array() -- that module's own docstring names this as
the contract task 2.7 should build on, so X_t's rows and edge_index positions
stay aligned to the SAME canonical segment_id ordering rather than each
script re-deriving its own. edge_index stays DIRECTED (task 2.2's
traversal-direction adjacency, respecting OSM oneway tags) -- not
symmetrized here, since that module's docstring documents this as a
deliberate choice, not an oversight; a two-way street already produces
edges in both directions on its own.

Feature set (F=6): elevation_m, slope_deg, length_m, distance_to_drain_m
(task 2.3, static) + rainfall_t, cumulative_rainfall_t (task 2.6, dynamic,
broadcast identically to every segment within a phase -- see that script's
docstring). Order matches working.md SS1.3's node-feature list.

Note on working.md line 90 ("F = 7 features"): that line counts SS1.7's
MUST-HAVE list literally, which bundles "road topology" (structural --
edge_index, not a row in X_t) and "flood label" (= Y_{t+1}, not a row in
X_t) in with the 5 real per-node scalars, and doesn't itemize length_m
separately even though SS1.3's own node-feature list includes it. Net: F=6
real per-node feature columns, not 7 -- a cosmetic doc mismatch, not a
functional gap. Flagged here rather than silently "corrected" upstream.

Timesteps (T): the 4 phases from task 2.5 (pre_event=0, rising=1, peak=2,
receding=3), read from task 2.6's own phase_id/phase_name columns rather
than hardcoded here, so a phase-boundary edit in
build_rainfall_phase_features.py propagates automatically.

Y_{t+1} -- NOT produced by this script. Task 3.4 (ground truth fusion)
hasn't run yet as of this task, so there is no real label data to write.
This script documents the CONTRACT (schema.json's "y_t1_contract" block)
that 3.4's output must satisfy to plug into task 2.8's data loader: one
binary flood label per segment_id per phase, joined the same way every
other feature table in this repo joins -- a plain key match on
(segment_id, phase_id). With 4 phases, (X_t, Y_{t+1}) is only defined for
t in {0,1,2} (receding, phase 3, has no "next" phase) -- 3 usable training
transitions, matching working.md SS1.4/SS1.6.

Usage:
    python src/models/gnn/build_model_input_schema.py

Required inputs:
    data/processed/graph/line_graph.gpickle          (task 2.2)
    data/processed/graph/static_features.geojson      (task 2.3)
    data/processed/features/dynamic_node_features.csv (task 2.6)

Outputs (data/processed/model_input/):
    node_order.json       -- canonical segment_id ordering, length N. Position
                              i is X[:, i, :]'s row and edge_index's node id i.
    edge_index.npy         -- int64 (2, E) array, PyTorch-Geometric convention,
                               directed (see module docstring above).
    X.npy                   -- float32 (T, N, F) array, X[t] is X_t.
    schema.json              -- feature names/order, shapes, phase_id -> name
                                 map, and the Y_{t+1} contract for task 2.8/3.4.
    schema_validation_report.json -- validation report (see validate_schema()).

Handoff for 2.8/3.4: task 2.8's data loader reshapes X/edge_index into
whatever a specific architecture needs (A3TGCN vs MPNN-LSTM have different
conventions -- see src/models/gnn/README.md's note on toy_gnn_prototype.py);
this script only produces and validates the canonical (T, N, F) form. Task
3.4 must produce labels satisfying schema.json's y_t1_contract before
training (task 4.1) can start.
"""
import json
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.graph.build_line_graph import edge_index_array, node_order  # noqa: E402

PROCESSED_GRAPH_DIR = REPO_ROOT / "data" / "processed" / "graph"
PROCESSED_FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"
OUT_DIR = REPO_ROOT / "data" / "processed" / "model_input"

LINE_GRAPH_GPICKLE = PROCESSED_GRAPH_DIR / "line_graph.gpickle"
STATIC_FEATURES = PROCESSED_GRAPH_DIR / "static_features.geojson"
DYNAMIC_FEATURES = PROCESSED_FEATURES_DIR / "dynamic_node_features.csv"

# Canonical column order for X_t -- matches working.md SS1.3's node-feature
# list (elevation, slope, length, distance_to_drain, then the two dynamic
# rainfall features).
FEATURE_COLUMNS = [
    "elevation_m",
    "slope_deg",
    "length_m",
    "distance_to_drain_m",
    "rainfall_t",
    "cumulative_rainfall_t",
]

STATIC_COLUMNS = ["elevation_m", "slope_deg", "length_m", "distance_to_drain_m"]
DYNAMIC_COLUMNS = ["rainfall_t", "cumulative_rainfall_t"]


class Phase2SchemaError(FileNotFoundError):
    """Raised when a required task 2.2/2.3/2.6 artifact is missing or
    malformed. Always names the exact expected path and which upstream
    script produces it, matching the convention used across src/graph and
    src/features."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase2SchemaError(
            f"Required artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first."
        )
    return path


# --------------------------------------------------------------------------
# Loading inputs
# --------------------------------------------------------------------------

def load_node_order_and_edge_index(path: Path = LINE_GRAPH_GPICKLE):
    """Load task 2.2's line graph and return (order, edge_index) using that
    module's own node_order()/edge_index_array() -- the canonical mapping
    task 2.2's docstring names as this task's contract.
    """
    _require_file(path, "src/graph/build_line_graph.py (task 2.2)")
    try:
        with open(path, "rb") as f:
            L = pickle.load(f)
    except Exception as e:
        raise Phase2SchemaError(f"Failed to unpickle {path}: {e}") from e

    order = node_order(L)
    if not order:
        raise Phase2SchemaError(f"{path} unpickled but its line graph has zero nodes.")
    edge_index = np.array(edge_index_array(L, order), dtype=np.int64)
    return order, edge_index


def load_static_features(path: Path = STATIC_FEATURES) -> pd.DataFrame:
    _require_file(path, "src/graph/build_static_features.py (task 2.3)")
    gdf = gpd.read_file(path)
    missing = {"segment_id", *STATIC_COLUMNS} - set(gdf.columns)
    if missing:
        raise Phase2SchemaError(f"{path} is missing required column(s) {sorted(missing)}.")
    return pd.DataFrame(gdf.drop(columns="geometry"))


def load_dynamic_features(path: Path = DYNAMIC_FEATURES) -> pd.DataFrame:
    _require_file(path, "src/features/attach_dynamic_node_features.py (task 2.6)")
    df = pd.read_csv(path)
    missing = {"segment_id", "phase_id", "phase_name", *DYNAMIC_COLUMNS} - set(df.columns)
    if missing:
        raise Phase2SchemaError(f"{path} is missing required column(s) {sorted(missing)}.")
    return df


# --------------------------------------------------------------------------
# Building X_t
# --------------------------------------------------------------------------

def phase_table(dynamic_df: pd.DataFrame) -> pd.DataFrame:
    """One row per phase (phase_id, phase_name), sorted -- read from task
    2.6's own output rather than hardcoded, so a phase-boundary edit in
    task 2.5 propagates automatically."""
    phases = dynamic_df[["phase_id", "phase_name"]].drop_duplicates().sort_values("phase_id")
    return phases.reset_index(drop=True)


def build_X(static_df: pd.DataFrame, dynamic_df: pd.DataFrame, order: list, phases: pd.DataFrame) -> np.ndarray:
    """Return X with shape (T, N, F): X[t] is X_t, row i is segment
    `order[i]`, columns are FEATURE_COLUMNS. A segment/phase missing from an
    input table fails loudly rather than silently producing a NaN row a
    downstream loader might not notice.
    """
    static_indexed = static_df.set_index("segment_id")
    missing_static = [s for s in order if s not in static_indexed.index]
    if missing_static:
        raise Phase2SchemaError(
            f"{len(missing_static)} line-graph segment(s) have no task 2.3 static features "
            f"(e.g. {missing_static[:5]}). Re-run task 2.3, or check for a stale file."
        )
    static_block = static_indexed.loc[order, STATIC_COLUMNS].to_numpy(dtype=np.float32)

    T, N, F = len(phases), len(order), len(FEATURE_COLUMNS)
    X = np.empty((T, N, F), dtype=np.float32)

    for t, (_, phase_row) in enumerate(phases.iterrows()):
        phase_dyn = dynamic_df[dynamic_df["phase_id"] == phase_row["phase_id"]].set_index("segment_id")
        missing_dyn = [s for s in order if s not in phase_dyn.index]
        if missing_dyn:
            raise Phase2SchemaError(
                f"Phase {phase_row['phase_id']} ({phase_row['phase_name']}) is missing task 2.6 dynamic "
                f"features for {len(missing_dyn)} segment(s) (e.g. {missing_dyn[:5]})."
            )
        dyn_block = phase_dyn.loc[order, DYNAMIC_COLUMNS].to_numpy(dtype=np.float32)
        X[t] = np.concatenate([static_block, dyn_block], axis=1)

    return X


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_schema(X: np.ndarray, edge_index: np.ndarray, order: list, phases: pd.DataFrame) -> dict:
    T, N, F = X.shape
    report = {
        "shapes": {"X": list(X.shape), "edge_index": list(edge_index.shape), "N_nodes": N, "F_features": F, "T_phases": T},
        "feature_columns": FEATURE_COLUMNS,
        "phases": phases.to_dict(orient="records"),
    }

    nan_mask = np.isnan(X)
    report["missing_values"] = {
        "total_nan_cells": int(nan_mask.sum()),
        "by_feature": {col: int(nan_mask[:, :, i].sum()) for i, col in enumerate(FEATURE_COLUMNS)},
    }

    if edge_index.size:
        lo, hi = int(edge_index.min()), int(edge_index.max())
    else:
        lo, hi = None, None
    report["edge_index_bounds"] = {"min": lo, "max": hi, "in_range": hi is None or (0 <= lo and hi < N)}
    if hi is not None and not (0 <= lo and hi < N):
        raise Phase2SchemaError(
            f"edge_index references node position {hi} (or {lo}), but only {N} nodes exist in node_order."
        )

    report["y_t1_contract"] = {
        "note": "Not produced by this script -- see task 3.4. Shape/join-key contract documented here for 2.8/3.4 to satisfy.",
        "shape_per_usable_timestep": [N, 1],
        "dtype": "binary (0/1)",
        "usable_transitions": [
            {"x_t_phase_id": int(phases.iloc[i]["phase_id"]), "y_t1_phase_id": int(phases.iloc[i + 1]["phase_id"])}
            for i in range(len(phases) - 1)
        ],
        "join_key": "segment_id, phase_id -- same convention as static_features.geojson / dynamic_node_features.csv",
    }
    return report


# --------------------------------------------------------------------------
# Saving outputs
# --------------------------------------------------------------------------

def save_outputs(order: list, edge_index: np.ndarray, X: np.ndarray, out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    node_order_path = out_dir / "node_order.json"
    node_order_path.write_text(json.dumps(order, indent=2), encoding="utf-8")

    edge_index_path = out_dir / "edge_index.npy"
    np.save(edge_index_path, edge_index)

    X_path = out_dir / "X.npy"
    np.save(X_path, X)

    return {"node_order_json": node_order_path, "edge_index_npy": edge_index_path, "X_npy": X_path}


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    print(f"Loading task 2.2 line graph <- {LINE_GRAPH_GPICKLE}")
    try:
        order, edge_index = load_node_order_and_edge_index()
    except Phase2SchemaError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> {len(order)} nodes (segments), edge_index shape {tuple(edge_index.shape)}")

    print(f"\nLoading task 2.3 static features <- {STATIC_FEATURES}")
    try:
        static_df = load_static_features()
    except Phase2SchemaError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Loading task 2.6 dynamic features <- {DYNAMIC_FEATURES}")
    try:
        dynamic_df = load_dynamic_features()
    except Phase2SchemaError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    phases = phase_table(dynamic_df)
    print(f"  -> {len(phases)} phases: {phases['phase_name'].tolist()}")

    print("\nBuilding X_t for every phase ...")
    try:
        X = build_X(static_df, dynamic_df, order, phases)
    except Phase2SchemaError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> X shape {X.shape}  (T, N, F)")

    print("\nSaving outputs ...")
    paths = save_outputs(order, edge_index, X)
    for label, p in paths.items():
        print(f"  {label}: {p}")

    print("\nRunning validation ...")
    try:
        report = validate_schema(X, edge_index, order, phases)
    except Phase2SchemaError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    schema = {
        "N_nodes": len(order),
        "F_features": len(FEATURE_COLUMNS),
        "T_phases": len(phases),
        "feature_columns": FEATURE_COLUMNS,
        "phases": phases.to_dict(orient="records"),
        "edge_index_directed": True,
        "edge_index_convention": "PyTorch-Geometric (2, E): [source_positions, target_positions]",
        "y_t1_contract": report["y_t1_contract"],
    }
    schema_path = OUT_DIR / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2, default=str), encoding="utf-8")
    print(f"  schema -> {schema_path}")

    report_path = OUT_DIR / "schema_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  validation report -> {report_path}")

    print("\n=== Validation summary ===")
    print(json.dumps(report, indent=2, default=str))

    print("\nDone. Next: task 2.8 (data loader) consumes node_order.json/edge_index.npy/X.npy;")
    print("task 3.4 (label fusion) must satisfy schema.json's y_t1_contract before training (task 4.1).")


if __name__ == "__main__":
    main()
