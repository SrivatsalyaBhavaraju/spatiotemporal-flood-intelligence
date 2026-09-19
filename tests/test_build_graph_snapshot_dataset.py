"""
Unit tests for src/models/gnn/build_graph_snapshot_dataset.py (task 2.8).

Builds a tiny synthetic task 2.7-shaped model_input directory (X.npy,
edge_index.npy, node_order.json, schema.json) directly, so the dataset
class can be tested without needing real Phase 2 data on disk. Run against
the real pipeline via `python src/models/gnn/build_graph_snapshot_dataset.py`.

Run with:
    pytest tests/test_build_graph_snapshot_dataset.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.gnn.build_graph_snapshot_dataset import (  # noqa: E402
    GraphSnapshotDataset,
    Phase2DatasetError,
    load_dataset,
)

SEGMENTS = ["segA", "segB", "segC"]
PHASES = [{"phase_id": 0, "phase_name": "pre_event"}, {"phase_id": 1, "phase_name": "rising"}, {"phase_id": 2, "phase_name": "peak"}]
N, T, F = len(SEGMENTS), len(PHASES), 4


def _make_X() -> np.ndarray:
    # deterministic values so per-cell checks are easy: X[t, n, f] = t*100 + n*10 + f
    X = np.empty((T, N, F), dtype=np.float32)
    for t in range(T):
        for n in range(N):
            for f in range(F):
                X[t, n, f] = t * 100 + n * 10 + f
    return X


def _make_edge_index() -> np.ndarray:
    return np.array([[0, 1], [1, 2]], dtype=np.int64)  # segA->segB, segB->segC


def _make_schema() -> dict:
    return {
        "N_nodes": N, "F_features": F, "T_phases": T,
        "feature_columns": ["f0", "f1", "f2", "f3"],
        "phases": PHASES,
        "y_t1_contract": {
            "shape_per_usable_timestep": [N, 1],
            "dtype": "binary (0/1)",
            "usable_transitions": [
                {"x_t_phase_id": 0, "y_t1_phase_id": 1},
                {"x_t_phase_id": 1, "y_t1_phase_id": 2},
            ],
            "join_key": "segment_id, phase_id",
        },
    }


def _make_dataset() -> GraphSnapshotDataset:
    return GraphSnapshotDataset(X=_make_X(), edge_index=_make_edge_index(), node_order=SEGMENTS, phases=PHASES, schema=_make_schema())


def _make_labels_df(segments=SEGMENTS, phase_ids=(1, 2)) -> pd.DataFrame:
    rows = []
    for phase_id in phase_ids:
        for i, seg in enumerate(segments):
            rows.append({"segment_id": seg, "phase_id": phase_id, "flood_label": i % 2})
    return pd.DataFrame(rows)


class TestConstruction:
    def test_shape_mismatch_raises(self):
        with pytest.raises(Phase2DatasetError, match="node_order"):
            GraphSnapshotDataset(X=_make_X(), edge_index=_make_edge_index(), node_order=["segA"], phases=PHASES, schema=_make_schema())

    def test_wrong_x_ndim_raises(self):
        with pytest.raises(Phase2DatasetError, match=r"\(T, N, F\)"):
            GraphSnapshotDataset(X=np.zeros((N, F)), edge_index=_make_edge_index(), node_order=SEGMENTS, phases=PHASES, schema=_make_schema())

    def test_out_of_range_edge_index_raises(self):
        bad_edge_index = np.array([[0], [5]], dtype=np.int64)
        with pytest.raises(Phase2DatasetError, match="node position"):
            GraphSnapshotDataset(X=_make_X(), edge_index=bad_edge_index, node_order=SEGMENTS, phases=PHASES, schema=_make_schema())


class TestIndexing:
    def test_len_equals_num_phases(self):
        ds = _make_dataset()
        assert len(ds) == T

    def test_getitem_returns_correct_snapshot(self):
        ds = _make_dataset()
        d = ds[1]
        assert d.phase_id == 1
        assert d.phase_name == "rising"
        assert tuple(d.x.shape) == (N, F)
        assert torch.equal(d.x, torch.as_tensor(_make_X()[1]))
        assert torch.equal(d.edge_index, torch.as_tensor(_make_edge_index(), dtype=torch.long))
        assert d.y is None  # no labels attached yet

    def test_properties(self):
        ds = _make_dataset()
        assert ds.num_nodes == N
        assert ds.num_features == F
        assert ds.num_periods == T


class TestAttachLabels:
    def test_missing_column_raises(self):
        ds = _make_dataset()
        bad_df = pd.DataFrame({"segment_id": SEGMENTS, "phase_id": [1, 1, 1]})  # no flood_label
        with pytest.raises(Phase2DatasetError, match="flood_label"):
            ds.attach_labels(bad_df)

    def test_missing_segment_raises(self):
        ds = _make_dataset()
        incomplete_df = _make_labels_df(segments=["segA", "segB"])  # segC missing
        with pytest.raises(Phase2DatasetError, match="segC"):
            ds.attach_labels(incomplete_df)

    def test_non_binary_label_raises(self):
        ds = _make_dataset()
        df = _make_labels_df()
        df.loc[df["segment_id"] == "segA", "flood_label"] = 2
        with pytest.raises(Phase2DatasetError, match="binary"):
            ds.attach_labels(df)

    def test_valid_labels_attach_and_populate_y(self):
        ds = _make_dataset()
        ds.attach_labels(_make_labels_df())
        d = ds[1]  # phase_id 1 is a target (y_t1_phase_id) of transition 0
        assert d.y is not None
        assert tuple(d.y.shape) == (N, 1)

        d0 = ds[0]  # phase_id 0 is never a target -- no label for it
        assert d0.y is None


class TestTransitionPairs:
    def test_raises_before_labels_attached(self):
        ds = _make_dataset()
        with pytest.raises(Phase2DatasetError, match="attach_labels"):
            list(ds.transition_pairs())

    def test_yields_correct_pairs_after_attach(self):
        ds = _make_dataset()
        ds.attach_labels(_make_labels_df())
        pairs = list(ds.transition_pairs())
        assert len(pairs) == 2  # 2 usable transitions
        x_snapshot, y = pairs[0]
        assert x_snapshot.phase_id == 0
        assert tuple(y.shape) == (N, 1)


class TestArchitectureViews:
    def test_a3tgcn_shape_and_values(self):
        ds = _make_dataset()
        X_nft, edge_index = ds.to_a3tgcn_input()
        assert tuple(X_nft.shape) == (N, F, T)
        assert torch.equal(edge_index, ds.edge_index)
        # spot check: X_nft[n, f, t] should equal original X[t, n, f]
        assert X_nft[1, 2, 0].item() == pytest.approx(0 * 100 + 1 * 10 + 2)
        assert X_nft[2, 3, 1].item() == pytest.approx(1 * 100 + 2 * 10 + 3)

    def test_mpnn_lstm_shape_and_tiling(self):
        ds = _make_dataset()
        X_tnf, edge_index_batched, edge_weight = ds.to_mpnn_lstm_input()
        assert tuple(X_tnf.shape) == (T * N, F)
        assert tuple(edge_index_batched.shape) == (2, ds.edge_index.shape[1] * T)
        assert edge_weight.shape[0] == edge_index_batched.shape[1]
        assert torch.all(edge_weight == 1.0)

        # row-major stacking: rows [0:N) are phase 0, [N:2N) are phase 1, etc.
        assert torch.equal(X_tnf[0:N], torch.as_tensor(_make_X()[0]))
        assert torch.equal(X_tnf[N:2 * N], torch.as_tensor(_make_X()[1]))

        # edge_index offset by t*N per timestep block
        E = ds.edge_index.shape[1]
        assert torch.equal(edge_index_batched[:, E:2 * E], ds.edge_index + N)


class TestLoadDataset:
    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(Phase2DatasetError, match="build_model_input_schema.py"):
            load_dataset(tmp_path)

    def test_round_trip(self, tmp_path):
        (tmp_path / "node_order.json").write_text(json.dumps(SEGMENTS), encoding="utf-8")
        np.save(tmp_path / "edge_index.npy", _make_edge_index())
        np.save(tmp_path / "X.npy", _make_X())
        (tmp_path / "schema.json").write_text(json.dumps(_make_schema()), encoding="utf-8")

        ds = load_dataset(tmp_path)
        assert len(ds) == T
        assert ds.num_nodes == N
        assert ds.node_order == SEGMENTS
