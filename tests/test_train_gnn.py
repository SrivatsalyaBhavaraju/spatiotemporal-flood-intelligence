"""
Unit tests for src/models/gnn/train_gnn.py (task 4.1).

Covers the pure logic (pos_weight, split-mask loading) with synthetic
data, plus a REAL end-to-end training run on a tiny synthetic graph (a
few nodes, few epochs) -- fast, but genuinely exercises the training loop
and A3TGCN forward/backward pass, not mocked. The full real-data training
run (17,195 nodes) is validated separately (see developing.md's task 4.1
notes).

Run with:
    pytest tests/test_train_gnn.py -v
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

from src.models.gnn.build_graph_snapshot_dataset import GraphSnapshotDataset  # noqa: E402
from src.models.gnn.train_gnn import (  # noqa: E402
    Phase4TrainingError,
    compute_feature_stats,
    compute_pos_weight,
    evaluate_model,
    load_split_masks,
    normalize_features,
    train_model,
)

SEGMENTS = ["s1", "s2", "s3", "s4"]
PHASES = [{"phase_id": 0, "phase_name": "pre_event"}, {"phase_id": 1, "phase_name": "rising"}, {"phase_id": 2, "phase_name": "peak"}]


def _make_schema():
    return {
        "y_t1_contract": {
            "usable_transitions": [
                {"x_t_phase_id": 0, "y_t1_phase_id": 1},
                {"x_t_phase_id": 1, "y_t1_phase_id": 2},
            ],
        },
    }


def _make_tiny_dataset():
    T, N, F = 3, len(SEGMENTS), 2
    rng = np.random.default_rng(0)
    X = rng.normal(size=(T, N, F)).astype(np.float32)
    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)  # a small chain
    ds = GraphSnapshotDataset(X=X, edge_index=edge_index, node_order=SEGMENTS, phases=PHASES, schema=_make_schema())
    labels_df = pd.DataFrame({
        "segment_id": SEGMENTS * 2,
        "phase_id": [1] * 4 + [2] * 4,
        "flood_label": [0, 1, 0, 1, 1, 1, 0, 0],
    })
    ds.attach_labels(labels_df)
    return ds


class TestComputePosWeight:
    def test_balanced_classes_gives_weight_one(self):
        y = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert compute_pos_weight(y) == pytest.approx(1.0)

    def test_imbalanced_favors_minority_class(self):
        y = torch.tensor([1.0, 0.0, 0.0, 0.0])  # 1 positive, 3 negative
        assert compute_pos_weight(y) == pytest.approx(3.0)

    def test_no_positive_examples_defaults_to_one(self):
        y = torch.tensor([0.0, 0.0, 0.0])
        assert compute_pos_weight(y) == pytest.approx(1.0)


class TestComputeFeatureStats:
    def test_stats_computed_only_from_train_split(self):
        ds = _make_tiny_dataset()
        train_mask = torch.tensor([True, True, False, False])
        mean, std = compute_feature_stats(ds, train_mask)

        expected = ds.X[:, train_mask, :].reshape(-1, ds.X.shape[-1])
        assert torch.allclose(mean, expected.mean(dim=0))
        assert torch.allclose(std, expected.std(dim=0))

    def test_constant_feature_gets_std_one_not_zero(self):
        T, N, F = 2, 3, 1
        X = np.zeros((T, N, F), dtype=np.float32)  # every value identical -> std would be 0
        edge_index = np.array([[0], [1]], dtype=np.int64)
        ds = GraphSnapshotDataset(X=X, edge_index=edge_index, node_order=["a", "b", "c"],
                                   phases=[{"phase_id": 0, "phase_name": "p0"}, {"phase_id": 1, "phase_name": "p1"}],
                                   schema={"y_t1_contract": {"usable_transitions": []}})
        train_mask = torch.tensor([True, True, False])
        mean, std = compute_feature_stats(ds, train_mask)
        assert std[0] == 1.0  # guarded, not 0 (which would divide-by-zero in normalize_features)


class TestNormalizeFeatures:
    def test_zero_mean_unit_variance_after_normalizing_the_source_data(self):
        x = torch.tensor([[1.0], [2.0], [3.0]])
        mean, std = x.mean(dim=0), x.std(dim=0)
        normalized = normalize_features(x, mean, std)
        assert torch.allclose(normalized.mean(dim=0), torch.zeros(1), atol=1e-6)

    def test_different_scale_features_end_up_comparable(self):
        # feature 0 spans 0-1000, feature 1 spans 0-1 -- normalization should
        # bring both to a comparable range, unlike the raw values
        x = torch.tensor([[0.0, 0.0], [1000.0, 1.0]])
        mean, std = x.mean(dim=0), x.std(dim=0)
        normalized = normalize_features(x, mean, std)
        assert normalized[:, 0].abs().max() == pytest.approx(normalized[:, 1].abs().max(), abs=1e-5)


class TestLoadSplitMasks:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Phase4TrainingError, match="build_training_harness.py"):
            load_split_masks(SEGMENTS, path=tmp_path / "nope.csv")

    def test_masks_align_to_node_order(self, tmp_path):
        p = tmp_path / "splits.csv"
        pd.DataFrame({
            "segment_id": ["s2", "s1", "s4", "s3"], "ward_no": [1, 1, 2, 2],
            "split": ["val", "train", "test", "train"],
        }).to_csv(p, index=False)
        masks = load_split_masks(SEGMENTS, path=p)  # SEGMENTS = [s1,s2,s3,s4]
        assert masks["train"].tolist() == [True, False, True, False]
        assert masks["val"].tolist() == [False, True, False, False]
        assert masks["test"].tolist() == [False, False, False, True]


class TestTrainAndEvaluateEndToEnd:
    def test_real_forward_backward_pass_on_tiny_graph(self):
        ds = _make_tiny_dataset()
        train_mask = torch.tensor([True, True, False, False])  # s1, s2 train; s3, s4 held out

        model, curve, feature_stats = train_model(ds, train_mask, epochs=5, device=torch.device("cpu"))

        assert len(curve) == 5
        assert all("loss" in c for c in curve)
        # loss values are finite real numbers, not NaN -- proves the backward pass didn't blow up
        assert all(np.isfinite(c["loss"]) for c in curve)
        mean, std = feature_stats
        assert mean.shape == (ds.num_features,)
        assert std.shape == (ds.num_features,)

    def test_evaluate_model_produces_metrics_per_transition_and_split(self):
        ds = _make_tiny_dataset()
        train_mask = torch.tensor([True, True, False, False])
        val_mask = torch.tensor([False, False, True, False])
        test_mask = torch.tensor([False, False, False, True])
        split_masks = {"train": train_mask, "val": val_mask, "test": test_mask}

        model, _, feature_stats = train_model(ds, train_mask, epochs=3, device=torch.device("cpu"))
        phase_id_to_name = {p["phase_id"]: p["phase_name"] for p in ds.phases}
        report = evaluate_model(model, ds, split_masks, phase_id_to_name, feature_stats, device=torch.device("cpu"))

        assert set(report.keys()) == {"pre_event->rising", "rising->peak"}
        for label, per_split in report.items():
            assert set(per_split.keys()) == {"train", "val", "test"}
            for split_metrics in per_split.values():
                assert "f1" in split_metrics and "accuracy" in split_metrics
