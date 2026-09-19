"""
Unit tests for src/models/gnn/tune_hyperparameters.py (task 4.2).

Covers the pure fold-construction/aggregation/threshold-sweep logic with
small synthetic data. The actual cross-validated training loop reuses
task 4.1's own train_model()/evaluate_model() (already tested in
test_train_gnn.py) and is validated end-to-end by a real run against real
data (see developing.md's task 4.2 notes).

Run with:
    pytest tests/test_tune_hyperparameters.py -v
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.gnn.tune_hyperparameters import (  # noqa: E402
    aggregate_fold_metrics,
    build_kfold_ward_groups,
    build_mask_for_segments,
    mean_flood_relevant_f1,
    sweep_threshold,
)


class TestBuildKfoldWardGroups:
    def test_all_wards_assigned_exactly_once(self):
        ward_sizes = {i: 100 for i in range(1, 13)}
        groups = build_kfold_ward_groups(ward_sizes, k=4)
        all_wards = [w for g in groups for w in g]
        assert sorted(all_wards) == list(range(1, 13))
        assert len(all_wards) == len(set(all_wards))  # no duplicates

    def test_produces_k_groups(self):
        ward_sizes = {i: 100 for i in range(1, 13)}
        groups = build_kfold_ward_groups(ward_sizes, k=4)
        assert len(groups) == 4

    def test_balanced_by_segment_count(self):
        ward_sizes = {i: 100 for i in range(1, 13)}  # 12 equal wards, 4 folds -> 3 each, 300 segments each
        groups = build_kfold_ward_groups(ward_sizes, k=4)
        totals = [sum(ward_sizes[w] for w in g) for g in groups]
        assert all(t == 300 for t in totals)

    def test_deterministic_with_fixed_seed(self):
        ward_sizes = {i: 100 for i in range(1, 13)}
        g1 = build_kfold_ward_groups(ward_sizes, k=4, seed=42)
        g2 = build_kfold_ward_groups(ward_sizes, k=4, seed=42)
        assert g1 == g2

    def test_handles_uneven_ward_sizes(self):
        ward_sizes = {1: 1000, 2: 10, 3: 10, 4: 10}
        groups = build_kfold_ward_groups(ward_sizes, k=2)
        # the single huge ward should end up alone in one fold, not paired with others
        totals = sorted(sum(ward_sizes[w] for w in g) for g in groups)
        assert totals[1] >= 1000  # the fold containing ward 1


class TestBuildMaskForSegments:
    def test_correct_boolean_mask(self):
        node_order = ["a", "b", "c", "d"]
        mask = build_mask_for_segments(node_order, {"b", "d"})
        assert mask.tolist() == [False, True, False, True]

    def test_empty_set_gives_all_false(self):
        node_order = ["a", "b"]
        mask = build_mask_for_segments(node_order, set())
        assert mask.tolist() == [False, False]


class TestAggregateFoldMetrics:
    def test_mean_and_std_computed_across_folds(self):
        fold_reports = [
            {"t1": {"heldout": {"f1": 0.5, "precision": 0.5, "recall": 0.5, "accuracy": 0.5}}},
            {"t1": {"heldout": {"f1": 0.7, "precision": 0.7, "recall": 0.7, "accuracy": 0.7}}},
        ]
        agg = aggregate_fold_metrics(fold_reports)
        assert agg["t1"]["f1"]["mean"] == 0.6
        assert agg["t1"]["f1"]["values"] == [0.5, 0.7]
        assert agg["t1"]["f1"]["std"] > 0

    def test_zero_std_when_folds_agree(self):
        fold_reports = [
            {"t1": {"heldout": {"f1": 0.8, "precision": 0.8, "recall": 0.8, "accuracy": 0.8}}},
            {"t1": {"heldout": {"f1": 0.8, "precision": 0.8, "recall": 0.8, "accuracy": 0.8}}},
        ]
        agg = aggregate_fold_metrics(fold_reports)
        assert agg["t1"]["f1"]["std"] == 0.0


class TestMeanFloodRelevantF1:
    def test_averages_only_flood_relevant_transitions(self):
        aggregated = {
            "pre_event->rising": {"f1": {"mean": 0.0}},
            "rising->peak": {"f1": {"mean": 0.8}},
            "peak->receding": {"f1": {"mean": 0.6}},
        }
        # (0.8 + 0.6) / 2 = 0.7, NOT averaged with pre_event->rising's 0.0
        assert mean_flood_relevant_f1(aggregated) == 0.7

    def test_missing_transition_handled(self):
        aggregated = {"rising->peak": {"f1": {"mean": 0.9}}}
        assert mean_flood_relevant_f1(aggregated) == 0.9


class TestSweepThreshold:
    def test_picks_threshold_maximizing_f1(self):
        # y_true all 1s; a threshold that predicts everything as flooded (low
        # threshold) should win over a high threshold that predicts nothing
        y_true = torch.tensor([[1.0]] * 10)
        y_prob = torch.tensor([[0.6]] * 10)  # every probability is 0.6
        pooled = {"rising->peak": (y_true, y_prob), "peak->receding": (y_true, y_prob)}
        best_threshold, sweep = sweep_threshold(pooled, thresholds=[0.3, 0.5, 0.7, 0.9])
        assert best_threshold in (0.3, 0.5)  # thresholds below 0.6 correctly predict all-positive -> F1=1.0
        assert sweep[best_threshold] == 1.0

    def test_returns_full_sweep_dict(self):
        y_true = torch.tensor([[1.0], [0.0]])
        y_prob = torch.tensor([[0.6], [0.4]])
        pooled = {"rising->peak": (y_true, y_prob), "peak->receding": (y_true, y_prob)}
        _, sweep = sweep_threshold(pooled, thresholds=[0.3, 0.5, 0.7])
        assert set(sweep.keys()) == {0.3, 0.5, 0.7}
