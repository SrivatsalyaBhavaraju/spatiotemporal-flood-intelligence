"""
Unit tests for src/models/gnn/evaluate_final_model.py (task 5.1).

Covers the pure logic (loading task 4.2's selected hyperparameters,
the reproducibility check) with small synthetic data. The actual
train_model()/evaluate_model() calls are task 4.1's own (already tested
in test_train_gnn.py) and this module's real job -- reproducing task 4.2's
final model exactly -- is validated by a real run against real data (see
developing.md's task 5.1 notes), not mocked.

Run with:
    pytest tests/test_evaluate_final_model.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.gnn.evaluate_final_model import (  # noqa: E402
    add_auc_to_report,
    check_reproduces_tuning_report,
    compute_auc,
    load_tuned_hyperparameters,
)
from src.models.gnn.train_gnn import Phase4TrainingError  # noqa: E402


class TestLoadTunedHyperparameters:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Phase4TrainingError):
            load_tuned_hyperparameters(tmp_path / "nope.json")

    def test_reads_lr_and_threshold(self, tmp_path):
        p = tmp_path / "tuning_report.json"
        p.write_text(json.dumps({
            "selected_learning_rate": 0.02, "selected_threshold": 0.1, "final_test_evaluation": {},
        }), encoding="utf-8")
        lr, threshold, report = load_tuned_hyperparameters(p)
        assert lr == 0.02
        assert threshold == 0.1
        assert report["selected_learning_rate"] == 0.02


class TestCheckReproducesTuningReport:
    def test_passes_when_metrics_match_exactly(self):
        tuning_report = {
            "final_test_evaluation": {
                "rising->peak": {"test": {"f1": 0.982, "accuracy": 0.9646}},
            }
        }
        report = {"rising->peak": {"train": {"f1": 0.99}, "test": {"f1": 0.982, "accuracy": 0.9646}}}
        check_reproduces_tuning_report(report, tuning_report)  # must not raise

    def test_raises_on_mismatch(self):
        tuning_report = {"final_test_evaluation": {"rising->peak": {"test": {"f1": 0.982}}}}
        report = {"rising->peak": {"test": {"f1": 0.5}}}  # doesn't match
        with pytest.raises(Phase4TrainingError, match="Reproducibility check failed"):
            check_reproduces_tuning_report(report, tuning_report)

    def test_checks_every_transition_not_just_the_first(self):
        tuning_report = {
            "final_test_evaluation": {
                "rising->peak": {"test": {"f1": 0.982}},
                "peak->receding": {"test": {"f1": 0.973}},
            }
        }
        report = {
            "rising->peak": {"test": {"f1": 0.982}},
            "peak->receding": {"test": {"f1": 0.111}},  # mismatch on the SECOND transition
        }
        with pytest.raises(Phase4TrainingError, match="peak->receding"):
            check_reproduces_tuning_report(report, tuning_report)

    def test_tolerates_extra_keys_not_present_in_expected(self):
        # `report` has an "auc" key task 4.2 never saved -- must not be treated as a mismatch
        tuning_report = {"final_test_evaluation": {"rising->peak": {"test": {"f1": 0.982}}}}
        report = {"rising->peak": {"test": {"f1": 0.982, "auc": 0.46}}}
        check_reproduces_tuning_report(report, tuning_report)  # must not raise


class TestComputeAuc:
    def test_perfect_separation_gives_auc_one(self):
        y_true = [0, 0, 0, 1, 1, 1]
        y_prob = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        assert compute_auc(y_true, y_prob) == 1.0

    def test_inverted_separation_gives_auc_zero(self):
        y_true = [0, 0, 0, 1, 1, 1]
        y_prob = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        assert compute_auc(y_true, y_prob) == 0.0

    def test_no_discrimination_gives_auc_near_half(self):
        # identical probability for every example regardless of label --
        # exactly the real degenerate case this function exists to catch
        y_true = [0, 0, 1, 1]
        y_prob = [0.5, 0.5, 0.5, 0.5]
        assert compute_auc(y_true, y_prob) == 0.5

    def test_single_class_returns_none(self):
        assert compute_auc([0, 0, 0], [0.1, 0.5, 0.9]) is None
        assert compute_auc([1, 1, 1], [0.1, 0.5, 0.9]) is None

    def test_handles_ties_correctly(self):
        # two positives and one negative all tied at 0.5 -- average rank
        # should still land near chance (0.5), not crash or be biased
        y_true = [0, 1, 1]
        y_prob = [0.5, 0.5, 0.5]
        assert compute_auc(y_true, y_prob) == 0.5


class TestAddAucToReport:
    def test_adds_auc_key_per_split(self):
        y_true = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
        y_prob = torch.tensor([[0.1], [0.2], [0.8], [0.9]])
        probabilities = {"rising->peak": (y_true, y_prob)}
        split_masks = {"test": torch.tensor([True, True, True, True])}
        report = {"rising->peak": {"test": {"f1": 1.0}}}
        result = add_auc_to_report(report, probabilities, split_masks)
        assert result["rising->peak"]["test"]["auc"] == 1.0

    def test_skips_splits_not_present_in_report(self):
        y_true = torch.tensor([[0.0], [1.0]])
        y_prob = torch.tensor([[0.1], [0.9]])
        probabilities = {"rising->peak": (y_true, y_prob)}
        split_masks = {"train": torch.tensor([True, True]), "val": torch.tensor([True, True])}
        report = {"rising->peak": {"train": {"f1": 1.0}}}  # no "val" key in report
        result = add_auc_to_report(report, probabilities, split_masks)
        assert "auc" in result["rising->peak"]["train"]
        assert "val" not in result["rising->peak"]
