"""
Unit tests for src/models/baseline/evaluate_per_transition.py (task 5.2).

Covers compute_baseline_report()'s pure grouping/merging logic with small
synthetic data. evaluate_split() itself is task 3.7's own function,
already tested in test_build_training_harness.py -- not retested here.
The real, current numbers are validated by an actual run against the real
committed baseline_predictions.csv/segment_splits.csv (see developing.md's
task 5.2 notes).

Run with:
    pytest tests/test_evaluate_per_transition_baseline.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline.evaluate_per_transition import compute_baseline_report  # noqa: E402

PHASE_ID_TO_NAME = {0: "pre_event", 1: "rising", 2: "peak", 3: "receding"}


def _predictions():
    return pd.DataFrame([
        {"segment_id": "s1", "x_t_phase_id": 1, "y_t1_phase_id": 2, "y_true": 1, "y_pred": 1},
        {"segment_id": "s2", "x_t_phase_id": 1, "y_t1_phase_id": 2, "y_true": 0, "y_pred": 0},
        {"segment_id": "s3", "x_t_phase_id": 1, "y_t1_phase_id": 2, "y_true": 1, "y_pred": 0},
        {"segment_id": "s1", "x_t_phase_id": 2, "y_t1_phase_id": 3, "y_true": 1, "y_pred": 1},
        {"segment_id": "s2", "x_t_phase_id": 2, "y_t1_phase_id": 3, "y_true": 1, "y_pred": 1},
        {"segment_id": "s3", "x_t_phase_id": 2, "y_t1_phase_id": 3, "y_true": 0, "y_pred": 0},
    ])


def _splits():
    return pd.DataFrame([
        {"segment_id": "s1", "split": "train"},
        {"segment_id": "s2", "split": "val"},
        {"segment_id": "s3", "split": "test"},
    ])


class TestComputeBaselineReport:
    def test_one_entry_per_transition(self):
        report = compute_baseline_report(_predictions(), _splits(), PHASE_ID_TO_NAME)
        assert set(report.keys()) == {"rising->peak", "peak->receding"}

    def test_labels_match_phase_names(self):
        report = compute_baseline_report(_predictions(), _splits(), PHASE_ID_TO_NAME)
        assert "rising->peak" in report

    def test_splits_correctly_separated(self):
        report = compute_baseline_report(_predictions(), _splits(), PHASE_ID_TO_NAME)
        rising_peak = report["rising->peak"]
        assert set(rising_peak.keys()) == {"train", "val", "test"}
        # s1 (train): y_true=1, y_pred=1 -> perfect
        assert rising_peak["train"]["f1"] == 1.0
        # s3 (test): y_true=1, y_pred=0 -> total miss
        assert rising_peak["test"]["f1"] == 0.0

    def test_missing_split_assignment_excluded_not_crashed(self):
        predictions = pd.DataFrame([
            {"segment_id": "unassigned", "x_t_phase_id": 1, "y_t1_phase_id": 2, "y_true": 1, "y_pred": 1},
        ])
        report = compute_baseline_report(predictions, _splits(), PHASE_ID_TO_NAME)
        assert "rising->peak" in report  # doesn't crash on an unmatched segment_id
