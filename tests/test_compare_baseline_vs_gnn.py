"""
Unit tests for src/models/compare_baseline_vs_gnn.py (task 5.3).

Covers the pure table-building/verdict logic with small synthetic
report-shaped dicts. `compute_fair_holdout_auc()` (loads a real model)
and the plot functions are validated by a real run against real data
(see developing.md's task 5.3 notes) -- not mocked here.

Run with:
    pytest tests/test_compare_baseline_vs_gnn.py -v
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.compare_baseline_vs_gnn import (  # noqa: E402
    build_comparison_table,
    evaluate_core_claim,
)


def _gnn_report():
    return {
        "rising->peak": {
            "train": {"f1": 0.96, "accuracy": 0.93, "auc": 0.78},
            "val": {"f1": 0.63, "accuracy": 0.46, "auc": 0.70},
            "test": {"f1": 0.98, "accuracy": 0.96, "auc": 0.46},
        },
    }


def _baseline_report():
    return {
        "rising->peak": {
            "train": {"f1": 0.0, "accuracy": 0.07},
            "val": {"f1": 0.0, "accuracy": 0.54},
            "test": {"f1": 0.0, "accuracy": 0.04},
        },
    }


def _fair_holdout_auc():
    return {"rising->peak": {"train": 0.78, "val": 0.59, "test": 0.42}}


def _anomaly_report(val_informative=1, test_informative=0):
    return {
        "by_phase": {
            "peak": {
                "informativeness_by_split": {
                    "train": {"n_informative_wards": 0},
                    "val": {"n_informative_wards": val_informative},
                    "test": {"n_informative_wards": test_informative},
                }
            }
        }
    }


class TestBuildComparisonTable:
    def test_merges_baseline_and_gnn_metrics(self):
        table = build_comparison_table(_gnn_report(), _baseline_report())
        assert table["rising->peak"]["test"]["baseline_f1"] == 0.0
        assert table["rising->peak"]["test"]["gnn_f1"] == 0.98
        assert table["rising->peak"]["test"]["gnn_auc_tuned_model"] == 0.46

    def test_missing_baseline_split_gives_none_not_crash(self):
        baseline_report = {"rising->peak": {"train": {"f1": 0.0}}}  # missing val/test
        table = build_comparison_table(_gnn_report(), baseline_report)
        assert table["rising->peak"]["val"]["baseline_f1"] is None
        assert table["rising->peak"]["val"]["gnn_f1"] == 0.63  # gnn side still populated


class TestEvaluateCoreClaim:
    def test_supported_on_val_when_informative_and_above_chance(self):
        table = build_comparison_table(_gnn_report(), _baseline_report())
        verdicts = evaluate_core_claim(table, _fair_holdout_auc(), _anomaly_report(val_informative=1))
        assert "SUPPORTED" in verdicts["rising->peak"]["val_conclusion"]

    def test_not_supported_on_val_when_auc_at_chance(self):
        table = build_comparison_table(_gnn_report(), _baseline_report())
        fair_auc = {"rising->peak": {"train": 0.78, "val": 0.51, "test": 0.42}}  # val at chance
        verdicts = evaluate_core_claim(table, fair_auc, _anomaly_report(val_informative=1))
        assert verdicts["rising->peak"]["val_conclusion"] == "NOT SUPPORTED on val."

    def test_test_split_inconclusive_when_no_informative_wards(self):
        table = build_comparison_table(_gnn_report(), _baseline_report())
        verdicts = evaluate_core_claim(table, _fair_holdout_auc(), _anomaly_report(test_informative=0))
        assert "INCONCLUSIVE" in verdicts["rising->peak"]["test_conclusion"]

    def test_test_split_evaluated_normally_when_informative_wards_exist(self):
        table = build_comparison_table(_gnn_report(), _baseline_report())
        fair_auc = {"rising->peak": {"train": 0.78, "val": 0.59, "test": 0.65}}
        verdicts = evaluate_core_claim(table, fair_auc, _anomaly_report(test_informative=2))
        assert verdicts["rising->peak"]["test_conclusion"] == "SUPPORTED on test."

    def test_reports_raw_values_for_transparency(self):
        table = build_comparison_table(_gnn_report(), _baseline_report())
        verdicts = evaluate_core_claim(table, _fair_holdout_auc(), _anomaly_report())
        v = verdicts["rising->peak"]
        assert v["baseline_test_f1"] == 0.0
        assert v["gnn_fair_holdout_val_auc"] == 0.59
        assert v["gnn_fair_holdout_test_auc"] == 0.42
        assert v["val_has_informative_wards"] is True
        assert v["test_has_informative_wards"] is False
