"""
Unit tests for src/ground_truth/sanity_check_comparison_anomalies.py (task 5.4).

Covers the pure per-ward diagnostic logic (label table construction,
informativeness classification, split summary) with small synthetic data.
The real, headline finding (only 1 of 16 wards is informative, landing in
val not test) is validated against the actual real committed data (see
developing.md's task 5.4 notes), not reproduced here with fake numbers.

Run with:
    pytest tests/test_sanity_check_comparison_anomalies.py -v
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ground_truth.sanity_check_comparison_anomalies import (  # noqa: E402
    build_ward_label_table,
    classify_informative_wards,
    summarize_informativeness_by_split,
)


def _labels_df():
    # ward 1 (train): mixed labeling, real elevation signal
    # ward 2 (test): near-uniformly flooded, no real signal
    rows = []
    for seg, ward, elev, label in [
        ("s1", 1, 2.0, 1), ("s2", 1, 4.0, 1), ("s3", 1, 8.0, 0), ("s4", 1, 10.0, 0),
        ("s5", 2, 3.0, 1), ("s6", 2, 5.0, 1), ("s7", 2, 7.0, 1), ("s8", 2, 9.0, 1), ("s9", 2, 11.0, 0),
    ]:
        rows.append({"segment_id": seg, "phase_name": "peak", "flood_label": label})
    return pd.DataFrame(rows)


def _splits_df():
    return pd.DataFrame([
        {"segment_id": "s1", "ward_no": 1, "split": "train"}, {"segment_id": "s2", "ward_no": 1, "split": "train"},
        {"segment_id": "s3", "ward_no": 1, "split": "train"}, {"segment_id": "s4", "ward_no": 1, "split": "train"},
        {"segment_id": "s5", "ward_no": 2, "split": "test"}, {"segment_id": "s6", "ward_no": 2, "split": "test"},
        {"segment_id": "s7", "ward_no": 2, "split": "test"}, {"segment_id": "s8", "ward_no": 2, "split": "test"},
        {"segment_id": "s9", "ward_no": 2, "split": "test"},
    ])


def _elevation_df():
    return pd.DataFrame([
        {"segment_id": s, "elevation_m": e} for s, e in
        [("s1", 2.0), ("s2", 4.0), ("s3", 8.0), ("s4", 10.0), ("s5", 3.0), ("s6", 5.0), ("s7", 7.0), ("s8", 9.0), ("s9", 11.0)]
    ])


class TestBuildWardLabelTable:
    def test_one_row_per_ward(self):
        table = build_ward_label_table(_labels_df(), _splits_df(), _elevation_df(), "peak")
        assert len(table) == 2
        assert set(table["ward_no"]) == {1, 2}

    def test_pct_flooded_correct(self):
        table = build_ward_label_table(_labels_df(), _splits_df(), _elevation_df(), "peak").set_index("ward_no")
        assert table.loc[1, "pct_flooded"] == 50.0   # 2/4 flooded
        assert table.loc[2, "pct_flooded"] == 80.0   # 4/5 flooded

    def test_minority_class_pct_correct(self):
        table = build_ward_label_table(_labels_df(), _splits_df(), _elevation_df(), "peak").set_index("ward_no")
        assert table.loc[1, "minority_class_pct"] == 50.0
        assert table.loc[2, "minority_class_pct"] == 20.0

    def test_correlation_reflects_real_relationship(self):
        table = build_ward_label_table(_labels_df(), _splits_df(), _elevation_df(), "peak").set_index("ward_no")
        # ward 1: elevation 2,4 -> flooded; 8,10 -> not flooded -- strong negative correlation
        assert table.loc[1, "elevation_flood_corr"] < -0.8

    def test_zero_variance_ward_gives_none_correlation(self):
        labels = pd.DataFrame([
            {"segment_id": "a", "phase_name": "peak", "flood_label": 1},
            {"segment_id": "b", "phase_name": "peak", "flood_label": 1},
        ])
        splits = pd.DataFrame([
            {"segment_id": "a", "ward_no": 9, "split": "train"},
            {"segment_id": "b", "ward_no": 9, "split": "train"},
        ])
        elevation = pd.DataFrame([{"segment_id": "a", "elevation_m": 5.0}, {"segment_id": "b", "elevation_m": 6.0}])
        table = build_ward_label_table(labels, splits, elevation, "peak")
        assert table.iloc[0]["elevation_flood_corr"] is None


class TestClassifyInformativeWards:
    def test_flags_informative_and_uninformative_correctly(self):
        table = build_ward_label_table(_labels_df(), _splits_df(), _elevation_df(), "peak")
        classified = classify_informative_wards(table, min_minority_pct=10.0)
        result = classified.set_index("ward_no")["informative"]
        assert result.loc[1] == True  # noqa: E712 -- 50% minority, clearly informative
        assert result.loc[2] == True  # noqa: E712 -- 20% minority, still above 10% threshold

    def test_threshold_excludes_near_uniform_ward(self):
        table = build_ward_label_table(_labels_df(), _splits_df(), _elevation_df(), "peak")
        classified = classify_informative_wards(table, min_minority_pct=25.0)
        result = classified.set_index("ward_no")["informative"]
        assert result.loc[1] == True   # noqa: E712 -- 50% >= 25%
        assert result.loc[2] == False  # noqa: E712 -- 20% < 25%


class TestSummarizeInformativenessBySplit:
    def test_counts_correct(self):
        table = build_ward_label_table(_labels_df(), _splits_df(), _elevation_df(), "peak")
        classified = classify_informative_wards(table, min_minority_pct=10.0)
        summary = summarize_informativeness_by_split(classified)
        assert summary["train"]["n_wards"] == 1
        assert summary["train"]["n_informative_wards"] == 1
        assert summary["test"]["n_wards"] == 1
        assert summary["test"]["n_informative_wards"] == 1

    def test_zero_informative_wards_in_a_split(self):
        table = build_ward_label_table(_labels_df(), _splits_df(), _elevation_df(), "peak")
        classified = classify_informative_wards(table, min_minority_pct=25.0)
        summary = summarize_informativeness_by_split(classified)
        assert summary["test"]["n_informative_wards"] == 0
        assert summary["test"]["n_informative_segments"] == 0
