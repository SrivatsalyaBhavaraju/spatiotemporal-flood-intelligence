"""
Unit tests for src/models/gnn/build_training_harness.py (task 3.7).

Covers the pure split/evaluation logic (ward-level greedy split, segment
split assignment, per-split metric evaluation) with small synthetic data.
The real spatial join (assign_wards_to_segments()) is validated by an
actual run against real data (see developing.md's task 3.7 notes).

Run with:
    pytest tests/test_build_training_harness.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.gnn.build_training_harness import (  # noqa: E402
    assign_segment_splits,
    build_ward_split,
    evaluate_split,
)


class TestBuildWardSplit:
    def test_all_wards_assigned(self):
        ward_sizes = {1: 100, 2: 200, 3: 300, 4: 400}
        split = build_ward_split(ward_sizes)
        assert set(split.keys()) == set(ward_sizes.keys())
        assert set(split.values()) <= {"train", "val", "test"}

    def test_deterministic_with_fixed_seed(self):
        ward_sizes = {i: 100 for i in range(1, 17)}
        split1 = build_ward_split(ward_sizes, seed=42)
        split2 = build_ward_split(ward_sizes, seed=42)
        assert split1 == split2

    def test_different_seeds_can_differ(self):
        ward_sizes = {i: 100 for i in range(1, 17)}
        split1 = build_ward_split(ward_sizes, seed=1)
        split2 = build_ward_split(ward_sizes, seed=999)
        assert split1 != split2  # not guaranteed in general, but true for this many wards/seeds

    def test_achieves_approximately_target_fractions(self):
        # 16 equal-sized wards, easy to hit 70/15/15 closely
        ward_sizes = {i: 100 for i in range(1, 17)}
        split = build_ward_split(ward_sizes, fractions={"train": 0.70, "val": 0.15, "test": 0.15}, seed=42)
        counts = {"train": 0, "val": 0, "test": 0}
        for ward_no, name in split.items():
            counts[name] += ward_sizes[ward_no]
        total = sum(ward_sizes.values())
        assert 0.60 <= counts["train"] / total <= 0.80
        assert counts["val"] / total <= 0.25
        assert counts["test"] / total <= 0.25

    def test_all_three_buckets_nonempty_with_enough_wards(self):
        ward_sizes = {i: 100 for i in range(1, 17)}
        split = build_ward_split(ward_sizes, seed=42)
        assert set(split.values()) == {"train", "val", "test"}


class TestAssignSegmentSplits:
    def test_maps_segment_ward_to_split_name(self):
        segment_ward = pd.Series({"s1": 1, "s2": 2, "s3": 1})
        ward_split = {1: "train", 2: "test"}
        result = assign_segment_splits(segment_ward, ward_split)
        assert result["s1"] == "train"
        assert result["s2"] == "test"
        assert result["s3"] == "train"


class TestEvaluateSplit:
    def test_computes_metrics_per_split(self):
        y_true = pd.Series([1, 1, 0, 0, 1, 0])
        y_pred = pd.Series([1, 0, 0, 0, 1, 1])
        split_labels = pd.Series(["train", "train", "train", "val", "val", "test"])
        result = evaluate_split(y_true, y_pred, split_labels)
        assert set(result.keys()) == {"train", "val", "test"}
        # train: y_true=[1,1,0], y_pred=[1,0,0] -> tp=1,fn=1,tn=1
        assert result["train"]["tp"] == 1
        assert result["train"]["fn"] == 1
        assert result["train"]["tn"] == 1

    def test_missing_split_omitted_not_errored(self):
        y_true = pd.Series([1, 0])
        y_pred = pd.Series([1, 0])
        split_labels = pd.Series(["train", "train"])  # no val/test present
        result = evaluate_split(y_true, y_pred, split_labels)
        assert set(result.keys()) == {"train"}
