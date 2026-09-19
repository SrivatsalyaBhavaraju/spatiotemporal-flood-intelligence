"""
Unit tests for src/graph/verify_graph_freeze.py (task 3.5).

Covers the pure fingerprinting/comparison logic with small synthetic data.
The real check against the actual committed FROZEN_FINGERPRINT constant is
exercised by an actual run against real data (see developing.md's task
3.5 notes).

Run with:
    pytest tests/test_verify_graph_freeze.py -v
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.graph.verify_graph_freeze import (  # noqa: E402
    FROZEN_FINGERPRINT,
    check_segment_id_consistency,
    compare_fingerprints,
    compute_fingerprint,
    hash_segment_ids,
)


class TestHashSegmentIds:
    def test_deterministic_regardless_of_input_order(self):
        assert hash_segment_ids(["b", "a", "c"]) == hash_segment_ids(["a", "b", "c"]) == hash_segment_ids(["c", "b", "a"])

    def test_different_sets_hash_differently(self):
        assert hash_segment_ids(["a", "b"]) != hash_segment_ids(["a", "c"])

    def test_same_count_different_ids_hash_differently(self):
        # the whole point of hashing the ID set, not just counting it
        assert hash_segment_ids(["seg1", "seg2", "seg3"]) != hash_segment_ids(["seg4", "seg5", "seg6"])

    def test_works_with_set_input(self):
        assert hash_segment_ids({"a", "b", "c"}) == hash_segment_ids(["a", "b", "c"])


class TestComputeFingerprint:
    def test_basic_fields(self):
        fp = compute_fingerprint(3, 5, ["a", "b", "c"])
        assert fp["node_count"] == 3
        assert fp["edge_count"] == 5
        assert fp["segment_id_set_sha256"] == hash_segment_ids(["a", "b", "c"])


class TestCompareFingerprints:
    def test_exact_match(self):
        frozen = {"node_count": 10, "edge_count": 20, "segment_id_set_sha256": "abc"}
        actual = {"node_count": 10, "edge_count": 20, "segment_id_set_sha256": "abc"}
        result = compare_fingerprints(actual, frozen)
        assert result["match"] is True
        assert result["mismatches"] == {}

    def test_node_count_drift_detected(self):
        frozen = {"node_count": 10, "edge_count": 20, "segment_id_set_sha256": "abc"}
        actual = {"node_count": 11, "edge_count": 20, "segment_id_set_sha256": "abc"}
        result = compare_fingerprints(actual, frozen)
        assert result["match"] is False
        assert "node_count" in result["mismatches"]
        assert result["mismatches"]["node_count"] == {"frozen": 10, "actual": 11}

    def test_hash_drift_detected_even_with_same_counts(self):
        frozen = {"node_count": 10, "edge_count": 20, "segment_id_set_sha256": "abc"}
        actual = {"node_count": 10, "edge_count": 20, "segment_id_set_sha256": "different"}
        result = compare_fingerprints(actual, frozen)
        assert result["match"] is False
        assert "segment_id_set_sha256" in result["mismatches"]

    def test_uses_module_frozen_fingerprint_by_default(self):
        result = compare_fingerprints(FROZEN_FINGERPRINT)
        assert result["match"] is True


class TestCheckSegmentIdConsistency:
    def test_fully_consistent(self):
        ref = {"a", "b", "c"}
        result = check_segment_id_consistency(ref, {"a", "b", "c"}, "test")
        assert result["consistent"] is True
        assert result["missing_count"] == 0
        assert result["extra_count"] == 0

    def test_missing_segments_detected(self):
        ref = {"a", "b", "c"}
        result = check_segment_id_consistency(ref, {"a", "b"}, "test")
        assert result["consistent"] is False
        assert result["missing_count"] == 1
        assert result["missing_sample"] == ["c"]

    def test_extra_segments_detected(self):
        ref = {"a", "b"}
        result = check_segment_id_consistency(ref, {"a", "b", "z"}, "test")
        assert result["consistent"] is False
        assert result["extra_count"] == 1
        assert result["extra_sample"] == ["z"]

    def test_both_missing_and_extra(self):
        ref = {"a", "b", "c"}
        result = check_segment_id_consistency(ref, {"a", "z"}, "test")
        assert result["consistent"] is False
        assert result["missing_count"] == 2
        assert result["extra_count"] == 1
