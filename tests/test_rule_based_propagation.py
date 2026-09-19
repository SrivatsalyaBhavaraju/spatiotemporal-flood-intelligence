"""
Unit tests for src/models/baseline/rule_based_propagation.py (task 3.6).

Covers the pure rule/evaluation logic (adjacency construction, the two
baseline rules, and the hand-rolled binary metrics) with small synthetic
data. Real-data loading is validated by an actual run against committed
data (see developing.md's task 3.6 notes).

Run with:
    pytest tests/test_rule_based_propagation.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline.rule_based_propagation import (  # noqa: E402
    ELEVATION_THRESHOLD_M,
    RAINFALL_INTENSITY_THRESHOLD_MM_PER_DAY,
    binary_classification_metrics,
    build_undirected_adjacency,
    predict_transition,
)

NODE_ORDER = ["a", "b", "c", "d"]


class TestBuildUndirectedAdjacency:
    def test_directed_edge_becomes_bidirectional(self):
        edge_index = np.array([[0], [1]])  # a -> b only, directed
        adj = build_undirected_adjacency(edge_index, NODE_ORDER)
        assert "b" in adj["a"]
        assert "a" in adj["b"]  # the reverse direction, not present in edge_index itself

    def test_isolated_node_has_empty_adjacency(self):
        edge_index = np.array([[0], [1]])
        adj = build_undirected_adjacency(edge_index, NODE_ORDER)
        assert adj["c"] == set()
        assert adj["d"] == set()

    def test_multiple_edges(self):
        edge_index = np.array([[0, 1], [1, 2]])  # a->b, b->c
        adj = build_undirected_adjacency(edge_index, NODE_ORDER)
        assert adj["b"] == {"a", "c"}


class TestPredictTransition:
    def test_rainfall_trigger_floods_everything(self):
        adjacency = {s: set() for s in NODE_ORDER}
        elevation = pd.Series({s: 100.0 for s in NODE_ORDER})  # all high elevation, no cascading possible
        y_t = {s: 0 for s in NODE_ORDER}
        pred = predict_transition(NODE_ORDER, adjacency, elevation, rainfall_intensity_mm_per_day=200.0, y_t=y_t)
        assert all(v == 1 for v in pred.values())

    def test_no_rainfall_no_neighbor_flood_stays_dry(self):
        adjacency = {s: set() for s in NODE_ORDER}
        elevation = pd.Series({s: 0.0 for s in NODE_ORDER})  # all low elevation
        y_t = {s: 0 for s in NODE_ORDER}
        pred = predict_transition(NODE_ORDER, adjacency, elevation, rainfall_intensity_mm_per_day=0.0, y_t=y_t)
        assert all(v == 0 for v in pred.values())

    def test_neighbor_cascading_requires_both_conditions(self):
        # b is a's only neighbor. b is flooded (y_t). a is low elevation -> should flood.
        adjacency = {"a": {"b"}, "b": {"a"}, "c": set(), "d": set()}
        elevation = pd.Series({"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0})
        y_t = {"a": 0, "b": 1, "c": 0, "d": 0}
        pred = predict_transition(NODE_ORDER, adjacency, elevation, rainfall_intensity_mm_per_day=0.0, y_t=y_t)
        assert pred["a"] == 1  # neighbor b flooded + low elevation

    def test_neighbor_flooded_but_high_elevation_stays_dry(self):
        adjacency = {"a": {"b"}, "b": {"a"}, "c": set(), "d": set()}
        elevation = pd.Series({"a": 100.0, "b": 0.0, "c": 0.0, "d": 0.0})  # a is high elevation
        y_t = {"a": 0, "b": 1, "c": 0, "d": 0}
        pred = predict_transition(NODE_ORDER, adjacency, elevation, rainfall_intensity_mm_per_day=0.0, y_t=y_t)
        assert pred["a"] == 0

    def test_low_elevation_but_no_flooded_neighbor_stays_dry(self):
        adjacency = {"a": {"b"}, "b": {"a"}, "c": set(), "d": set()}
        elevation = pd.Series({"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0})
        y_t = {"a": 0, "b": 0, "c": 0, "d": 0}  # no neighbor flooded
        pred = predict_transition(NODE_ORDER, adjacency, elevation, rainfall_intensity_mm_per_day=0.0, y_t=y_t)
        assert pred["a"] == 0

    def test_exactly_at_elevation_threshold_counts_as_low(self):
        adjacency = {"a": {"b"}, "b": {"a"}, "c": set(), "d": set()}
        elevation = pd.Series({"a": ELEVATION_THRESHOLD_M, "b": 0.0, "c": 0.0, "d": 0.0})
        y_t = {"a": 0, "b": 1, "c": 0, "d": 0}
        pred = predict_transition(NODE_ORDER, adjacency, elevation, rainfall_intensity_mm_per_day=0.0, y_t=y_t)
        assert pred["a"] == 1

    def test_rainfall_at_threshold_does_not_trigger(self):
        # strict ">" per module docstring/implementation, not ">="
        adjacency = {s: set() for s in NODE_ORDER}
        elevation = pd.Series({s: 100.0 for s in NODE_ORDER})
        y_t = {s: 0 for s in NODE_ORDER}
        pred = predict_transition(
            NODE_ORDER, adjacency, elevation,
            rainfall_intensity_mm_per_day=RAINFALL_INTENSITY_THRESHOLD_MM_PER_DAY, y_t=y_t,
        )
        assert all(v == 0 for v in pred.values())


class TestBinaryClassificationMetrics:
    def test_perfect_predictions(self):
        metrics = binary_classification_metrics([1, 0, 1, 0], [1, 0, 1, 0])
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["accuracy"] == 1.0

    def test_all_wrong(self):
        metrics = binary_classification_metrics([1, 0, 1, 0], [0, 1, 0, 1])
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
        assert metrics["accuracy"] == 0.0

    def test_confusion_matrix_counts(self):
        metrics = binary_classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        assert metrics["tp"] == 1
        assert metrics["fp"] == 1
        assert metrics["fn"] == 1
        assert metrics["tn"] == 1

    def test_no_positive_predictions_zero_precision_not_nan(self):
        metrics = binary_classification_metrics([1, 0], [0, 0])
        assert metrics["precision"] == 0.0  # would be 0/0 without the guard
        assert metrics["recall"] == 0.0

    def test_no_positive_labels_zero_recall_not_nan(self):
        metrics = binary_classification_metrics([0, 0], [1, 0])
        assert metrics["recall"] == 0.0  # would be 0/0 without the guard
