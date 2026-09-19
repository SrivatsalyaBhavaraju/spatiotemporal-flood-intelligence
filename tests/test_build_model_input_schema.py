"""
Unit tests for src/models/gnn/build_model_input_schema.py (task 2.7).

Builds a tiny synthetic line graph (3 segments) plus synthetic task
2.3/2.6-shaped static/dynamic feature tables directly, so X_t/edge_index
construction can be verified without real Phase 2 data on disk. Run against
the real pipeline via `python src/models/gnn/build_model_input_schema.py`.

Run with:
    pytest tests/test_build_model_input_schema.py -v
"""
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.gnn.build_model_input_schema import (  # noqa: E402
    FEATURE_COLUMNS,
    Phase2SchemaError,
    build_X,
    load_dynamic_features,
    load_node_order_and_edge_index,
    load_static_features,
    phase_table,
    save_outputs,
    validate_schema,
)

SEGMENTS = ["segA", "segB", "segC"]


def _make_line_graph() -> nx.DiGraph:
    L = nx.DiGraph()
    L.add_nodes_from(SEGMENTS)
    L.add_edge("segA", "segB")
    L.add_edge("segB", "segC")
    L.add_edge("segC", "segA")
    return L


def _make_static_df() -> pd.DataFrame:
    return pd.DataFrame({
        "segment_id": SEGMENTS,
        "elevation_m": [10.0, 5.0, 8.0],
        "slope_deg": [1.0, 2.0, 3.0],
        "length_m": [100.0, 200.0, 150.0],
        "distance_to_drain_m": [50.0, 20.0, 80.0],
    })


PHASES = [(0, "pre_event", 10.0, 10.0), (1, "rising", 5.0, 15.0)]


def _make_dynamic_df(segments=SEGMENTS) -> pd.DataFrame:
    rows = []
    for phase_id, phase_name, rainfall_t, cum in PHASES:
        for seg in segments:
            rows.append({
                "segment_id": seg, "phase_id": phase_id, "phase_name": phase_name,
                "rainfall_t": rainfall_t, "cumulative_rainfall_t": cum,
            })
    return pd.DataFrame(rows)


class TestLoadNodeOrderAndEdgeIndex:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Phase2SchemaError, match="build_line_graph.py"):
            load_node_order_and_edge_index(tmp_path / "does_not_exist.gpickle")

    def test_empty_graph_raises(self, tmp_path):
        p = tmp_path / "line_graph.gpickle"
        with open(p, "wb") as f:
            pickle.dump(nx.DiGraph(), f)
        with pytest.raises(Phase2SchemaError, match="zero nodes"):
            load_node_order_and_edge_index(p)

    def test_order_and_edge_index_match_line_graph(self, tmp_path):
        L = _make_line_graph()
        p = tmp_path / "line_graph.gpickle"
        with open(p, "wb") as f:
            pickle.dump(L, f)
        order, edge_index = load_node_order_and_edge_index(p)
        assert order == sorted(SEGMENTS)
        assert edge_index.shape == (2, 3)
        assert edge_index.dtype == np.int64
        assert int(edge_index.max()) < len(order)


class TestLoaders:
    def test_load_static_features_missing_column_raises(self, tmp_path):
        p = tmp_path / "bad_static.geojson"
        import geopandas as gpd
        from shapely.geometry import LineString
        gpd.GeoDataFrame(
            {"segment_id": ["segA"], "geometry": [LineString([(0, 0), (1, 1)])]},  # no elevation_m etc.
            crs="EPSG:4326",
        ).to_file(p, driver="GeoJSON")
        with pytest.raises(Phase2SchemaError, match="elevation_m"):
            load_static_features(p)

    def test_load_dynamic_features_missing_column_raises(self, tmp_path):
        p = tmp_path / "bad_dynamic.csv"
        pd.DataFrame({"segment_id": ["segA"], "phase_id": [0]}).to_csv(p, index=False)
        with pytest.raises(Phase2SchemaError, match="rainfall_t"):
            load_dynamic_features(p)


class TestPhaseTable:
    def test_sorted_unique_phases(self):
        dynamic_df = _make_dynamic_df()
        phases = phase_table(dynamic_df)
        assert phases["phase_id"].tolist() == [0, 1]
        assert phases["phase_name"].tolist() == ["pre_event", "rising"]


class TestBuildX:
    def test_shape_and_values(self):
        order = sorted(SEGMENTS)
        static_df = _make_static_df()
        dynamic_df = _make_dynamic_df()
        phases = phase_table(dynamic_df)
        X = build_X(static_df, dynamic_df, order, phases)

        assert X.shape == (2, 3, len(FEATURE_COLUMNS))
        assert X.dtype == np.float32

        idx_A = order.index("segA")
        row_t0 = X[0, idx_A]
        expected = [10.0, 1.0, 100.0, 50.0, 10.0, 10.0]  # elevation, slope, length, dist_drain, rainfall_t, cum
        assert row_t0.tolist() == pytest.approx(expected)

        # rainfall_t/cumulative_rainfall_t (last 2 cols) are uniform across
        # segments within a phase -- broadcast, per task 2.6.
        assert np.all(X[1, :, -2] == X[1, 0, -2])
        assert np.all(X[1, :, -1] == X[1, 0, -1])

    def test_missing_static_segment_raises(self):
        order = sorted(SEGMENTS) + ["segD"]  # segD has no static features
        static_df = _make_static_df()
        dynamic_df = _make_dynamic_df()
        phases = phase_table(dynamic_df)
        with pytest.raises(Phase2SchemaError, match="segD"):
            build_X(static_df, dynamic_df, order, phases)

    def test_missing_dynamic_segment_for_a_phase_raises(self):
        order = sorted(SEGMENTS)
        static_df = _make_static_df()
        dynamic_df = _make_dynamic_df(segments=["segA", "segB"])  # segC missing from every phase
        phases = phase_table(dynamic_df)
        with pytest.raises(Phase2SchemaError, match="segC"):
            build_X(static_df, dynamic_df, order, phases)


class TestValidateSchema:
    def test_reports_shapes_and_y_t1_contract(self):
        order = sorted(SEGMENTS)
        static_df = _make_static_df()
        dynamic_df = _make_dynamic_df()
        phases = phase_table(dynamic_df)
        X = build_X(static_df, dynamic_df, order, phases)
        edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)

        report = validate_schema(X, edge_index, order, phases)
        assert report["shapes"] == {"X": [2, 3, 6], "edge_index": [2, 2], "N_nodes": 3, "F_features": 6, "T_phases": 2}
        assert report["missing_values"]["total_nan_cells"] == 0
        assert report["edge_index_bounds"]["in_range"] is True
        assert report["y_t1_contract"]["shape_per_usable_timestep"] == [3, 1]
        assert report["y_t1_contract"]["usable_transitions"] == [{"x_t_phase_id": 0, "y_t1_phase_id": 1}]

    def test_out_of_range_edge_index_raises(self):
        order = sorted(SEGMENTS)
        static_df = _make_static_df()
        dynamic_df = _make_dynamic_df()
        phases = phase_table(dynamic_df)
        X = build_X(static_df, dynamic_df, order, phases)
        bad_edge_index = np.array([[0], [5]], dtype=np.int64)  # 5 is out of range for N=3

        with pytest.raises(Phase2SchemaError, match="node position"):
            validate_schema(X, bad_edge_index, order, phases)


class TestSaveOutputs:
    def test_round_trip(self, tmp_path):
        order = sorted(SEGMENTS)
        static_df = _make_static_df()
        dynamic_df = _make_dynamic_df()
        phases = phase_table(dynamic_df)
        X = build_X(static_df, dynamic_df, order, phases)
        edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)

        paths = save_outputs(order, edge_index, X, out_dir=tmp_path)

        import json
        loaded_order = json.loads(paths["node_order_json"].read_text())
        assert loaded_order == order
        assert np.array_equal(np.load(paths["edge_index_npy"]), edge_index)
        assert np.allclose(np.load(paths["X_npy"]), X)
