"""
Unit tests for src/graph/build_line_graph.py (task 2.2).

Builds a small synthetic PRIMAL graph directly (in the same shape task 2.1's
build_primal_graph() would produce -- nodes with x/y, edges with
segment_id/length_m/geometry/OSM tags) so the line-graph transform can be
tested without needing real Phase 1/2.1 data on disk.

Run with:
    pytest tests/test_build_line_graph.py -v
"""
import pickle
import sys
from pathlib import Path

import networkx as nx
import pytest
from shapely.geometry import LineString

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.graph.build_line_graph import (  # noqa: E402
    Phase2ArtifactError,
    build_line_graph,
    build_line_graph_edge_table,
    build_line_graph_node_table,
    edge_index_array,
    load_primal_graph,
    node_order,
    save_outputs,
    validate_line_graph,
)

# Same square-loop + oneway-spur shape as tests/test_build_primal_graph.py,
# built directly as an already-2.1-annotated primal graph (segment_id,
# length_m, geometry already set on every edge, exactly as
# build_primal_graph.build_primal_graph() would leave it).
#   A(1001) -- B(1002) -- E(1005)  [oneway spur B->E]
#   |            |
#   D(1004) -- C(1003)
A, B, C, D, E = 1001, 1002, 1003, 1004, 1005
NODE_COORDS = {A: (80.20, 13.00), B: (80.21, 13.00), C: (80.21, 12.99), D: (80.20, 12.99), E: (80.22, 13.00)}


def _make_primal_graph(with_deadend=False):
    G = nx.MultiDiGraph()
    for n, (x, y) in NODE_COORDS.items():
        G.add_node(n, x=x, y=y, street_count=3)

    def add_edge(u, v, **attrs):
        ux, uy = NODE_COORDS[u]
        vx, vy = NODE_COORDS[v]
        geom = LineString([(ux, uy), (vx, vy)])
        k = G.new_edge_key(u, v) if G.has_edge(u, v) else 0
        seg_id = f"{u}_{v}_{k}"
        base = dict(
            highway="residential", oneway=False, length=geom.length * 111_000,
            length_m=round(geom.length * 111_000, 3), geometry=geom, segment_id=seg_id,
        )
        base.update(attrs)
        G.add_edge(u, v, key=k, **base)

    add_edge(A, B, name="North St", lanes="2")
    add_edge(B, A, name="North St", lanes="2")
    add_edge(B, C, name="East St")
    add_edge(C, B, name="East St")
    add_edge(C, D, name="South St")
    add_edge(D, C, name="South St")
    add_edge(D, A, name="West St")
    add_edge(A, D, name="West St")
    add_edge(B, E, name="Spur Rd", oneway=True)  # oneway dead-end spur -> E has no outgoing edge
    if with_deadend:
        pass  # E is already a natural dead-end (no outgoing edges) in the base fixture
    return G


class TestLoadPrimalGraph:
    def test_missing_file_raises_actionable_error(self, tmp_path):
        with pytest.raises(Phase2ArtifactError, match="build_primal_graph.py"):
            load_primal_graph(tmp_path / "does_not_exist.gpickle")

    def test_missing_segment_id_raises(self, tmp_path):
        G = nx.MultiDiGraph()
        G.add_node(1, x=0.0, y=0.0)
        G.add_node(2, x=1.0, y=1.0)
        G.add_edge(1, 2, length_m=1.0)  # no segment_id
        p = tmp_path / "bad.gpickle"
        with open(p, "wb") as f:
            pickle.dump(G, f)
        with pytest.raises(Phase2ArtifactError, match="segment_id"):
            load_primal_graph(p)

    def test_loads_valid_graph(self, tmp_path):
        G = _make_primal_graph()
        p = tmp_path / "primal_graph.gpickle"
        with open(p, "wb") as f:
            pickle.dump(G, f)
        loaded = load_primal_graph(p)
        assert loaded.number_of_edges() == G.number_of_edges()


class TestBuildLineGraph:
    def test_node_count_equals_primal_edge_count(self):
        G = _make_primal_graph()
        L = build_line_graph(G)
        assert L.number_of_nodes() == G.number_of_edges() == 9

    def test_node_ids_are_segment_ids(self):
        G = _make_primal_graph()
        L = build_line_graph(G)
        primal_seg_ids = {d["segment_id"] for _, _, d in G.edges(data=True)}
        assert set(L.nodes) == primal_seg_ids

    def test_node_carries_full_primal_edge_attributes(self):
        G = _make_primal_graph()
        L = build_line_graph(G)
        seg_id = f"{A}_{B}_0"
        node = L.nodes[seg_id]
        assert node["highway"] == "residential"
        assert node["name"] == "North St"
        assert node["lanes"] == "2"
        assert node["length_m"] > 0
        assert node["u"] == f"osm_{A}"
        assert node["v"] == f"osm_{B}"
        assert hasattr(node["geometry"], "wkt")

    def test_adjacent_segments_connected_at_shared_intersection(self):
        # A->B (ends at B) should connect to B->C (starts at B) and B->E (starts at B)
        G = _make_primal_graph()
        L = build_line_graph(G)
        seg_AB = f"{A}_{B}_0"
        seg_BC = f"{B}_{C}_0"
        seg_BE = f"{B}_{E}_0"
        assert L.has_edge(seg_AB, seg_BC)
        assert L.has_edge(seg_AB, seg_BE)

    def test_non_adjacent_segments_not_connected(self):
        # A->B and C->D share no endpoint -> must not be connected either direction
        G = _make_primal_graph()
        L = build_line_graph(G)
        seg_AB = f"{A}_{B}_0"
        seg_CD = f"{C}_{D}_0"
        assert not L.has_edge(seg_AB, seg_CD)
        assert not L.has_edge(seg_CD, seg_AB)

    def test_oneway_deadend_segment_has_no_outgoing_line_graph_edges(self):
        # B->E is a oneway spur ending at E, which has no further outgoing
        # primal edges -> segment B->E must have out-degree 0 in the line graph
        G = _make_primal_graph()
        L = build_line_graph(G)
        seg_BE = f"{B}_{E}_0"
        assert L.out_degree(seg_BE) == 0
        # both A->B and C->B feed into B, so both connect onto the B->E spur
        assert L.has_edge(f"{A}_{B}_0", seg_BE)
        assert L.has_edge(f"{C}_{B}_0", seg_BE)
        assert L.in_degree(seg_BE) == 2

    def test_reverse_direction_not_implied_by_oneway_spur(self):
        # nothing should connect INTO B->E's "reverse" as if E->B existed
        G = _make_primal_graph()
        L = build_line_graph(G)
        assert f"{E}_{B}_0" not in L.nodes  # E->B was never a primal edge (oneway)

    def test_junction_edge_attrs_correct(self):
        G = _make_primal_graph()
        L = build_line_graph(G)
        seg_AB, seg_BC = f"{A}_{B}_0", f"{B}_{C}_0"
        edge_data = L.edges[seg_AB, seg_BC]
        assert edge_data["junction_osmid"] == B
        assert edge_data["junction_node_id"] == f"osm_{B}"
        assert edge_data["junction_x"] == pytest.approx(NODE_COORDS[B][0])
        assert edge_data["junction_y"] == pytest.approx(NODE_COORDS[B][1])


class TestNodeOrderAndEdgeIndex:
    def test_node_order_is_sorted_and_stable(self):
        G = _make_primal_graph()
        L = build_line_graph(G)
        order1 = node_order(L)
        order2 = node_order(L)
        assert order1 == order2 == sorted(L.nodes)

    def test_edge_index_shapes_and_values_consistent(self):
        G = _make_primal_graph()
        L = build_line_graph(G)
        order = node_order(L)
        ei = edge_index_array(L, order)
        assert len(ei) == 2
        assert len(ei[0]) == len(ei[1]) == L.number_of_edges()
        pos = {seg: i for i, seg in enumerate(order)}
        # spot check one real edge is represented
        seg_AB, seg_BC = f"{A}_{B}_0", f"{B}_{C}_0"
        idx = list(L.edges()).index((seg_AB, seg_BC))
        assert ei[0][idx] == pos[seg_AB]
        assert ei[1][idx] == pos[seg_BC]


class TestTables:
    def test_node_table_row_count_matches_line_graph(self):
        G = _make_primal_graph()
        L = build_line_graph(G)
        nodes_gdf = build_line_graph_node_table(L)
        assert len(nodes_gdf) == L.number_of_nodes()
        assert nodes_gdf["segment_id"].is_unique

    def test_edge_table_row_count_matches_line_graph(self):
        G = _make_primal_graph()
        L = build_line_graph(G)
        edges_gdf = build_line_graph_edge_table(L)
        assert len(edges_gdf) == L.number_of_edges()
        assert set(["segment_id_from", "segment_id_to", "junction_node_id", "geometry"]).issubset(edges_gdf.columns)


class TestValidation:
    def test_clean_graph_reports_bijection_and_zero_problems(self):
        G = _make_primal_graph()
        L = build_line_graph(G)
        nodes_gdf = build_line_graph_node_table(L)
        edges_gdf = build_line_graph_edge_table(L)
        report = validate_line_graph(G, L, nodes_gdf, edges_gdf)
        assert report["segment_id_bijection"]["is_bijection"] is True
        assert report["counts"]["primal_edges_input"] == report["counts"]["line_graph_nodes"] == 9
        assert report["self_loops"]["count"] == 0
        assert report["duplicate_edges"]["count"] == 0
        assert report["junction_consistency_check"]["inconsistent_count"] == 0

    def test_deadend_segment_flagged_as_isolated_only_if_truly_isolated(self):
        # B->E has in-degree 1 (not isolated); a TRUE isolated node needs a
        # segment with no primal neighbours on either end. Build one directly.
        G = _make_primal_graph()
        G.add_node(9001, x=80.30, y=13.10)
        G.add_node(9002, x=80.31, y=13.10)
        geom = LineString([(80.30, 13.10), (80.31, 13.10)])
        G.add_edge(9001, 9002, key=0, segment_id="9001_9002_0", length_m=1000.0, geometry=geom, highway="residential")
        L = build_line_graph(G)
        nodes_gdf = build_line_graph_node_table(L)
        edges_gdf = build_line_graph_edge_table(L)
        report = validate_line_graph(G, L, nodes_gdf, edges_gdf)
        assert report["isolated_nodes"]["count"] == 1
        assert "9001_9002_0" in report["isolated_nodes"]["segment_ids_sample"]


class TestSaveOutputs:
    def test_saves_all_files_and_reloads(self, tmp_path):
        G = _make_primal_graph()
        L = build_line_graph(G)
        nodes_gdf = build_line_graph_node_table(L)
        edges_gdf = build_line_graph_edge_table(L)
        paths = save_outputs(L, nodes_gdf, edges_gdf, out_dir=tmp_path)
        for p in paths.values():
            assert p.exists()

        with open(paths["line_graph_gpickle"], "rb") as f:
            L_reloaded = pickle.load(f)
        assert L_reloaded.number_of_nodes() == L.number_of_nodes()
        assert L_reloaded.number_of_edges() == L.number_of_edges()
        some_node = next(iter(L_reloaded.nodes(data=True)))[1]
        assert hasattr(some_node["geometry"], "wkt")

        import geopandas as gpd
        nodes_reloaded = gpd.read_file(paths["line_graph_nodes_geojson"])
        assert len(nodes_reloaded) == len(nodes_gdf)
        assert "segment_id" in nodes_reloaded.columns
