"""
Task 2.2 -- transform task 2.1's primal road graph (node = intersection,
edge = road segment) into the LINE graph (node = road segment, edge = "these
two segments meet at a shared intersection and you can drive from one
directly onto the other"). This is the graph structure the GNN actually
trains on (working.md SS1.2): every road segment becomes one graph node, so
per-segment features (task 2.3's static features, task 2.6's dynamic
rainfall features) attach naturally as node features, and X_t / edge_index /
Y_{t+1} (task 2.7) are defined over this line graph, not the primal one.

Directionality: the line graph built here is a DiGraph, with a directed edge
segment_A -> segment_B iff segment_A's head node equals segment_B's tail
node in the primal MultiDiGraph -- i.e. adjacency follows the same
one-way/traversal direction already encoded in task 2.1's graph, rather than
treating "these two segments touch" as symmetric. A two-way street (which
task 2.1 represents as two opposite directed primal edges) naturally
produces line-graph edges in both directions between its neighbours.

Usage:
    python src/graph/build_line_graph.py

Required input (data/processed/graph/, produced by task 2.1's
build_primal_graph.py):
    data/processed/graph/primal_graph.gpickle  -- the full-fidelity primal MultiDiGraph

Outputs (data/processed/graph/):
    line_graph.gpickle          -- full-fidelity NetworkX DiGraph (real shapely geometries)
    line_graph.graphml          -- portable GraphML version (geometry as WKT)
    line_graph_nodes.geojson    -- one row per road segment (= line-graph node): segment_id,
                                    endpoints, geometry, length_m, and every source OSM
                                    attribute 2.1 carried through -- ready for task 2.3's
                                    static feature computation
    line_graph_edges.geojson    -- one row per segment-to-segment adjacency: segment_id_from,
                                    segment_id_to, the shared junction node, and a Point
                                    geometry at that junction (for quick spatial QA/plotting)
    line_graph_validation_report.json -- validation report (see validate_line_graph())

Handoff contract for task 2.3/2.6/2.7: every line-graph node's id IS the
segment_id from task 2.1's segments.geojson (not a re-derived id), so a
plain join on segment_id is all that's needed to attach static/dynamic
features. edge_index for the model (task 2.7) is simply this graph's edge
list with segment_ids mapped to integer positions -- see
`node_order()`/`edge_index_array()` below.
"""
import json
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import Point

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_GRAPH_DIR = REPO_ROOT / "data" / "processed" / "graph"
PRIMAL_GRAPH_GPICKLE = PROCESSED_GRAPH_DIR / "primal_graph.gpickle"

STORAGE_CRS = "EPSG:4326"  # matches every other geospatial artifact in this repo


class Phase2ArtifactError(FileNotFoundError):
    """Raised when a required task 2.1 artifact is missing or malformed.
    Always names the exact expected path and which upstream script produces
    it, matching the convention used by build_primal_graph.py's
    Phase1ArtifactError."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase2ArtifactError(
            f"Required task 2.1 artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first "
            f"(see src/graph/README.md), or place the file there if it "
            f"was generated elsewhere."
        )
    return path


# --------------------------------------------------------------------------
# Loading the primal graph
# --------------------------------------------------------------------------

def load_primal_graph(path: Path = PRIMAL_GRAPH_GPICKLE) -> nx.MultiDiGraph:
    """Load task 2.1's primal graph from its full-fidelity pickle (the
    recommended reload path per src/graph/README.md). Using the pickle
    rather than re-parsing segments.geojson means every edge already carries
    `segment_id` / `length_m` / real shapely `geometry`, exactly as 2.1 built
    them -- single source of truth.
    """
    _require_file(path, "src/graph/build_primal_graph.py (task 2.1)")
    try:
        with open(path, "rb") as f:
            G = pickle.load(f)
    except Exception as e:
        raise Phase2ArtifactError(f"Failed to unpickle {path}: {e}") from e

    if not isinstance(G, nx.MultiDiGraph):
        raise Phase2ArtifactError(
            f"{path} did not unpickle to a NetworkX MultiDiGraph (got {type(G).__name__})."
        )
    if G.number_of_edges() == 0:
        raise Phase2ArtifactError(f"{path} loaded but contains zero edges -- nothing to build a line graph from.")

    missing_seg_id = [
        (u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if "segment_id" not in d
    ]
    if missing_seg_id:
        raise Phase2ArtifactError(
            f"{path}: {len(missing_seg_id)} edge(s) missing `segment_id` (e.g. {missing_seg_id[0]}). "
            f"This graph may predate task 2.1's segment_id annotation step, or was produced by a "
            f"different pipeline than build_primal_graph.py. Re-run task 2.1."
        )
    return G


# --------------------------------------------------------------------------
# Building the line graph
# --------------------------------------------------------------------------

def build_line_graph(G: nx.MultiDiGraph) -> nx.DiGraph:
    """Transform the primal MultiDiGraph into the line graph: one node per
    primal edge (road segment), keyed by its durable `segment_id`. A
    directed line-graph edge segment_in -> segment_out is added whenever
    segment_in's head node equals segment_out's tail node in the primal
    graph (they meet at a shared intersection and you can drive from one
    onto the other) -- this is exactly `networkx.line_graph`'s definition
    for a directed graph, but built by hand here so every line-graph node
    keeps its full attribute set (highway, length_m, oneway, lanes, bridge,
    tunnel, access, ... -- everything task 2.1 attached to the primal edge)
    rather than networkx.line_graph's bare (u, v, key)-tuple nodes with no
    attributes.

    A plain `DiGraph` (not Multi-) is correct here, not a simplification:
    for a given ordered pair (segment_in, segment_out), segment_in has
    exactly one head node and segment_out has exactly one tail node, so if
    they're adjacent at all they're adjacent at exactly one junction --
    there is no way for two distinct parallel line-graph edges to exist
    between the same ordered pair of segments.
    """
    L = nx.DiGraph()

    for u, v, k, data in G.edges(keys=True, data=True):
        seg_id = data["segment_id"]
        node_attrs = dict(data)
        node_attrs["u"] = f"osm_{u}"
        node_attrs["v"] = f"osm_{v}"
        node_attrs["u_osmid"] = u
        node_attrs["v_osmid"] = v
        node_attrs["primal_key"] = k
        L.add_node(seg_id, **node_attrs)

    outgoing_by_tail = {}  # primal node -> [segment_id, ...] of segments starting there
    incoming_by_head = {}  # primal node -> [segment_id, ...] of segments ending there
    for u, v, k, data in G.edges(keys=True, data=True):
        seg_id = data["segment_id"]
        outgoing_by_tail.setdefault(u, []).append(seg_id)
        incoming_by_head.setdefault(v, []).append(seg_id)

    for n, node_data in G.nodes(data=True):
        segs_in = incoming_by_head.get(n, [])
        segs_out = outgoing_by_tail.get(n, [])
        for seg_in in segs_in:
            for seg_out in segs_out:
                L.add_edge(
                    seg_in,
                    seg_out,
                    junction_node_id=f"osm_{n}",
                    junction_osmid=n,
                    junction_x=node_data.get("x"),
                    junction_y=node_data.get("y"),
                )

    return L


def node_order(L: nx.DiGraph) -> list:
    """Deterministic ordering of line-graph node ids (segment_ids), sorted
    for reproducibility. Task 2.7 (model input schema) should build its
    node-index mapping FROM this function rather than relying on Python
    dict/insertion order, so X_t rows and edge_index stay aligned across
    reruns/reloads.
    """
    return sorted(L.nodes)


def edge_index_array(L: nx.DiGraph, order: list = None):
    """Return an edge_index in PyTorch-Geometric's convention: a (2, E)
    list of [source_positions, target_positions], with positions given by
    `order` (defaults to node_order(L)). This is a thin convenience for
    task 2.7/2.8, not required by 2.2 itself -- kept here so the position
    mapping is defined in exactly one place.
    """
    if order is None:
        order = node_order(L)
    pos = {seg_id: i for i, seg_id in enumerate(order)}
    src = [pos[u] for u, v in L.edges()]
    dst = [pos[v] for u, v in L.edges()]
    return [src, dst]


# --------------------------------------------------------------------------
# Node / edge tables
# --------------------------------------------------------------------------

def build_line_graph_node_table(L: nx.DiGraph) -> gpd.GeoDataFrame:
    """One row per line-graph node (= road segment), carrying every
    attribute the primal edge had (task 2.1's segment_id, length_m,
    geometry, and source OSM tags). This is the table task 2.3 joins
    static features onto.
    """
    records = []
    for seg_id, data in L.nodes(data=True):
        rec = {"segment_id": seg_id}
        rec.update({k: v for k, v in data.items() if k != "geometry"})
        rec["geometry"] = data.get("geometry")
        records.append(rec)
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=STORAGE_CRS)
    return gdf.set_index("segment_id", drop=False)


def build_line_graph_edge_table(L: nx.DiGraph) -> gpd.GeoDataFrame:
    """One row per segment-to-segment adjacency, with a Point geometry at
    the shared junction (for quick spatial QA/plotting -- edges of a line
    graph have no geometry of their own, so this is a convenience, not a
    structural necessity).
    """
    records = []
    for seg_in, seg_out, data in L.edges(data=True):
        x, y = data.get("junction_x"), data.get("junction_y")
        geom = Point(x, y) if x is not None and y is not None else None
        records.append({
            "segment_id_from": seg_in,
            "segment_id_to": seg_out,
            "junction_node_id": data.get("junction_node_id"),
            "junction_osmid": data.get("junction_osmid"),
            "geometry": geom,
        })
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=STORAGE_CRS)
    return gdf


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_line_graph(
    G: nx.MultiDiGraph,
    L: nx.DiGraph,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    sample_size: int = 5,
) -> dict:
    report = {}

    report["counts"] = {
        "primal_edges_input": int(G.number_of_edges()),
        "line_graph_nodes": int(L.number_of_nodes()),
        "line_graph_edges": int(L.number_of_edges()),
    }

    # bijection check: every primal edge (segment) must map to exactly one
    # line-graph node, and vice versa -- this IS the 2.1<->2.2 handoff
    # contract, so verify it explicitly rather than assuming.
    primal_seg_ids = {data["segment_id"] for _, _, data in G.edges(data=True)}
    line_graph_seg_ids = set(L.nodes)
    report["segment_id_bijection"] = {
        "primal_segment_count": len(primal_seg_ids),
        "line_graph_node_count": len(line_graph_seg_ids),
        "missing_from_line_graph": sorted(primal_seg_ids - line_graph_seg_ids)[:20],
        "extra_in_line_graph": sorted(line_graph_seg_ids - primal_seg_ids)[:20],
        "is_bijection": primal_seg_ids == line_graph_seg_ids,
    }

    isolated = list(nx.isolates(L))
    report["isolated_nodes"] = {
        "count": len(isolated),
        "segment_ids_sample": isolated[:20],
        "note": "A segment with no isolated-node entry has neither an upstream nor a downstream neighbour -- typically a dead-end/cul-de-sac segment at both ends, or an artifact at the AOI boundary.",
    }

    weak_components = list(nx.weakly_connected_components(L))
    weak_sizes = sorted((len(c) for c in weak_components), reverse=True)
    n_nodes = L.number_of_nodes() or 1
    report["connected_components"] = {
        "weakly_connected": {
            "count": len(weak_components),
            "largest_size": weak_sizes[0] if weak_sizes else 0,
            "largest_pct_of_nodes": round(100 * (weak_sizes[0] if weak_sizes else 0) / n_nodes, 2),
            "component_sizes_top10": weak_sizes[:10],
        },
    }

    self_loop_edges = list(nx.selfloop_edges(L))
    report["self_loops"] = {
        "count": len(self_loop_edges),
        "examples": self_loop_edges[:10],
        "note": "A line-graph self-loop means a primal segment's own head==tail (a primal self-loop, already flagged by task 2.1's validation) or, more subtly, a segment immediately followed by itself in the traversal graph.",
    }

    dup_mask = edges_gdf.duplicated(subset=["segment_id_from", "segment_id_to"], keep=False)
    report["duplicate_edges"] = {
        "count": int(dup_mask.sum()),
        "note": "Should always be 0 -- see build_line_graph()'s docstring for why an ordered (from, to) pair can only be adjacent at one junction.",
    }

    # spot-check: for every edge, confirm the shared junction is really the
    # head of `from` and the tail of `to` (catches any indexing bug directly,
    # rather than trusting the construction logic).
    bad_junction_edges = []
    for _, row in edges_gdf.head(min(len(edges_gdf), 2000)).iterrows():
        from_node = nodes_gdf.loc[row["segment_id_from"]]
        to_node = nodes_gdf.loc[row["segment_id_to"]]
        if from_node["v"] != to_node["u"] or from_node["v"] != row["junction_node_id"]:
            bad_junction_edges.append((row["segment_id_from"], row["segment_id_to"]))
    report["junction_consistency_check"] = {
        "checked": int(min(len(edges_gdf), 2000)),
        "inconsistent_count": len(bad_junction_edges),
        "examples": bad_junction_edges[:10],
    }

    report["samples"] = {
        "nodes": nodes_gdf.drop(columns="geometry").head(sample_size).to_dict(orient="records"),
        "edges": edges_gdf.drop(columns="geometry", errors="ignore").head(sample_size).to_dict(orient="records"),
    }

    return report


# --------------------------------------------------------------------------
# Saving outputs
# --------------------------------------------------------------------------

def save_outputs(L: nx.DiGraph, nodes_gdf, edges_gdf, out_dir: Path = PROCESSED_GRAPH_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    gpickle_path = out_dir / "line_graph.gpickle"
    with open(gpickle_path, "wb") as f:
        pickle.dump(L, f, protocol=pickle.HIGHEST_PROTOCOL)

    L_graphml = L.copy()
    for _, data in L_graphml.nodes(data=True):
        geom = data.get("geometry")
        if geom is not None and hasattr(geom, "wkt"):
            data["geometry"] = geom.wkt
        for k, v in list(data.items()):
            if isinstance(v, (list, dict)):
                data[k] = json.dumps(v)
    graphml_path = out_dir / "line_graph.graphml"
    nx.write_graphml(L_graphml, graphml_path)

    nodes_path = out_dir / "line_graph_nodes.geojson"
    nodes_gdf.reset_index(drop=True).to_file(nodes_path, driver="GeoJSON")

    edges_path = out_dir / "line_graph_edges.geojson"
    edges_gdf.reset_index(drop=True).to_file(edges_path, driver="GeoJSON")

    return {
        "line_graph_gpickle": gpickle_path,
        "line_graph_graphml": graphml_path,
        "line_graph_nodes_geojson": nodes_path,
        "line_graph_edges_geojson": edges_path,
    }


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    print(f"Loading task 2.1 primal graph <- {PRIMAL_GRAPH_GPICKLE}")
    try:
        G = load_primal_graph()
    except Phase2ArtifactError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> {G.number_of_nodes()} primal nodes, {G.number_of_edges()} primal edges (segments)")

    print("\nBuilding line graph (segment = node) ...")
    L = build_line_graph(G)
    print(f"  -> {L.number_of_nodes()} line-graph nodes, {L.number_of_edges()} line-graph edges (adjacencies)")

    nodes_gdf = build_line_graph_node_table(L)
    edges_gdf = build_line_graph_edge_table(L)

    print("\nSaving outputs ...")
    paths = save_outputs(L, nodes_gdf, edges_gdf)
    for label, p in paths.items():
        print(f"  {label}: {p}")

    print("\nRunning validation ...")
    report = validate_line_graph(G, L, nodes_gdf, edges_gdf)
    report_path = PROCESSED_GRAPH_DIR / "line_graph_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  validation report -> {report_path}")

    print("\n=== Validation summary ===")
    print(json.dumps(report, indent=2, default=str))

    print("\nDone. Next: task 2.3 (static node features) joins onto")
    print(f"  {paths['line_graph_nodes_geojson']} by `segment_id` -- one row per line-graph node already.")


if __name__ == "__main__":
    main()
