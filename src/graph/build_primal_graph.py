"""
Task 2.1 -- build the primal road graph (node = intersection/endpoint, edge =
road segment) from Phase 1's OSM outputs, and stage the drainage layer that
task 2.3 will join against for `distance_to_drain`. This is a thin,
validation-heavy transform on top of what 1.1/1.2/1.4 already produced -- see
src/graph/README.md: "2.1: build the primal graph from roads.graphml +
waterways.geojson (this task's output *is* 2.1's input -- no further OSM
pulling needed, just the primal -> line-graph transform)".

Per working.md SS1.2, "primal graph" here is intentionally node=intersection,
edge=segment -- task 2.2 is what converts this into the line graph
(segment=node) that the GNN actually trains on. Waterways/drainage are kept
as a SEPARATE spatial layer here, not merged into road edges as traversable
segments -- 2.3 computes distance_to_drain as a road-segment-to-nearest-
drainage-geometry distance, not as graph connectivity.

Usage:
    python src/graph/build_primal_graph.py

Required inputs (data/raw/, produced by src/graph/fetch_osm.py -- tasks
1.1/1.2/1.3, see that script's docstring):
    data/raw/osm/roads.graphml         -- required (task 1.1)
    data/raw/osm/waterways.geojson     -- required, may be empty (task 1.2)

Optional inputs (used only to enrich validation, never block a run):
    data/raw/wards/study_wards.geojson -- AOI polygon for the drainage/AOI overlap check
    data/raw/osm/coverage_summary.json -- task 1.4's recorded counts, cross-checked against this run

Outputs (data/processed/graph/):
    primal_graph.gpickle    -- full-fidelity NetworkX MultiDiGraph (real shapely geometries) -- the
                               recommended way to reload this graph in Python (`pickle.load`)
    primal_graph.graphml    -- portable GraphML version (geometries stringified as WKT, matching
                               the convention already used for data/raw/osm/roads.graphml)
    nodes.geojson           -- node table: node_id, coordinates, Point geometry
    segments.geojson        -- edge/segment table: segment_id, u, v, geometry, length_m, source
                               OSM attributes -- THIS is the primal-edge <-> segment_id map that
                               task 2.2's primal->line-graph transform depends on
    drainage.geojson        -- pass-through copy of the waterway layer (unmodified geometry/CRS)
    validation_report.json  -- full machine-readable validation report (see validate_primal_graph())

Graph type: NetworkX MultiDiGraph. Directed to preserve one-way streets
(distinct (u, v) vs (v, u) edges); Multi to preserve parallel carriageways
(distinct `key` per (u, v) pair) rather than silently collapsing them into
one edge. This matches what osmnx already returns from graph_from_polygon()
and is the type task 2.2 (networkx.line_graph) expects as input.
"""
import json
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, Point

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_OSM_DIR = REPO_ROOT / "data" / "raw" / "osm"
RAW_WARDS_DIR = REPO_ROOT / "data" / "raw" / "wards"
OUT_DIR = REPO_ROOT / "data" / "processed" / "graph"

ROADS_GRAPHML = RAW_OSM_DIR / "roads.graphml"
WATERWAYS_GEOJSON = RAW_OSM_DIR / "waterways.geojson"
STUDY_WARDS_GEOJSON = RAW_WARDS_DIR / "study_wards.geojson"
COVERAGE_SUMMARY_JSON = RAW_OSM_DIR / "coverage_summary.json"

# Matches every other geospatial artifact already in this repo (see
# data/README.md, src/ground_truth/*.py) -- storage stays in EPSG:4326,
# metric operations project on demand.
STORAGE_CRS = "EPSG:4326"

# Source OSM/osmnx edge attributes worth preserving explicitly if present.
# OSM tagging is inconsistent -- any of these may legitimately be absent on
# a given edge; we never assume they all exist, we just never discard one
# that does.
ROAD_ATTRS_TO_KEEP = [
    "osmid", "highway", "name", "oneway", "reversed", "lanes", "bridge",
    "tunnel", "access", "junction", "maxspeed", "ref", "service", "width",
]

IMPLAUSIBLE_LENGTH_M = 3000.0  # soft flag threshold for a single segment, not a hard failure


class Phase1ArtifactError(FileNotFoundError):
    """Raised when a required Phase 1 artifact (task 1.1/1.2) is missing or
    malformed. Always names the exact expected path and which upstream
    script/task is supposed to have produced it, per this repo's convention
    of failing loudly with an actionable message rather than guessing."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase1ArtifactError(
            f"Required Phase 1 artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first "
            f"(see src/graph/README.md), or place the file there if it "
            f"was generated elsewhere."
        )
    return path


# --------------------------------------------------------------------------
# Loading Phase 1 artifacts
# --------------------------------------------------------------------------

def load_road_graph(path: Path = ROADS_GRAPHML) -> nx.MultiDiGraph:
    """Load task 1.1's road network exactly as osmnx saved it. osmnx's
    default `simplify=True` (used in fetch_osm.py's `graph_from_polygon`
    call) already nodes the network at true topological intersections and
    dead-ends, so no additional re-noding/splitting library is needed here
    -- we only validate that the loaded graph actually has that property
    (see validate_primal_graph: self-loops, isolated nodes, duplicate
    segments all double as noding-quality checks).
    """
    _require_file(path, "src/graph/fetch_osm.py (tasks 1.1/1.2/1.3)")
    try:
        G = ox.load_graphml(path)
    except Exception as e:
        raise Phase1ArtifactError(
            f"Failed to parse {path} as an osmnx/NetworkX GraphML file: {e}"
        ) from e

    if not isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        raise Phase1ArtifactError(
            f"{path} did not load as a NetworkX MultiDiGraph (got {type(G).__name__}). "
            f"Expected osmnx.graph_from_polygon() output."
        )
    if G.number_of_nodes() == 0:
        raise Phase1ArtifactError(f"{path} loaded but contains zero nodes.")

    missing_xy = [n for n, d in G.nodes(data=True) if "x" not in d or "y" not in d]
    if missing_xy:
        raise Phase1ArtifactError(
            f"{path}: {len(missing_xy)} node(s) missing x/y coordinates "
            f"(e.g. node {missing_xy[0]!r}). Expected every node to carry x/y "
            f"lon/lat, as osmnx always sets."
        )
    missing_length = sum(1 for _, _, d in G.edges(data=True) if "length" not in d)
    if missing_length:
        raise Phase1ArtifactError(
            f"{path}: {missing_length} edge(s) missing a `length` attribute, which "
            f"osmnx adds automatically in graph_from_polygon(). This file may have "
            f"been hand-edited or produced by a different pipeline than "
            f"src/graph/fetch_osm.py."
        )
    return G


def load_waterways(path: Path = WATERWAYS_GEOJSON) -> gpd.GeoDataFrame:
    """Load task 1.2's clipped waterway/drainage layer. An EMPTY result is
    allowed and expected in places -- src/graph/README.md's task 1.4 writeup
    documents that OSM drainage tagging in this study area is sparse/
    inconsistent. That's a data-quality fact to surface in validation, not
    an error to raise here.
    """
    _require_file(path, "src/graph/fetch_osm.py (task 1.2)")
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        raise Phase1ArtifactError(
            f"Failed to parse {path} as a GeoJSON/vector file: {e}"
        ) from e
    if gdf.crs is None:
        raise Phase1ArtifactError(
            f"{path} has no CRS defined -- cannot safely reproject or measure "
            f"distances against it."
        )
    return gdf


def load_study_wards(path: Path = STUDY_WARDS_GEOJSON):
    """Optional -- only used to enrich the drainage/AOI overlap validation
    check. Not a hard dependency of 2.1 per developing.md (2.1 depends on
    1.1, 1.2, 1.4 -- not 1.3), so its absence downgrades that one check to a
    documented warning, never a failure.
    """
    if not path.exists():
        return None
    try:
        return gpd.read_file(path)
    except Exception:
        return None


def load_coverage_summary(path: Path = COVERAGE_SUMMARY_JSON):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------------------
# Building node / segment tables
# --------------------------------------------------------------------------

def _edge_geometry(u_data: dict, v_data: dict, edge_data: dict) -> LineString:
    """osmnx only attaches an explicit `geometry` to edges whose shape needed
    to survive simplification (curved/multi-vertex ways); a straight edge
    between two nodes has no `geometry` attribute and is implicitly the
    straight line between its two endpoints. Make that explicit so every
    segment in our output carries a real geometry -- nothing implicit.
    """
    geom = edge_data.get("geometry")
    if geom is not None:
        return geom
    return LineString([(u_data["x"], u_data["y"]), (v_data["x"], v_data["y"])])


def build_node_table(G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """Every road intersection / segment endpoint becomes one row. `node_id`
    is derived from OSM's own node id (`osm_<osmid>`) -- durable across
    reruns of fetch_osm.py as long as the underlying OSM data doesn't
    change, and unambiguous.
    """
    records = []
    for n, data in G.nodes(data=True):
        records.append({
            "node_id": f"osm_{n}",
            "osmid": n,
            "x": float(data["x"]),
            "y": float(data["y"]),
            "street_count": data.get("street_count"),
            "geometry": Point(float(data["x"]), float(data["y"])),
        })
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=STORAGE_CRS)
    return gdf


def build_segment_table(G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """Every road edge becomes one row. `segment_id` is `{u}_{v}_{key}`,
    built from the (u, v, key) triple that already uniquely identifies an
    edge in a MultiDiGraph -- durable for the lifetime of this frozen
    roads.graphml (Phase 1 is not re-fetched once complete, per
    developing.md's Phase 3.5 "freeze graph structure" step), and this is
    the exact field task 2.2 uses as the new line-graph node id.
    """
    nodes = dict(G.nodes(data=True))
    records = []
    for u, v, k, data in G.edges(keys=True, data=True):
        geom = _edge_geometry(nodes[u], nodes[v], data)
        rec = {
            "segment_id": f"{u}_{v}_{k}",
            "u": f"osm_{u}",
            "v": f"osm_{v}",
            "u_osmid": u,
            "v_osmid": v,
            "key": k,
            "geometry": geom,
        }
        for attr in ROAD_ATTRS_TO_KEEP:
            val = data.get(attr)
            # osmnx sometimes stores list-valued tags (e.g. multiple osmid on a
            # merged/simplified way, or conflicting oneway values). Stringify
            # anything non-scalar so it round-trips safely through
            # GeoJSON/GraphML -- never silently drop it.
            if isinstance(val, (list, tuple)):
                val = json.dumps(list(val))
            rec[attr] = val
        rec["length_m_osmnx"] = data.get("length")
        records.append(rec)
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=STORAGE_CRS)
    return gdf


def add_metric_length(segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute segment length in a locally-appropriate PROJECTED (metric) CRS
    -- never directly in EPSG:4326 degrees. Uses geopandas' automatic UTM
    zone estimation from the data's own extent (`estimate_utm_crs`) rather
    than a hardcoded EPSG code, so this is correct regardless of which city
    the team ends up using (working.md SS0.3 documents Chennai vs. a
    Bengaluru fallback as an explicit open decision at one point).
    """
    if len(segments) == 0:
        segments = segments.copy()
        segments["length_m"] = []
        segments["_utm_crs_used"] = None
        return segments
    utm_crs = segments.estimate_utm_crs()
    projected_geom = segments.geometry.to_crs(utm_crs)
    segments = segments.copy()
    segments["length_m"] = projected_geom.length.round(3).values
    segments["_utm_crs_used"] = str(utm_crs)
    return segments


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build_primal_graph(roads_path: Path = ROADS_GRAPHML, waterways_path: Path = WATERWAYS_GEOJSON):
    """Load Phase 1 artifacts and assemble the primal graph + companion
    tables. Returns (G, nodes_gdf, segments_gdf, waterways_gdf), where G is
    the same MultiDiGraph structure osmnx produced, annotated with our
    durable node_id / segment_id / length_m so the graph object and the two
    tables are always mutually consistent (single source of truth =
    segments_gdf / nodes_gdf).
    """
    G = load_road_graph(roads_path)
    waterways_gdf = load_waterways(waterways_path)

    nodes_gdf = build_node_table(G).set_index("node_id", drop=False)
    segments_gdf = build_segment_table(G)
    segments_gdf = add_metric_length(segments_gdf)
    segments_gdf = segments_gdf.set_index("segment_id", drop=False)

    nx.set_node_attributes(G, {n: f"osm_{n}" for n in G.nodes}, "node_id")
    for u, v, k in G.edges(keys=True):
        seg_id = f"{u}_{v}_{k}"
        row = segments_gdf.loc[seg_id]
        G.edges[u, v, k]["segment_id"] = seg_id
        G.edges[u, v, k]["length_m"] = float(row["length_m"])
        G.edges[u, v, k]["geometry"] = row["geometry"]

    return G, nodes_gdf, segments_gdf, waterways_gdf


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_primal_graph(
    G: nx.MultiDiGraph,
    nodes_gdf: gpd.GeoDataFrame,
    segments_gdf: gpd.GeoDataFrame,
    waterways_gdf: gpd.GeoDataFrame,
    study_wards_gdf=None,
    coverage_summary: dict = None,
    sample_size: int = 5,
    implausible_length_m: float = IMPLAUSIBLE_LENGTH_M,
) -> dict:
    """Produce the full validation report required for task 2.1. Never
    raises on data-quality findings (zero waterways, disconnected
    components, etc.) -- those are documented facts about this specific
    study area (see src/graph/README.md task 1.4 writeup), not pipeline
    bugs. Only load_road_graph/load_waterways raise, for genuinely missing/
    malformed *inputs*.
    """
    report = {}

    report["crs"] = {
        "nodes_input_output": str(nodes_gdf.crs),
        "segments_input_output": str(segments_gdf.crs),
        "waterways": str(waterways_gdf.crs) if waterways_gdf is not None else None,
        "metric_crs_used_for_lengths": (
            segments_gdf["_utm_crs_used"].iloc[0] if len(segments_gdf) else None
        ),
    }

    report["counts"] = {
        "nodes": int(G.number_of_nodes()),
        "edges_segments": int(G.number_of_edges()),
    }

    isolated = list(nx.isolates(G))
    report["isolated_nodes"] = {
        "count": len(isolated),
        "node_ids_sample": [f"osm_{n}" for n in isolated[:20]],
    }

    weak_components = list(nx.weakly_connected_components(G))
    weak_sizes = sorted((len(c) for c in weak_components), reverse=True)
    strong_components = list(nx.strongly_connected_components(G))
    strong_sizes = sorted((len(c) for c in strong_components), reverse=True)
    n_nodes = G.number_of_nodes() or 1
    report["connected_components"] = {
        "weakly_connected": {
            "count": len(weak_components),
            "largest_size": weak_sizes[0] if weak_sizes else 0,
            "largest_pct_of_nodes": round(100 * (weak_sizes[0] if weak_sizes else 0) / n_nodes, 2),
            "component_sizes_top10": weak_sizes[:10],
        },
        "strongly_connected": {
            "count": len(strong_components),
            "largest_size": strong_sizes[0] if strong_sizes else 0,
            "largest_pct_of_nodes": round(100 * (strong_sizes[0] if strong_sizes else 0) / n_nodes, 2),
            "component_sizes_top10": strong_sizes[:10],
            "note": (
                "Expected to be far more fragmented than the weakly-connected "
                "count -- one-way streets legitimately block reverse "
                "traversal. This reflects the real road network, not a bug."
            ),
        },
    }

    self_loop_edges = list(nx.selfloop_edges(G, keys=True))
    report["self_loops"] = {
        "count": len(self_loop_edges),
        "examples": [f"{u}->{v} (key {k})" for u, v, k in self_loop_edges[:10]],
    }

    if len(segments_gdf):
        wkt_series = segments_gdf["geometry"].apply(lambda g: g.wkt if g is not None else None)
        dup_mask = segments_gdf.duplicated(subset=["u", "v"], keep=False) & wkt_series.duplicated(keep=False)
    else:
        dup_mask = segments_gdf.index.to_series().astype(bool) if len(segments_gdf) else []
    report["duplicate_segments"] = {
        "count": int(dup_mask.sum()) if len(segments_gdf) else 0,
        "note": "Same (u, v) endpoints AND identical geometry under >1 key -- distinct from legitimate parallel carriageways, which have different geometry.",
        "segment_ids_sample": segments_gdf.loc[dup_mask, "segment_id"].tolist()[:20] if len(segments_gdf) else [],
    }

    if len(segments_gdf):
        invalid_geom = segments_gdf["geometry"].isna() | ~segments_gdf["geometry"].apply(
            lambda g: g is not None and not g.is_empty and g.is_valid
        )
    else:
        invalid_geom = segments_gdf.index.to_series().astype(bool)
    report["invalid_or_missing_geometry"] = {
        "count": int(invalid_geom.sum()) if len(segments_gdf) else 0,
        "segment_ids_sample": segments_gdf.loc[invalid_geom, "segment_id"].tolist()[:20] if len(segments_gdf) else [],
    }

    if len(segments_gdf):
        zero_len = segments_gdf["length_m"] <= 0
        implausible_len = segments_gdf["length_m"] > implausible_length_m
        diff = (segments_gdf["length_m"] - segments_gdf["length_m_osmnx"]).abs()
        length_summary = {
            "min": float(segments_gdf["length_m"].min()),
            "max": float(segments_gdf["length_m"].max()),
            "mean": round(float(segments_gdf["length_m"].mean()), 2),
            "median": round(float(segments_gdf["length_m"].median()), 2),
        }
        osmnx_diff_max = round(float(diff.max()), 3)
    else:
        zero_len = implausible_len = segments_gdf.index.to_series().astype(bool)
        length_summary = {"min": None, "max": None, "mean": None, "median": None}
        osmnx_diff_max = None
    report["length_checks"] = {
        "zero_or_negative_length_count": int(zero_len.sum()) if len(segments_gdf) else 0,
        "zero_length_segment_ids_sample": segments_gdf.loc[zero_len, "segment_id"].tolist()[:20] if len(segments_gdf) else [],
        "implausible_length_threshold_m": implausible_length_m,
        "implausible_length_count": int(implausible_len.sum()) if len(segments_gdf) else 0,
        "implausible_length_segment_ids_sample": segments_gdf.loc[implausible_len, "segment_id"].tolist()[:20] if len(segments_gdf) else [],
        "length_m_summary": length_summary,
        "max_abs_diff_vs_osmnx_reported_length_m": osmnx_diff_max,
    }

    report["samples"] = {
        "nodes": nodes_gdf.drop(columns="geometry").head(sample_size).to_dict(orient="records"),
        "segments": segments_gdf.drop(columns=["geometry", "_utm_crs_used"], errors="ignore")
        .head(sample_size)
        .to_dict(orient="records"),
    }

    drainage_report = {"feature_count": int(len(waterways_gdf)) if waterways_gdf is not None else 0}
    if waterways_gdf is not None and len(waterways_gdf) and len(segments_gdf):
        utm_crs = segments_gdf["_utm_crs_used"].iloc[0]
        network_union = segments_gdf.geometry.to_crs(utm_crs).union_all()
        waterways_proj = waterways_gdf.geometry.to_crs(utm_crs)
        distances = waterways_proj.distance(network_union)
        drainage_report.update({
            "min_distance_to_road_network_m": round(float(distances.min()), 2),
            "features_within_500m_of_network": int((distances <= 500).sum()),
        })
        if study_wards_gdf is not None and len(study_wards_gdf):
            wards_union = study_wards_gdf.to_crs(utm_crs).union_all()
            drainage_report["overlaps_study_wards"] = bool(waterways_proj.intersects(wards_union).any())
        else:
            drainage_report["overlaps_study_wards"] = None
            drainage_report["note"] = (
                "study_wards.geojson not found -- overlap checked against the "
                "road-network extent only, not ward polygons."
            )
    elif waterways_gdf is not None and len(waterways_gdf) == 0:
        drainage_report["warning"] = (
            "Zero waterway features. This is documented/expected for this study "
            "area (src/graph/README.md task 1.4 writeup: sparse OSM drainage "
            "tagging), not necessarily a pipeline bug -- but distance_to_drain "
            "(task 2.3) will have low discriminative power wherever it's true."
        )
    report["drainage"] = drainage_report

    if coverage_summary is not None:
        report["cross_check_vs_task_1_4"] = {
            "task_1_4_recorded_waterway_count": coverage_summary.get("waterways", {}).get("total_features"),
            "this_run_waterway_count": int(len(waterways_gdf)) if waterways_gdf is not None else 0,
            "task_1_4_recorded_road_nodes": coverage_summary.get("roads", {}).get("nodes"),
            "this_run_road_nodes": int(G.number_of_nodes()),
            "task_1_4_recorded_road_edges": coverage_summary.get("roads", {}).get("edges"),
            "this_run_road_edges": int(G.number_of_edges()),
        }
    else:
        report["cross_check_vs_task_1_4"] = None

    return report


# --------------------------------------------------------------------------
# Saving outputs
# --------------------------------------------------------------------------

def save_outputs(G: nx.MultiDiGraph, nodes_gdf, segments_gdf, waterways_gdf, out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full-fidelity NetworkX reload -- pickle keeps real shapely geometry
    #    objects intact, no lossy stringification. This is the recommended
    #    reload path for downstream Python code (task 2.2).
    gpickle_path = out_dir / "primal_graph.gpickle"
    with open(gpickle_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 2. Portable GraphML reload -- geometry stringified as WKT, matching the
    #    convention already used for data/raw/osm/roads.graphml.
    G_graphml = G.copy()
    for _, _, data in G_graphml.edges(data=True):
        geom = data.get("geometry")
        if geom is not None and hasattr(geom, "wkt"):
            data["geometry"] = geom.wkt
        for k, v in list(data.items()):
            if isinstance(v, (list, dict)):
                data[k] = json.dumps(v)
    graphml_path = out_dir / "primal_graph.graphml"
    nx.write_graphml(G_graphml, graphml_path)

    # 3. Node table.
    nodes_path = out_dir / "nodes.geojson"
    nodes_gdf.reset_index(drop=True).to_file(nodes_path, driver="GeoJSON")

    # 4. Segment/edge table -- the primal-edge <-> segment_id map task 2.2 needs.
    segments_path = out_dir / "segments.geojson"
    segments_gdf.drop(columns=["_utm_crs_used"], errors="ignore").reset_index(drop=True).to_file(
        segments_path, driver="GeoJSON"
    )

    # 5. Drainage pass-through -- unmodified geometry/CRS. Task 2.3 picks its
    #    own working CRS at join time; we don't bake in a lossy reprojection
    #    here.
    drainage_path = out_dir / "drainage.geojson"
    if waterways_gdf is not None:
        waterways_gdf.reset_index(drop=True).to_file(drainage_path, driver="GeoJSON")
    else:
        drainage_path = None

    return {
        "primal_graph_gpickle": gpickle_path,
        "primal_graph_graphml": graphml_path,
        "nodes_geojson": nodes_path,
        "segments_geojson": segments_path,
        "drainage_geojson": drainage_path,
    }


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    print(f"Loading Phase 1 road network  <- {ROADS_GRAPHML}")
    print(f"Loading Phase 1 waterway layer <- {WATERWAYS_GEOJSON}")
    try:
        G, nodes_gdf, segments_gdf, waterways_gdf = build_primal_graph()
    except Phase1ArtifactError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    print(f"  -> {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (segments)")

    study_wards_gdf = load_study_wards()
    if study_wards_gdf is None:
        print(f"  (optional) {STUDY_WARDS_GEOJSON} not found -- drainage/AOI overlap check will be against road-network extent only")
    coverage_summary = load_coverage_summary()

    print("\nSaving outputs ...")
    paths = save_outputs(G, nodes_gdf, segments_gdf, waterways_gdf)
    for label, p in paths.items():
        print(f"  {label}: {p}")

    print("\nRunning validation ...")
    report = validate_primal_graph(G, nodes_gdf, segments_gdf, waterways_gdf, study_wards_gdf, coverage_summary)
    report_path = OUT_DIR / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  validation report -> {report_path}")

    print("\n=== Validation summary ===")
    print(json.dumps(report, indent=2, default=str))

    print("\nDone. Next: task 2.2 (primal -> line graph transform) reads")
    print(f"  {paths['primal_graph_gpickle']} (or the .graphml) and uses the")
    print("  `segment_id` already on every edge as the new line-graph node id.")


if __name__ == "__main__":
    main()
