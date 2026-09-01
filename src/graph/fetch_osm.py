"""
Task 1.1 / 1.2 / 1.3 — pull OSM road network + drainage/waterway layer for the
confirmed 16-ward Chennai study cluster, and save the ward boundary subset
itself. See developing.md Phase 0 notes and working.md SS1.9 for how this
ward list and event window were confirmed.

Study wards (Adyar river corridor, geometrically confirmed contiguous):
    142, 168, 169, 170-182

Usage:
    python src/graph/fetch_osm.py

Outputs (data/raw/, not committed -- see data/README.md):
    data/raw/wards/study_wards.geojson   -- the 16 ward polygons + dissolved AOI
    data/raw/osm/roads.graphml           -- drivable road network (osmnx graph)
    data/raw/osm/roads_edges.geojson     -- same, as a flat edge GeoDataFrame
    data/raw/osm/waterways.geojson       -- waterway=* features (drain/ditch/stream/...)
    data/raw/osm/coverage_summary.json   -- counts + tag breakdown for task 1.4
"""
import json
from pathlib import Path

import geopandas as gpd
import osmnx as ox
from shapely.geometry import shape
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parents[2]
WARDS_SRC = REPO_ROOT / "data" / "raw" / "wards" / "chennai_wards_full.geojson"
OUT_WARDS = REPO_ROOT / "data" / "raw" / "wards" / "study_wards.geojson"
OUT_OSM_DIR = REPO_ROOT / "data" / "raw" / "osm"

STUDY_WARDS = [142, 168, 169] + list(range(170, 183))

# Waterway tags that count as "drainage" per working.md SS1.7/SS1.8
DRAINAGE_WATERWAY_VALUES = {"drain", "ditch", "stream", "canal", "river", "brook"}


def load_study_ward_polygons() -> gpd.GeoDataFrame:
    if not WARDS_SRC.exists():
        raise FileNotFoundError(
            f"{WARDS_SRC} not found. Copy the DataMeet Chennai Wards.geojson there first "
            "(github.com/datameet/Municipal_Spatial_Data/blob/master/Chennai/Wards.geojson)."
        )
    gdf = gpd.read_file(WARDS_SRC)
    study = gdf[gdf["Ward_No"].isin(STUDY_WARDS)].copy()
    if len(study) != len(STUDY_WARDS):
        found = sorted(study["Ward_No"].tolist())
        missing = sorted(set(STUDY_WARDS) - set(found))
        raise ValueError(f"Expected {len(STUDY_WARDS)} wards, found {len(study)}. Missing: {missing}")
    study = study.set_crs(epsg=4326, allow_override=True) if study.crs is None else study.to_crs(epsg=4326)
    return study


def main():
    OUT_WARDS.parent.mkdir(parents=True, exist_ok=True)
    OUT_OSM_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(STUDY_WARDS)} study wards from {WARDS_SRC.name} ...")
    study = load_study_ward_polygons()
    aoi_polygon = unary_union(study.geometry.values)
    print(f"  -> {len(study)} wards loaded, dissolved AOI area (deg^2): {aoi_polygon.area:.6f}")
    study.to_file(OUT_WARDS, driver="GeoJSON")
    print(f"  saved -> {OUT_WARDS}")

    print("\nFetching drivable road network from OSM (osmnx.graph_from_polygon) ...")
    G = ox.graph_from_polygon(aoi_polygon, network_type="drive", retain_all=True, truncate_by_edge=True)
    n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
    print(f"  -> {n_nodes} nodes, {n_edges} edges")
    ox.save_graphml(G, OUT_OSM_DIR / "roads.graphml")
    edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
    edges_gdf.reset_index().to_file(OUT_OSM_DIR / "roads_edges.geojson", driver="GeoJSON")
    print(f"  saved -> {OUT_OSM_DIR / 'roads.graphml'}")
    print(f"  saved -> {OUT_OSM_DIR / 'roads_edges.geojson'}")

    # connectivity check (weakly connected components) -- relevant to the
    # narrow 168/169 neck flagged in the feasibility report
    import networkx as nx
    n_components = nx.number_weakly_connected_components(G)
    largest = max(nx.weakly_connected_components(G), key=len)
    print(f"  weakly connected components: {n_components} (largest = {len(largest)}/{n_nodes} nodes)")

    print("\nFetching waterway features from OSM (osmnx.features_from_polygon) ...")
    try:
        waterways = ox.features_from_polygon(aoi_polygon, tags={"waterway": True})
    except ox._errors.InsufficientResponseError:
        waterways = gpd.GeoDataFrame(columns=["waterway", "geometry"], geometry="geometry", crs="EPSG:4326")
    print(f"  -> {len(waterways)} waterway features")
    tag_counts = waterways["waterway"].value_counts().to_dict() if "waterway" in waterways.columns and len(waterways) else {}
    print(f"  tag breakdown: {tag_counts}")
    drainage_count = sum(v for k, v in tag_counts.items() if k in DRAINAGE_WATERWAY_VALUES)
    print(f"  of which drainage-relevant ({sorted(DRAINAGE_WATERWAY_VALUES)}): {drainage_count}")

    if len(waterways):
        # osmnx does NOT clip returned ways to the query polygon -- a way that only
        # partially crosses the AOI (e.g. a long river/canal) comes back in full. Clip
        # explicitly, or one long way can dwarf the whole study area on a map/analysis.
        waterways_out = waterways.reset_index()
        waterways_out["geometry"] = waterways_out.geometry.intersection(aoi_polygon)
        before = len(waterways_out)
        waterways_out = waterways_out[~waterways_out.geometry.is_empty & waterways_out.geometry.notna()]
        if len(waterways_out) < before:
            print(f"  clipped to AOI: {before} -> {len(waterways_out)} features (some fell fully outside after precise clip)")
        # osmnx returns some non-serializable columns (lists) sometimes; stringify object columns that aren't geometry
        for col in waterways_out.columns:
            if col != "geometry" and waterways_out[col].dtype == object:
                waterways_out[col] = waterways_out[col].apply(lambda v: str(v) if isinstance(v, (list, dict)) else v)
        waterways_out.to_file(OUT_OSM_DIR / "waterways.geojson", driver="GeoJSON")
        print(f"  saved -> {OUT_OSM_DIR / 'waterways.geojson'} (clipped to AOI)")
    else:
        print("  WARNING: zero waterway features returned -- drainage coverage check (1.4) will need real scrutiny")

    summary = {
        "study_wards": STUDY_WARDS,
        "roads": {
            "nodes": n_nodes,
            "edges": n_edges,
            "weakly_connected_components": n_components,
            "largest_component_nodes": len(largest),
        },
        "waterways": {
            "total_features": int(len(waterways)),
            "tag_breakdown": {str(k): int(v) for k, v in tag_counts.items()},
            "drainage_relevant_count": int(drainage_count),
        },
    }
    with open(OUT_OSM_DIR / "coverage_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved coverage summary -> {OUT_OSM_DIR / 'coverage_summary.json'}")
    print("\nDone. Next: task 1.4 (manual coverage check) -- eyeball waterways.geojson against")
    print("satellite/known drains in the study wards; osmnx drainage tagging in India is known")
    print("to be sparse/inconsistent (working.md SS1.8) so a low count here is expected, not a bug.")


if __name__ == "__main__":
    main()
