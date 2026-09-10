"""
Task 1.14 -- draft initial gazetteer (road / locality / landmark names) for
the study wards.

This is Objective 2's spatial anchor: task 2.10's fuzzy-matching geoparser
will look up place-name mentions found in distress text (task 1.13's
corpus) against this table to resolve them to a ward/road-segment. No
gazetteer, no way to turn "flooding near Velachery bus depot" into
coordinates the graph can use.

Sources pulled (all within the 16 study wards, `data/raw/wards/study_wards.geojson`):
  - roads      -- `name` column already in `data/raw/osm/roads_edges.geojson` (task 1.1)
  - waterways  -- `name` column already in `data/raw/osm/waterways.geojson` (task 1.2)
  - localities -- OSM `place=neighbourhood/suburb/quarter/locality/village` nodes (new pull)
  - landmarks  -- OSM points commonly referenced in flood/distress reports:
                  hospitals, bus stations, railway stations, colleges, schools,
                  places of worship (amenity=*, railway=station)

Each entry also carries `name:ta` (Tamil name) when OSM has it -- this is
what makes the gazetteer usable for Tamil-language distress messages, not
just English ones, which is the whole point of Objective 2 being
"multilingual" rather than English-only.

This is a DRAFT (task 1.14) -- task 2.9 revisits it once real distress text
(task 1.13) shows which names actually get mentioned and which don't.

Usage:
    python src/nlp/build_gazetteer.py

Output:
    data/raw/gazetteer/gazetteer_draft.csv (gitignored -- rerun to regenerate)
"""
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parents[2]
WARDS = REPO_ROOT / "data" / "raw" / "wards" / "study_wards.geojson"
ROADS_EDGES = REPO_ROOT / "data" / "raw" / "osm" / "roads_edges.geojson"
WATERWAYS = REPO_ROOT / "data" / "raw" / "osm" / "waterways.geojson"
OUT_DIR = REPO_ROOT / "data" / "raw" / "gazetteer"
OUT_CSV = OUT_DIR / "gazetteer_draft.csv"

PLACE_VALUES = ["neighbourhood", "suburb", "quarter", "locality", "village"]
LANDMARK_AMENITIES = ["hospital", "bus_station", "college", "school", "place_of_worship"]


def load_aoi_and_wards():
    wards = gpd.read_file(WARDS)
    aoi = unary_union(wards.geometry.values)
    return aoi, wards


def tag_ward(gdf: gpd.GeoDataFrame, wards: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Spatial-join each row to the ward its representative point falls in."""
    pts = gdf.copy()
    pts["geometry"] = pts.geometry.representative_point()
    joined = gpd.sjoin(pts, wards[["Ward_No", "geometry"]], how="left", predicate="within")
    gdf = gdf.copy()
    gdf["ward_no"] = joined["Ward_No"].values
    return gdf


def rows_from_points(gdf: gpd.GeoDataFrame, kind: str) -> pd.DataFrame:
    if len(gdf) == 0:
        return pd.DataFrame(columns=["name", "name_ta", "type", "subtype", "ward_no", "lon", "lat"])
    pts = gdf.geometry.representative_point()
    subtype_col = "place" if kind == "locality" else "amenity" if "amenity" in gdf.columns else None
    return pd.DataFrame({
        "name": gdf.get("name"),
        "name_ta": gdf.get("name:ta"),
        "type": kind,
        "subtype": gdf[subtype_col] if subtype_col and subtype_col in gdf.columns else gdf.get("railway"),
        "ward_no": gdf["ward_no"],
        "lon": pts.x.values,
        "lat": pts.y.values,
    })


def rows_from_lines(gdf: gpd.GeoDataFrame, kind: str, subtype_col: str) -> pd.DataFrame:
    if len(gdf) == 0:
        return pd.DataFrame(columns=["name", "name_ta", "type", "subtype", "ward_no", "lon", "lat"])
    pts = gdf.geometry.representative_point()
    return pd.DataFrame({
        "name": gdf.get("name"),
        "name_ta": gdf.get("name:ta") if "name:ta" in gdf.columns else None,
        "type": kind,
        "subtype": gdf[subtype_col] if subtype_col in gdf.columns else None,
        "ward_no": gdf["ward_no"],
        "lon": pts.x.values,
        "lat": pts.y.values,
    })


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    aoi, wards = load_aoi_and_wards()
    print(f"AOI: {len(wards)} study wards.\n")

    frames = []

    print("Roads (from task 1.1's roads_edges.geojson) ...")
    roads = gpd.read_file(ROADS_EDGES)
    roads = roads[roads["name"].notna()].copy()
    roads = tag_ward(roads, wards)
    roads_rows = rows_from_lines(roads, "road", "highway")
    print(f"  -> {len(roads_rows)} named road segments")
    frames.append(roads_rows)

    print("Waterways (from task 1.2's waterways.geojson) ...")
    waterways = gpd.read_file(WATERWAYS)
    if "name" in waterways.columns:
        waterways = waterways[waterways["name"].notna()].copy()
        waterways = tag_ward(waterways, wards)
        wway_rows = rows_from_lines(waterways, "waterway", "waterway")
    else:
        wway_rows = pd.DataFrame(columns=["name", "name_ta", "type", "subtype", "ward_no", "lon", "lat"])
    print(f"  -> {len(wway_rows)} named waterways")
    frames.append(wway_rows)

    print("\nFetching OSM localities (place=neighbourhood/suburb/quarter/locality/village) ...")
    try:
        places = ox.features_from_polygon(aoi, tags={"place": PLACE_VALUES})
        places = places[places.geometry.type == "Point"] if len(places) else places
    except ox._errors.InsufficientResponseError:
        places = gpd.GeoDataFrame(columns=["place", "name", "geometry"], geometry="geometry", crs="EPSG:4326")
    places = places[places.get("name").notna()] if len(places) else places
    if len(places):
        places = places.reset_index()
        places = tag_ward(places, wards)
    place_rows = rows_from_points(places, "locality")
    print(f"  -> {len(place_rows)} named localities")
    frames.append(place_rows)

    print(f"\nFetching OSM landmarks (amenity in {LANDMARK_AMENITIES}, railway=station) ...")
    try:
        landmarks = ox.features_from_polygon(
            aoi, tags={"amenity": LANDMARK_AMENITIES, "railway": ["station"]}
        )
    except ox._errors.InsufficientResponseError:
        landmarks = gpd.GeoDataFrame(columns=["amenity", "name", "geometry"], geometry="geometry", crs="EPSG:4326")
    landmarks = landmarks[landmarks.get("name").notna()] if len(landmarks) else landmarks
    if len(landmarks):
        landmarks = landmarks.reset_index()
        landmarks = tag_ward(landmarks, wards)
    landmark_rows = rows_from_points(landmarks, "landmark")
    print(f"  -> {len(landmark_rows)} named landmarks")
    frames.append(landmark_rows)

    gaz = pd.concat(frames, ignore_index=True)
    # OSM sometimes returns a tag as a list when a way was assembled from
    # segments with conflicting values (e.g. name changes mid-road) -- these
    # aren't hashable, so drop_duplicates() below would crash on them.
    import numpy as np
    for col in ["name", "name_ta", "subtype", "ward_no"]:
        gaz[col] = gaz[col].apply(lambda v: str(v) if isinstance(v, (list, tuple, np.ndarray)) else v)
    gaz = gaz[gaz["name"].notna() & (gaz["name"].str.strip() != "")]
    before = len(gaz)
    gaz = gaz.drop_duplicates(subset=["name", "type", "ward_no"])
    gaz = gaz.sort_values(["type", "name"]).reset_index(drop=True)
    print(f"\nTotal: {before} rows -> {len(gaz)} after de-duplication (same name+type+ward)")

    n_with_ta = gaz["name_ta"].notna().sum()
    print(f"Entries with a Tamil name (name:ta) from OSM: {n_with_ta}/{len(gaz)}")

    gaz.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved -> {OUT_CSV}")
    print("\nBreakdown by type:")
    print(gaz["type"].value_counts().to_string())
    print("\nThis is a DRAFT gazetteer (task 1.14) -- task 2.9 finalizes it once task 1.13's")
    print("real distress text shows which names actually get referenced.")


if __name__ == "__main__":
    main()
