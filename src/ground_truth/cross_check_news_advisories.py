"""
Task 3.3 -- cross-check task 3.1/3.2's satellite-derived flood extents
against real news/advisory text, geocoding named roads/localities via task
2.9's gazetteer + task 2.10's fuzzy matcher (working.md SS1.5: "News/
advisory cross-check -- segments matching named flooded roads/localities
are manually confirmed or added, catching what SAR misses").

Method:
  1. Re-fetch every real paragraph from task 1.13's source pages (reusing
     task 2.9's own fetch_all_paragraphs() -- not duplicating that logic),
     and run task 2.10's geoparse_text() over each one to find place
     mentions and resolve them to coordinates.
  2. For each uniquely-mentioned place, resolve it to line-graph segments:
       - First, try an EXACT match against every line-graph segment's OSM
         `name` field (task 2.1/2.2 carried that through to every node),
         regardless of the gazetteer's `type` -- not just for type=="road".
         Task 2.9's OSM-verified additions are all generically tagged
         "distress_text_candidate" even when the matched feature actually
         is a road (e.g. "Gandhi Road"), so gating this on type=="road"
         would miss a genuine, more-precise match for those; an exact
         string match against a real segment name has no real false-
         positive risk, so trying it broadly is strictly better.
       - If that finds nothing, fall back to every segment in the place's
         containing study ward -- coarser, ward-level precision, same tier
         as task 3.2's raster cross-check, disclosed as such.
  3. Cross-check the resulting "news-flagged" segment set against task
     3.1's Sentinel-1 polygon and task 3.2's Bhuvan/NRSC polygon: how much
     agrees, and -- the actual point of this task -- how many news-flagged
     segments are NEWS-ONLY (neither satellite source flagged them). That
     news-only set is the concrete "catches what SAR misses" this task is
     named for, not just a citation of the idea.

This is qualitative/coarse corroborating evidence, same tier as task 3.2,
not primary ground truth (task 3.1 stays primary, working.md SS1.5) -- and
it is NOT phase-resolved: the real corpus (task 1.13) has no reliable
per-passage date metadata (checked directly -- only the ReliefWeb sitrep
source's own URL carries any date range, and that's one of 19 passages),
so a news-flagged segment means "flooded at some point during the event,"
not "flooded in phase X." Task 3.4's fusion must decide how to use a
phase-unresolved signal -- this task doesn't guess at a phase in its place.

Usage:
    python src/ground_truth/cross_check_news_advisories.py

Required inputs:
    data/raw/gazetteer/gazetteer_final.csv     (task 2.9, needs `type`/`ward_no`)
    data/raw/gazetteer/gazetteer_final.json    (task 2.9, name->coordinate)
    data/processed/graph/line_graph_nodes.geojson (task 2.2)
    data/raw/wards/study_wards.geojson          (task 1.3)
    Network access to task 1.13's source pages (see finalize_gazetteer.py)

Optional (cross-check only, not required to run):
    data/processed/ground_truth/sentinel1_flood_extent.geojson  (task 3.1)
    data/processed/ground_truth/bhuvan_nrsc_flood_extent.geojson (task 3.2)

Outputs (data/processed/ground_truth/):
    news_cross_check_places.csv        -- one row per uniquely-mentioned place
    news_cross_check_segments.geojson  -- one row per news-flagged line-graph segment
    news_cross_check_validation_report.json
"""
import ast
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nlp.finalize_gazetteer import fetch_all_paragraphs  # noqa: E402
from src.nlp.fuzzy_geoparse import geoparse_text, load_gazetteer  # noqa: E402

GAZETTEER_CSV = REPO_ROOT / "data" / "raw" / "gazetteer" / "gazetteer_final.csv"
GAZETTEER_JSON = REPO_ROOT / "data" / "raw" / "gazetteer" / "gazetteer_final.json"
LINE_GRAPH_NODES = REPO_ROOT / "data" / "processed" / "graph" / "line_graph_nodes.geojson"
WARDS_PATH = REPO_ROOT / "data" / "raw" / "wards" / "study_wards.geojson"
SENTINEL1_PATH = REPO_ROOT / "data" / "processed" / "ground_truth" / "sentinel1_flood_extent.geojson"
BHUVAN_PATH = REPO_ROOT / "data" / "processed" / "ground_truth" / "bhuvan_nrsc_flood_extent.geojson"
OUT_DIR = REPO_ROOT / "data" / "processed" / "ground_truth"


class Phase3ArtifactError(FileNotFoundError):
    """Raised when a required task 1.3/2.2/2.9 artifact is missing.
    Matches the *ArtifactError convention used across src/graph,
    src/ground_truth, src/nlp."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase3ArtifactError(
            f"Required artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first."
        )
    return path


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_gazetteer_table(path: Path = GAZETTEER_CSV) -> pd.DataFrame:
    _require_file(path, "src/nlp/finalize_gazetteer.py (task 2.9)")
    df = pd.read_csv(path)
    required = {"name", "type", "ward_no"}
    missing = required - set(df.columns)
    if missing:
        raise Phase3ArtifactError(f"{path} is missing required column(s) {sorted(missing)}.")
    return df


def load_line_graph_nodes(path: Path = LINE_GRAPH_NODES) -> gpd.GeoDataFrame:
    _require_file(path, "src/graph/build_line_graph.py (task 2.2)")
    return gpd.read_file(path)


def load_wards(path: Path = WARDS_PATH) -> gpd.GeoDataFrame:
    _require_file(path, "task 1.3")
    return gpd.read_file(path)


# --------------------------------------------------------------------------
# Geoparsing the real corpus (network-dependent -- validated by a real run)
# --------------------------------------------------------------------------

def geoparse_corpus(gazetteer: dict) -> pd.DataFrame:
    """Re-fetch every real paragraph (task 1.13's sources, via task 2.9's
    own fetch_all_paragraphs()) and run task 2.10's geoparse_text() over
    each one. Returns one row per resolved mention (a place can appear
    many times, from different paragraphs -- aggregated later)."""
    paragraphs = fetch_all_paragraphs()
    rows = []
    for p in paragraphs:
        for m in geoparse_text(p["paragraph_text"], gazetteer):
            rows.append({
                "matched_name": m["matched_name"], "mention_text": m["text"], "score": m["score"],
                "lon": m["lon"], "lat": m["lat"], "source_url": p["source_url"], "source_label": p["source_label"],
            })
    return pd.DataFrame(rows, columns=["matched_name", "mention_text", "score", "lon", "lat", "source_url", "source_label"])


# --------------------------------------------------------------------------
# Resolving places to line-graph segments (pure logic -- unit-tested)
# --------------------------------------------------------------------------

def normalize_name_field(value) -> list:
    """task 1.14/2.1's OSM `name` field is sometimes a single string,
    sometimes a list (a segment assembled from OSM ways with conflicting
    name tags -- see src/graph/build_gazetteer.py's own comment on this),
    and GeoJSON round-tripping can turn that list into either a real
    Python list/ndarray or a stringified repr like "['A', 'B']". Returns a
    flat list of name strings, whatever the original shape was.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return [str(v) for v in value.tolist()]
    except ImportError:
        pass
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(v) for v in parsed]
        except (ValueError, SyntaxError):
            pass
    return [text]


def find_segments_by_road_name(road_name: str, line_graph_nodes: gpd.GeoDataFrame) -> list:
    """segment_ids whose (possibly multi-valued) OSM name field contains
    `road_name` (case-insensitive, exact)."""
    target = road_name.strip().lower()
    matches = []
    for _, row in line_graph_nodes.iterrows():
        names = normalize_name_field(row.get("name"))
        if any(n.strip().lower() == target for n in names):
            matches.append(row["segment_id"])
    return matches


def find_segments_in_ward(ward_no, line_graph_nodes: gpd.GeoDataFrame, wards: gpd.GeoDataFrame) -> list:
    """Every line-graph segment whose representative point falls inside
    ward `ward_no` -- the ward-level fallback for non-road place types.
    Compares as numbers, not strings: the gazetteer's ward_no column is
    float64 (pandas' default for a column with any NaN in it, e.g.
    task 2.9's distress_text_candidate rows with no ward match), so
    `str(177.0) == "177.0"` would never equal wards.geojson's own
    `Ward_No` (int32, `str(177) == "177"`) under naive string comparison.
    """
    ward_rows = wards[wards["Ward_No"].astype(int) == int(float(ward_no))]
    if len(ward_rows) == 0:
        return []
    ward_geom = ward_rows.geometry.iloc[0]
    pts = line_graph_nodes.geometry.representative_point()
    inside = pts.within(ward_geom)
    return line_graph_nodes.loc[inside, "segment_id"].tolist()


def resolve_matches_to_segments(
    geoparsed: pd.DataFrame, gazetteer_table: pd.DataFrame, line_graph_nodes: gpd.GeoDataFrame, wards: gpd.GeoDataFrame
) -> pd.DataFrame:
    """One row per uniquely-mentioned place: its gazetteer type, resolution
    method, ward, resolved segment_ids, and how many/which paragraphs
    mentioned it. See module docstring for the road-name-match-vs-ward-
    fallback resolution rule.
    """
    gaz_by_name = gazetteer_table.drop_duplicates(subset="name").set_index("name")
    rows = []
    for matched_name, group in geoparsed.groupby("matched_name"):
        if matched_name not in gaz_by_name.index:
            continue
        gaz_row = gaz_by_name.loc[matched_name]
        place_type = gaz_row["type"]
        ward_no = gaz_row["ward_no"]

        # Try the exact segment-name match first regardless of gazetteer
        # `type` -- not just when type=="road". Task 2.9's OSM-verified
        # additions are all generically tagged "distress_text_candidate"
        # even when the matched OSM feature actually is a road (e.g.
        # "Gandhi Road"), so restricting this to type=="road" would miss
        # a genuine, more-precise match for those. An exact string match
        # against a real line-graph segment name has no real false-positive
        # risk, so trying it broadly is strictly better, never worse.
        segment_ids = find_segments_by_road_name(matched_name, line_graph_nodes)
        if segment_ids:
            resolution = "segment_name_match"
        elif pd.notna(ward_no):
            segment_ids = find_segments_in_ward(ward_no, line_graph_nodes, wards)
            resolution = "ward_fallback"
        else:
            resolution = "unresolved_no_ward"

        rows.append({
            "matched_name": matched_name, "type": place_type, "ward_no": ward_no,
            "resolution_method": resolution, "n_segments": len(segment_ids),
            "segment_ids": ";".join(map(str, segment_ids)),
            "n_mentions": len(group), "source_urls": ";".join(sorted(group["source_url"].unique())),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Cross-check against task 3.1/3.2
# --------------------------------------------------------------------------

def cross_check_against_satellite(
    resolved: pd.DataFrame, line_graph_nodes: gpd.GeoDataFrame,
    sentinel1_path: Path = SENTINEL1_PATH, bhuvan_path: Path = BHUVAN_PATH,
) -> dict:
    """How much does the news-flagged segment set agree with task 3.1/3.2's
    satellite-derived flood extents -- and, the actual point of this task,
    how many news-flagged segments are NEWS-ONLY (neither satellite source
    flagged them)."""
    all_news_segments = set()
    for ids in resolved["segment_ids"]:
        all_news_segments.update(s for s in ids.split(";") if s)

    report = {"n_news_flagged_segments": len(all_news_segments)}
    if not all_news_segments:
        return report

    news_gdf = line_graph_nodes[line_graph_nodes["segment_id"].isin(all_news_segments)]

    for label, path in [("sentinel1", sentinel1_path), ("bhuvan_nrsc", bhuvan_path)]:
        if not path.exists():
            report[f"{label}_overlap"] = {"available": False}
            continue
        flood_gdf = gpd.read_file(path)
        if len(flood_gdf) == 0:
            report[f"{label}_overlap"] = {"available": True, "n_overlapping_segments": 0}
            continue
        flood_union = flood_gdf.geometry.union_all()
        overlaps = news_gdf.geometry.intersects(flood_union)
        report[f"{label}_overlap"] = {
            "available": True,
            "n_overlapping_segments": int(overlaps.sum()),
            "pct_of_news_flagged": round(100 * overlaps.sum() / len(news_gdf), 1),
        }

    both_available = report.get("sentinel1_overlap", {}).get("available") and report.get("bhuvan_nrsc_overlap", {}).get("available")
    if both_available:
        s1_gdf = gpd.read_file(sentinel1_path)
        bh_gdf = gpd.read_file(bhuvan_path)
        satellite_union = None
        if len(s1_gdf):
            satellite_union = s1_gdf.geometry.union_all()
        if len(bh_gdf):
            satellite_union = bh_gdf.geometry.union_all() if satellite_union is None else satellite_union.union(bh_gdf.geometry.union_all())
        if satellite_union is not None:
            news_only = ~news_gdf.geometry.intersects(satellite_union)
            report["news_only_segments"] = {
                "count": int(news_only.sum()),
                "pct_of_news_flagged": round(100 * news_only.sum() / len(news_gdf), 1),
                "note": "Segments neither satellite source flagged -- the concrete 'catches what SAR misses' this task is named for.",
                "segment_ids_sample": news_gdf.loc[news_only, "segment_id"].tolist()[:20],
            }

    return report


# --------------------------------------------------------------------------
# Saving outputs
# --------------------------------------------------------------------------

def save_outputs(resolved: pd.DataFrame, line_graph_nodes: gpd.GeoDataFrame, out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    places_path = out_dir / "news_cross_check_places.csv"
    resolved.to_csv(places_path, index=False, encoding="utf-8")

    all_segments = set()
    for ids in resolved["segment_ids"]:
        all_segments.update(s for s in ids.split(";") if s)
    segments_gdf = line_graph_nodes[line_graph_nodes["segment_id"].isin(all_segments)][["segment_id", "geometry"]].copy()
    segments_path = out_dir / "news_cross_check_segments.geojson"
    segments_gdf.to_file(segments_path, driver="GeoJSON")

    return {"places_csv": places_path, "segments_geojson": segments_path}


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    print(f"Loading task 2.9 gazetteer <- {GAZETTEER_CSV}")
    try:
        gazetteer_table = load_gazetteer_table()
        gazetteer_dict = load_gazetteer(GAZETTEER_JSON)
        line_graph_nodes = load_line_graph_nodes()
        wards = load_wards()
    except Phase3ArtifactError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> {len(gazetteer_table)} gazetteer rows, {len(line_graph_nodes)} line-graph segments, {len(wards)} wards")

    print("\nRe-fetching real distress/news text and geoparsing it (task 2.10) ...")
    geoparsed = geoparse_corpus(gazetteer_dict)
    print(f"  -> {len(geoparsed)} resolved mentions, {geoparsed['matched_name'].nunique()} unique places")

    print("\nResolving places to line-graph segments (road-name match, else ward fallback) ...")
    resolved = resolve_matches_to_segments(geoparsed, gazetteer_table, line_graph_nodes, wards)
    print(f"  -> {len(resolved)} unique places resolved, {resolved['n_segments'].sum()} total segment-flags")
    print(resolved["resolution_method"].value_counts().to_string())

    print("\nSaving outputs ...")
    paths = save_outputs(resolved, line_graph_nodes)
    for label, p in paths.items():
        print(f"  {label}: {p}")

    print("\nCross-checking against task 3.1/3.2 ...")
    report = cross_check_against_satellite(resolved, line_graph_nodes)
    report_path = OUT_DIR / "news_cross_check_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  validation report -> {report_path}")

    print("\n=== Validation summary ===")
    print(json.dumps(report, indent=2, default=str))

    print("\nDone. This is coarse corroborating evidence (ward-level for non-road mentions),")
    print("NOT phase-resolved -- task 3.4's fusion must decide how to use a phase-unresolved signal.")


if __name__ == "__main__":
    main()
