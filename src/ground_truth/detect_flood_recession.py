"""
Task 4.4 -- ground-truth feedback loop: fix the peak/receding identical-
label issue that task 3.6/4.1's training runs surfaced.

The problem (see developing.md's 3.6/4.1 notes): task 3.4 assigns
`ever_flooded` to BOTH peak and receding identically, because working.md
SS1.5's original phase design anchors both phases to the same 6 Dec 2015
Sentinel-1 pass (no phase-resolved ground truth exists). That makes
peak->receding a copy-the-input exercise, not a real test of propagation
modeling -- confirmed directly (baseline F1=0.933 on it, almost entirely
from predicting "same as peak").

The fix, using REAL data that was already confirmed reachable but never
used: task 1.9 also confirmed a FOURTH Sentinel-1 pass on 18 Dec 2015
(`S1A_IW_GRDH_1SDV_20151218T003119_20151218T003148_009089_00D0E6_767A`,
12 days after the 6 Dec peak pass). This module runs the exact same
change-detection method as task 3.1 (same threshold, same speckle filter,
same JRC permanent-water mask -- imported directly from
run_sentinel1_change_detection.py, not reimplemented) against the
24 Nov -> 18 Dec pair instead of 24 Nov -> 6 Dec. A segment that showed a
SAR flood signature at 6 Dec but no longer does at 18 Dec has genuinely
receded; task 3.4 can now use that to set flood_label[receding]=0 for it
while keeping flood_label[peak]=1.

Disclosed scope limitation: this recession signal only exists for
SAR-covered segments (5.4% of all flagged segments per task 3.4's note --
most of the 87.46% "ever flooded" comes from Bhuvan/news, which are single
static snapshots with no time axis at all, see fuse_flood_labels.py). A
segment flagged only by Bhuvan/news has no data to determine whether it
receded by 18 Dec -- it is NOT reclassified, deliberately, rather than
guessed at. This partially fixes the identical-label issue (proportional
to how much of the flooded set SAR actually covers), it doesn't eliminate
it -- disclosed as such in developing.md, not hidden.

Usage:
    python src/ground_truth/detect_flood_recession.py

Required inputs:
    data/raw/wards/study_wards.geojson                        (task 1.3)
    data/processed/ground_truth/sentinel1_flood_extent.geojson (task 3.1, the peak/6-Dec extent)
    data/processed/graph/line_graph_nodes.geojson              (task 2.2)
    GEE authenticated (task 1.9)

Outputs (data/processed/ground_truth/):
    sentinel1_recession_check_vv_db.tif   -- downloaded 18-Dec raster (for QA/reruns)
    sentinel1_recession_check_extent.geojson -- flood polygons detected at 18 Dec (same method as 3.1)
    flood_recession_validation_report.json
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ground_truth.run_sentinel1_change_detection import (  # noqa: E402
    Phase3ArtifactError,
    classify_flood_mask,
    clip_to_wards,
    download_ee_image,
    fetch_permanent_water_image,
    fetch_vv_db_image,
    load_wards_union,
    per_ward_area_breakdown,
    read_single_band,
    sieve_mask,
    vectorize_mask,
)

WARDS_PATH = REPO_ROOT / "data" / "raw" / "wards" / "study_wards.geojson"
LINE_GRAPH_NODES = REPO_ROOT / "data" / "processed" / "graph" / "line_graph_nodes.geojson"
GROUND_TRUTH_DIR = REPO_ROOT / "data" / "processed" / "ground_truth"
PEAK_FLOOD_EXTENT_PATH = GROUND_TRUTH_DIR / "sentinel1_flood_extent.geojson"
OUT_DIR = GROUND_TRUTH_DIR

GEE_PROJECT = "flood-intelligence-507219"  # same project as task 3.1/1.9
PRE_EVENT_DATE = "2015-11-24"          # same baseline as task 3.1, for a like-for-like comparison
RECESSION_CHECK_DATE = "2015-12-18"    # the 4th confirmed pass (task 1.9), unused until now
DOWNLOAD_SCALE_M = 10
STORAGE_CRS = "EPSG:4326"


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase3ArtifactError(
            f"Required artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first."
        )
    return path


def load_line_graph_geometries(path: Path = LINE_GRAPH_NODES) -> gpd.GeoDataFrame:
    _require_file(path, "src/graph/build_line_graph.py (task 2.2)")
    return gpd.read_file(path)[["segment_id", "geometry"]]


def flag_segments_intersecting(line_graph_nodes: gpd.GeoDataFrame, flood_extent_path: Path) -> set:
    """Same logic as fuse_flood_labels.py's own function of the same name
    (not imported from there to avoid a circular import -- fuse_flood_labels
    will import THIS module's recovered-segment output instead)."""
    if not flood_extent_path.exists():
        return set()
    flood_gdf = gpd.read_file(flood_extent_path)
    if len(flood_gdf) == 0:
        return set()
    flood_union = flood_gdf.geometry.union_all()
    intersects = line_graph_nodes.geometry.intersects(flood_union)
    return set(line_graph_nodes.loc[intersects, "segment_id"])


def compute_recovered_segments(
    line_graph_nodes: gpd.GeoDataFrame,
    peak_extent_path: Path,
    recession_check_extent_path: Path,
) -> set:
    """segment_ids that showed a SAR flood signature at peak (6 Dec) but no
    longer do at the 18 Dec recession-check pass -- i.e. genuinely receded,
    per real data, not an assumption. Only ever a subset of the segments
    SAR flagged at peak (see module docstring's disclosed scope limit)."""
    peak_sar_segments = flag_segments_intersecting(line_graph_nodes, peak_extent_path)
    still_flooded_at_recession_check = flag_segments_intersecting(line_graph_nodes, recession_check_extent_path)
    return peak_sar_segments - still_flooded_at_recession_check


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_recession(
    peak_segments: set, recession_check_segments: set, recovered_segments: set,
) -> dict:
    return {
        "peak_sar_flagged_segments": len(peak_segments),
        "still_flooded_at_18dec_segments": len(recession_check_segments & peak_segments),
        "recovered_segments": len(recovered_segments),
        "pct_of_peak_sar_segments_recovered": round(
            100 * len(recovered_segments) / len(peak_segments), 2
        ) if peak_segments else 0.0,
    }


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    import ee

    print(f"Loading study wards <- {WARDS_PATH}")
    try:
        wards, wards_union = load_wards_union(WARDS_PATH)
        line_graph_nodes = load_line_graph_geometries()
        _require_file(PEAK_FLOOD_EXTENT_PATH, "src/ground_truth/run_sentinel1_change_detection.py (task 3.1)")
    except Phase3ArtifactError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> {len(wards)} wards")

    print(f"\nInitializing GEE (project={GEE_PROJECT}) ...")
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception as e:
        print("\nERROR: GEE not authenticated. Run `earthengine authenticate` first.", file=sys.stderr)
        print(f"(original error: {e})", file=sys.stderr)
        raise SystemExit(1)

    ee_geom = ee.Geometry(mapping(wards_union))
    region = ee_geom.bounds()

    print(f"Fetching pre-event ({PRE_EVENT_DATE}, same baseline as task 3.1) and "
          f"recession-check ({RECESSION_CHECK_DATE}, the 4th confirmed pass, task 1.9) VV scenes ...")
    pre_img = fetch_vv_db_image(ee_geom, PRE_EVENT_DATE)
    recession_img = fetch_vv_db_image(ee_geom, RECESSION_CHECK_DATE)
    jrc_img = fetch_permanent_water_image(ee_geom)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pre_path = OUT_DIR / "sentinel1_pre_event_vv_db.tif"  # reuse task 3.1's own pre-event raster if present
    recession_path = OUT_DIR / "sentinel1_recession_check_vv_db.tif"
    jrc_path = OUT_DIR / "sentinel1_jrc_occurrence.tif"

    print("Downloading rasters (same region/scale/crs as task 3.1, so grids align) ...")
    download_ee_image(pre_img, region, pre_path)
    download_ee_image(recession_img, region, recession_path)
    download_ee_image(jrc_img, region, jrc_path)
    print(f"  -> {pre_path.name}, {recession_path.name}, {jrc_path.name}")

    pre_db, transform, crs = read_single_band(pre_path)
    recession_db, _, _ = read_single_band(recession_path)
    jrc_occurrence, _, _ = read_single_band(jrc_path)

    print("\nClassifying flood mask at 18 Dec (SAME method/threshold as task 3.1) ...")
    mask = classify_flood_mask(pre_db, recession_db, jrc_occurrence)
    print(f"  raw candidate flood pixels: {int(mask.sum())} / {mask.size}")
    mask = sieve_mask(mask)
    print(f"  after sieve: {int(mask.sum())} pixels")

    print("\nVectorizing and clipping to study wards ...")
    recession_gdf = vectorize_mask(mask, transform, crs)
    recession_gdf = clip_to_wards(recession_gdf, wards_union)
    print(f"  -> {len(recession_gdf)} flood polygon(s) still present at 18 Dec")

    recession_path_out = OUT_DIR / "sentinel1_recession_check_extent.geojson"
    recession_gdf.to_file(recession_path_out, driver="GeoJSON")
    print(f"  saved -> {recession_path_out}")

    print("\nComparing against task 3.1's peak (6 Dec) extent to find recovered segments ...")
    peak_segments = flag_segments_intersecting(line_graph_nodes, PEAK_FLOOD_EXTENT_PATH)
    recession_check_segments = flag_segments_intersecting(line_graph_nodes, recession_path_out)
    recovered_segments = compute_recovered_segments(line_graph_nodes, PEAK_FLOOD_EXTENT_PATH, recession_path_out)

    report = validate_recession(peak_segments, recession_check_segments, recovered_segments)
    if len(recession_gdf) > 0:
        utm_crs = recession_gdf.estimate_utm_crs()
        report["per_ward_flooded_area_km2_at_18dec"] = per_ward_area_breakdown(
            recession_gdf.to_crs(utm_crs), wards.to_crs(utm_crs)
        )

    recovered_path = OUT_DIR / "recovered_segments.json"
    recovered_path.write_text(json.dumps(sorted(recovered_segments)), encoding="utf-8")
    report["recovered_segments_path"] = str(recovered_path)

    report_path = OUT_DIR / "flood_recession_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved validation report -> {report_path}")
    print(json.dumps(report, indent=2, default=str))

    print("\nDone. Next: task 3.4's fuse_flood_labels.py re-run picks up "
          f"{recovered_path.name} to set flood_label[receding]=0 for genuinely recovered segments.")


if __name__ == "__main__":
    main()
