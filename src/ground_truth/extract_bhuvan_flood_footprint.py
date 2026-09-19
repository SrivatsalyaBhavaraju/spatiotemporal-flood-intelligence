"""
Task 3.2 -- extract a flood-footprint polygon from Bhuvan-sourced data for
Chennai 2015 (working.md SS1.5's "if Chennai chosen" -- Chennai was
confirmed, task 0.7).

Re-verified directly as part of this task (19 Sep 2026, re-running
verify_bhuvan_wms.py -- task 0.6 round 2): Bhuvan's actual Chennai-2015
RISAT-1/Cartosat-2 WMS layers are STILL dead -- 'ch_exp_0306dec15' and
'ch_c2_sat' both still return 'ServiceException: invalid layer'. There is no
live, exportable RISAT SAR flood footprint to extract for this event. This
was true when 0.6 checked it and is unchanged now -- re-verifying it here
rather than just citing the old result, in case the portal had been fixed
since.

What this script extracts instead: the one real Bhuvan-portal-sourced
artifact this project actually has -- task 0.6's follow-up NRSC/ISRO
hydrological-simulation flood-depth raster
(data/raw/bhuvan/fig8_georeferenced.tif, see georeference_nrsc_simulation.py
and src/ground_truth/README.md). This is NOT an observed RISAT SAR
footprint -- it's a modeled rainfall-runoff/DEM simulation obtained from the
same portal's report archive, already downgraded to opportunistic/
secondary-only by task 0.6 (verdict unchanged here). Thresholding its depth
values and vectorizing gives a polygon in the same shape as task 3.1's
Sentinel-1 output, so tasks 3.3/3.4 can consume both the same way.

Threshold note: 94% of the raster's valid pixels already show >0.1m modeled
depth -- checked directly (see validation report's depth histogram) rather
than assumed. This means the source figure's whole colored region already
IS the simulation's claimed inundation zone (a depth-within-the-flood-zone
map, not a depth-over-the-whole-city map), so MIN_DEPTH_M=0.1 here is a
noise/edge floor, not a "significant flooding" cutoff picked to produce a
particular-looking result -- most of what's classified as flood comes from
the raster's own boundary, not this threshold. Read the output as "was this
ward inside the simulation's modeled flood zone" (coarse, ward-scale,
~563m RMSE per task 0.6's georeferencing), never a segment-level signal --
the same caveat src/ground_truth/README.md already states for this raster.

Usage:
    python src/ground_truth/extract_bhuvan_flood_footprint.py

Required inputs:
    data/raw/bhuvan/fig8_georeferenced.tif  (task 0.6 follow-up)
    data/raw/wards/study_wards.geojson       (task 1.3)

Optional (cross-check only, not required to run):
    data/processed/ground_truth/sentinel1_flood_extent.geojson  (task 3.1)

Outputs (data/processed/ground_truth/):
    bhuvan_nrsc_flood_extent.geojson
    bhuvan_nrsc_extraction_validation_report.json
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ground_truth.run_sentinel1_change_detection import (  # noqa: E402
    Phase3ArtifactError,
    clip_to_wards,
    load_wards_union,
    per_ward_area_breakdown,
    read_single_band,
    sieve_mask,
    vectorize_mask,
)
NRSC_SIMULATION_PATH = REPO_ROOT / "data" / "raw" / "bhuvan" / "fig8_georeferenced.tif"
WARDS_PATH = REPO_ROOT / "data" / "raw" / "wards" / "study_wards.geojson"
SENTINEL1_FLOOD_EXTENT_PATH = REPO_ROOT / "data" / "processed" / "ground_truth" / "sentinel1_flood_extent.geojson"
OUT_DIR = REPO_ROOT / "data" / "processed" / "ground_truth"

MIN_DEPTH_M = 0.1     # noise/edge floor, not a "significant flooding" cutoff -- see module docstring
MIN_SIEVE_PIXELS = 4  # raster is coarse (~22m/px, vs 3.1's 10m/px) -- a smaller MMU than task 3.1's


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase3ArtifactError(
            f"Required artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first."
        )
    return path


# --------------------------------------------------------------------------
# Loading / thresholding (pure array math -- unit-tested with synthetic data)
# --------------------------------------------------------------------------

def threshold_depth_mask(depth: np.ndarray, min_depth_m: float = MIN_DEPTH_M) -> np.ndarray:
    """Boolean flood mask: modeled depth exceeds `min_depth_m`. NaN pixels
    (outside the georeferenced figure's actual extent) are never flagged."""
    valid = ~np.isnan(depth)
    return (depth > min_depth_m) & valid


# --------------------------------------------------------------------------
# Cross-check against task 3.1's independently-sourced Sentinel-1 result
# --------------------------------------------------------------------------

def cross_check_against_sentinel1(bhuvan_gdf: gpd.GeoDataFrame, sentinel1_path: Path = SENTINEL1_FLOOD_EXTENT_PATH) -> dict:
    """Spatial overlap between this (coarse, simulation-derived) footprint
    and task 3.1's independently-sourced Sentinel-1 SAR result. Given how
    different the two methods and error profiles are (a ~563m-RMSE
    georeferenced simulation vs. 10m SAR change detection), even modest
    overlap is corroborating -- their disagreement is expected too (SAR
    underestimates urban flooding per working.md SS1.5, and this raster
    likely overestimates it, since almost its whole extent is "flooded").
    """
    if not sentinel1_path.exists() or len(bhuvan_gdf) == 0:
        return {"available": False, "reason": "task 3.1 output or Bhuvan/NRSC polygons not available"}

    sentinel1_gdf = gpd.read_file(sentinel1_path)
    if len(sentinel1_gdf) == 0:
        return {"available": False, "reason": "task 3.1 produced zero flood polygons"}

    utm_crs = bhuvan_gdf.estimate_utm_crs()
    bhuvan_proj = bhuvan_gdf.to_crs(utm_crs)
    sentinel1_proj = sentinel1_gdf.to_crs(utm_crs)

    bhuvan_union = bhuvan_proj.geometry.union_all()
    sentinel1_union = sentinel1_proj.geometry.union_all()
    intersection_area_km2 = bhuvan_union.intersection(sentinel1_union).area / 1e6
    sentinel1_area_km2 = sentinel1_union.area / 1e6

    return {
        "available": True,
        "bhuvan_nrsc_area_km2": round(bhuvan_union.area / 1e6, 4),
        "sentinel1_area_km2": round(sentinel1_area_km2, 4),
        "intersection_area_km2": round(intersection_area_km2, 4),
        "pct_of_sentinel1_area_inside_bhuvan_nrsc_zone": round(100 * intersection_area_km2 / sentinel1_area_km2, 2) if sentinel1_area_km2 else None,
        "note": "Overlap is corroborating, not proof; disagreement is also expected -- see docstring on each method's opposite bias.",
    }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_extraction(bhuvan_gdf: gpd.GeoDataFrame, wards: gpd.GeoDataFrame, depth: np.ndarray) -> dict:
    report = {}

    valid_depth = depth[~np.isnan(depth)]
    report["depth_histogram_m"] = {
        "n_valid_pixels": int(valid_depth.size),
        "pct_above_min_depth": round(100 * float(np.sum(valid_depth > MIN_DEPTH_M)) / valid_depth.size, 2) if valid_depth.size else None,
        "min_depth_m_used": MIN_DEPTH_M,
        "p10": round(float(np.percentile(valid_depth, 10)), 3) if valid_depth.size else None,
        "p50": round(float(np.percentile(valid_depth, 50)), 3) if valid_depth.size else None,
        "p90": round(float(np.percentile(valid_depth, 90)), 3) if valid_depth.size else None,
        "note": "High pct_above_min_depth is expected -- see module docstring on why this raster IS a depth-within-the-flood-zone map.",
    }

    if len(bhuvan_gdf) == 0:
        report["flood_extent"] = {"n_polygons": 0, "total_area_km2": 0.0}
        report["per_ward_flooded_area_km2"] = {}
        return report

    utm_crs = bhuvan_gdf.estimate_utm_crs()
    bhuvan_proj = bhuvan_gdf.to_crs(utm_crs)
    areas_km2 = bhuvan_proj.geometry.area / 1e6
    report["flood_extent"] = {
        "n_polygons": int(len(bhuvan_gdf)),
        "total_area_km2": round(float(areas_km2.sum()), 4),
    }

    wards_proj = wards.to_crs(utm_crs)
    report["per_ward_flooded_area_km2"] = per_ward_area_breakdown(bhuvan_proj, wards_proj)

    return report


# --------------------------------------------------------------------------
# Saving outputs
# --------------------------------------------------------------------------

def save_outputs(bhuvan_gdf: gpd.GeoDataFrame, out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bhuvan_nrsc_flood_extent.geojson"
    bhuvan_gdf.to_file(path, driver="GeoJSON")
    return {"flood_extent_geojson": path}


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    print(f"Loading task 0.6 follow-up NRSC simulation raster <- {NRSC_SIMULATION_PATH}")
    try:
        _require_file(NRSC_SIMULATION_PATH, "src/ground_truth/georeference_nrsc_simulation.py (task 0.6 follow-up)")
        depth, transform, crs = read_single_band(NRSC_SIMULATION_PATH)
    except Phase3ArtifactError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> shape {depth.shape}, {int(np.sum(~np.isnan(depth)))} valid pixels")

    print(f"\nLoading study wards <- {WARDS_PATH}")
    try:
        wards, wards_union = load_wards_union(WARDS_PATH)
    except Phase3ArtifactError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    print(f"\nThresholding depth mask (> {MIN_DEPTH_M}m) ...")
    mask = threshold_depth_mask(depth)
    print(f"  raw candidate pixels: {int(mask.sum())} / {mask.size}")
    mask = sieve_mask(mask, min_pixels=MIN_SIEVE_PIXELS)
    print(f"  after sieve (min {MIN_SIEVE_PIXELS}px): {int(mask.sum())} pixels")

    print("\nVectorizing and clipping to study wards ...")
    bhuvan_gdf = vectorize_mask(mask, transform, crs)
    bhuvan_gdf = clip_to_wards(bhuvan_gdf, wards_union)
    print(f"  -> {len(bhuvan_gdf)} polygon(s)")

    print("\nSaving outputs ...")
    paths = save_outputs(bhuvan_gdf)
    for label, p in paths.items():
        print(f"  {label}: {p}")

    print("\nRunning validation ...")
    report = validate_extraction(bhuvan_gdf, wards, depth)
    report["sentinel1_cross_check"] = cross_check_against_sentinel1(bhuvan_gdf)
    report_path = OUT_DIR / "bhuvan_nrsc_extraction_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  validation report -> {report_path}")

    print("\n=== Validation summary ===")
    print(json.dumps(report, indent=2, default=str))

    print("\nDone. This is SECONDARY/OPPORTUNISTIC ground truth only (task 0.6's verdict,")
    print("unchanged) -- task 3.3/3.4 should weight Sentinel-1 (task 3.1) as primary.")


if __name__ == "__main__":
    main()
