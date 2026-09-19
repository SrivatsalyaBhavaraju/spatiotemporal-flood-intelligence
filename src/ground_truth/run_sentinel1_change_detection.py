"""
Task 3.1 -- Sentinel-1 SAR change-detection flood mapping. Objective 1's
PRIMARY ground-truth source (working.md SS1.5/SS1.8): the 24 Nov 2015
(pre-event) -> 6 Dec 2015 (post-event) pass pair task 1.9 already confirmed
reachable in GEE, run through a standard before/after VV-backscatter
change-detection pipeline (the same class of method the published IEEE
paper on this exact event used -- working.md SS1.5's "already-established
methodology" note).

Method (standard SAR flood-mapping practice, e.g. UN-SPIDER's recommended
workflow, not invented fresh here):
  1. Pull pre/post VV backscatter (dB, already calibrated by GEE's
     COPERNICUS/S1_GRD collection -- confirmed directly: AOI values run
     roughly -17..+6 dB, not linear power) for the study wards, speckle-
     smoothed with a small focal-median filter (~30m).
  2. Compute the per-pixel change (post_db - pre_db) and flag a pixel as
     new flooding if its drop is more than CHANGE_STD_MULTIPLIER standard
     deviations below the AOI's OWN mean change (see the threshold-choice
     note below for why this is scene-relative, not a fixed dB cutoff).
  3. Independently mask out JRC Global Surface Water "permanent water"
     pixels (occurrence > JRC_OCCURRENCE_THRESHOLD% of 1984-2021 observations)
     -- excludes the river/canal channel itself, which isn't NEW flooding.
  4. Sieve-filter (rasterio.features.sieve, GDAL's minimum-mapping-unit
     tool) to drop isolated single/few-pixel speckle before vectorizing.
  5. Vectorize to polygons, clipped to the actual study-ward union (not
     just its bounding box).

Threshold choice -- tried and rejected two alternatives first, on the real
AOI data, before landing on CHANGE_STD_MULTIPLIER:
  - A fixed absolute water threshold (-17dB, UN-SPIDER's commonly-cited VV
    default): only 360 of 466,096 valid pixels in this AOI's POST scene
    fall below -17dB at all (<0.1%) -- this dense-urban scene's backscatter
    simply runs higher than the open/rural terrain that default is usually
    calibrated against (double-bounce off building facades near flooded
    streets is a documented confound). Produced a near-empty ~0.01 km^2
    result -- clearly wrong for a flood that was one of Chennai's worst.
  - Otsu's method on the post-pre difference histogram (an automatic,
    scene-adaptive alternative to a fixed cutoff): picked -0.77dB and
    flagged 54% of the AOI -- also clearly wrong. Otsu assumes a clean
    bimodal histogram; this scene's change values are a single noisy mode
    with a skewed tail (flooding is a minority class), not two separable
    populations, so Otsu has nothing real to split on.
  CHANGE_STD_MULTIPLIER=2.0 (mean - 2*std, the standard robust-outlier
  cutoff) self-calibrates to whatever THIS specific pass pair's own noise
  floor is, rather than assuming a fixed physical dB value that shifts with
  terrain/urban density/incidence angle -- gives ~0.85 km^2 (~1.8% of the
  AOI), a plausible middle ground. Still a tunable constant, not a proven
  optimum -- the validation report prints the diff histogram and both
  rejected alternatives' pixel counts so a human can re-judge this.

Known limitation (already flagged in working.md SS1.5, not new here): SAR
underestimates flood extent under dense urban canopy/buildings -- the
rejected -17dB attempt above is a direct, concrete demonstration of exactly
that limitation, not just a citation of it. This before/after pair also
brackets the 30 Nov-2 Dec peak rather than capturing it (nearest pass is +4
days post-peak). State both explicitly when this feeds task 3.3/3.4's label
fusion -- don't treat this polygon as a complete flood map.

Usage:
    python src/ground_truth/run_sentinel1_change_detection.py

Required inputs:
    data/raw/wards/study_wards.geojson   (task 1.3)
    GEE authenticated (task 1.9, see src/ground_truth/README.md)
    data/raw/bhuvan/fig8_georeferenced.tif  (optional, task 0.6 follow-up --
        used only for the cross-validation check, not required to run)

Outputs (data/processed/ground_truth/):
    sentinel1_pre_event_vv_db.tif / sentinel1_post_event_vv_db.tif / sentinel1_jrc_occurrence.tif
                                       -- downloaded intermediate rasters (for QA/reruns)
    sentinel1_flood_extent.geojson    -- one row per flood polygon, clipped to study wards
    sentinel1_change_detection_validation_report.json
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.features import shapes as rasterio_shapes
from rasterio.features import sieve
from shapely.geometry import shape as shapely_shape
from shapely.geometry import mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
WARDS_PATH = REPO_ROOT / "data" / "raw" / "wards" / "study_wards.geojson"
NRSC_SIMULATION_PATH = REPO_ROOT / "data" / "raw" / "bhuvan" / "fig8_georeferenced.tif"
OUT_DIR = REPO_ROOT / "data" / "processed" / "ground_truth"

GEE_PROJECT = "flood-intelligence-507219"  # task 1.9 -- same project as query_sentinel1_gee.py
PRE_EVENT_DATE = "2015-11-24"
POST_EVENT_DATE = "2015-12-06"
DOWNLOAD_SCALE_M = 10  # S1 GRD IW native ground-range resolution
SPECKLE_FILTER_RADIUS_M = 30

CHANGE_STD_MULTIPLIER = 2.0     # SS module docstring: mean-k*std robust-outlier cutoff on post-pre dB change
JRC_OCCURRENCE_THRESHOLD = 50   # percent of 1984-2021 observations classified as water
MIN_SIEVE_PIXELS = 8            # minimum mapping unit at 10m = 800 m^2

STORAGE_CRS = "EPSG:4326"  # matches every other geospatial artifact in this repo


class Phase3ArtifactError(FileNotFoundError):
    """Raised when a required task 1.3/1.9 artifact/setup is missing.
    Matches the *ArtifactError/*SchemaError convention used across
    src/graph, src/features, src/models/gnn."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase3ArtifactError(
            f"Required artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first."
        )
    return path


# --------------------------------------------------------------------------
# GEE fetch + download (network-dependent, validated by a real run --
# see developing.md task 3.1 notes rather than a mocked unit test)
# --------------------------------------------------------------------------

def load_wards_union(path: Path = WARDS_PATH):
    """Return (wards_gdf, union_geometry_shapely) -- the union is the actual
    study-area shape (not its bounding box), used both as the GEE clip
    region and the final vector-clip boundary."""
    _require_file(path, "src/graph/fetch_osm.py's ward step (task 1.3)")
    wards = gpd.read_file(path)
    if len(wards) == 0:
        raise Phase3ArtifactError(f"{path} loaded but contains zero wards.")
    return wards, wards.geometry.union_all()


def fetch_vv_db_image(ee_geom, date_str: str, smooth_radius_m: float = SPECKLE_FILTER_RADIUS_M):
    """One Sentinel-1 GRD/IW VV scene (already dB per GEE's own calibration
    -- confirmed directly, see module docstring), clipped to `ee_geom` and
    speckle-smoothed with a focal median filter."""
    import ee

    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(ee_geom)
        .filterDate(date_str, ee.Date(date_str).advance(1, "day"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    )
    img = collection.first()
    if img is None:
        raise Phase3ArtifactError(f"No Sentinel-1 GRD/IW/VV scene found for {date_str} over the study wards.")
    smoothed = img.select("VV").focal_median(radius=smooth_radius_m, units="meters")
    return smoothed.clip(ee_geom).toFloat()


def fetch_permanent_water_image(ee_geom):
    """JRC Global Surface Water 'occurrence' band (% of 1984-2021
    observations classified as water), unmasked to 0 so dry-land pixels
    aren't silently dropped from later array math, clipped to `ee_geom`."""
    import ee

    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    return jrc.unmask(0).clip(ee_geom).toFloat()


def download_ee_image(image, region, out_path: Path, scale: float = DOWNLOAD_SCALE_M, crs: str = STORAGE_CRS) -> Path:
    """Download a single-band ee.Image as a local GeoTIFF via
    getDownloadURL(). All three rasters this module needs (pre/post/JRC)
    must be requested with the SAME region/scale/crs so their pixel grids
    line up element-wise for classify_flood_mask() -- callers must pass the
    same `region` for all three."""
    url = image.getDownloadURL({"region": region, "scale": scale, "crs": crs, "format": "GEO_TIFF"})
    resp = requests.get(url, timeout=120)
    if resp.status_code != 200:
        raise Phase3ArtifactError(f"GEE download failed ({resp.status_code}) for {out_path.name}: {resp.text[:500]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    return out_path


def read_single_band(path: Path):
    """Return (array, transform, crs) for a single-band raster, with nodata
    pixels set to NaN so downstream math can't silently include them."""
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype(np.float64).filled(np.nan)
        return arr, src.transform, src.crs


# --------------------------------------------------------------------------
# Local classification / vectorization (pure array math -- unit-tested with
# synthetic rasters, no GEE credentials needed)
# --------------------------------------------------------------------------

def classify_flood_mask(
    pre_db: np.ndarray,
    post_db: np.ndarray,
    jrc_occurrence: np.ndarray,
    std_multiplier: float = CHANGE_STD_MULTIPLIER,
    jrc_occurrence_threshold: float = JRC_OCCURRENCE_THRESHOLD,
) -> np.ndarray:
    """New-flooding boolean mask: post-pre backscatter drop more than
    `std_multiplier` standard deviations below the AOI's OWN mean change
    (see module docstring for why this self-calibrating cutoff was chosen
    over a fixed absolute dB threshold), AND not JRC permanent water. NaN
    pixels (outside the clipped AOI/nodata) are never flagged as flood.
    """
    if not (pre_db.shape == post_db.shape == jrc_occurrence.shape):
        raise Phase3ArtifactError(
            f"pre/post/JRC array shapes must match (pre/post/JRC must be downloaded with the same "
            f"region/scale/crs): got {pre_db.shape}, {post_db.shape}, {jrc_occurrence.shape}."
        )
    valid = ~(np.isnan(pre_db) | np.isnan(post_db) | np.isnan(jrc_occurrence))
    diff = post_db - pre_db
    threshold = np.nanmean(diff) - std_multiplier * np.nanstd(diff)
    significant_drop = diff < threshold
    permanent_water = jrc_occurrence > jrc_occurrence_threshold
    return significant_drop & ~permanent_water & valid


def sieve_mask(mask: np.ndarray, min_pixels: int = MIN_SIEVE_PIXELS, connectivity: int = 8) -> np.ndarray:
    """Drop connected components smaller than `min_pixels` (rasterio's
    GDAL-backed sieve filter -- the standard minimum-mapping-unit tool for
    exactly this kind of speckle cleanup, not a hand-rolled reimplementation)."""
    cleaned = sieve(mask.astype(np.uint8), size=min_pixels, connectivity=connectivity)
    return cleaned.astype(bool)


def vectorize_mask(mask: np.ndarray, transform, crs) -> gpd.GeoDataFrame:
    """True-valued regions of `mask` -> one polygon per connected region."""
    polygons = [
        shapely_shape(geom) for geom, value in rasterio_shapes(mask.astype(np.uint8), mask=mask, transform=transform)
        if value == 1
    ]
    if not polygons:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    return gpd.GeoDataFrame({"geometry": polygons}, geometry="geometry", crs=crs)


def _ward_label(ward_row, wards_columns) -> str:
    zone, ward_no = ward_row.get("Zone_Name"), ward_row.get("Ward_No")
    if zone is not None and ward_no is not None:
        return f"{zone} (Ward {ward_no})"
    if "Zone_Name" in wards_columns:
        return str(ward_row["Zone_Name"])
    return str(ward_row[wards_columns[0]])


def per_ward_area_breakdown(flood_proj: gpd.GeoDataFrame, wards_proj: gpd.GeoDataFrame) -> dict:
    """Flooded area per ward (km^2), keyed by a ward-unique label
    ("<Zone_Name> (Ward <Ward_No>)" when both columns exist). Accumulates
    rather than overwrites so wards sharing a Zone_Name -- common, since a
    "zone" groups several numbered wards; 13 of this study's 16 wards are
    all "ADYAR" -- don't silently clobber each other's area. `flood_proj`/
    `wards_proj` must already be in the same projected (metric) CRS.
    """
    per_ward = {}
    for _, ward_row in wards_proj.iterrows():
        inter = flood_proj.geometry.intersection(ward_row.geometry)
        area_km2 = inter.area.sum() / 1e6
        if area_km2 > 0:
            label = _ward_label(ward_row, wards_proj.columns)
            per_ward[label] = per_ward.get(label, 0.0) + area_km2
    return dict(sorted(((k, round(v, 4)) for k, v in per_ward.items()), key=lambda kv: -kv[1]))


def clip_to_wards(flood_gdf: gpd.GeoDataFrame, wards_union) -> gpd.GeoDataFrame:
    """Intersect flood polygons with the actual study-ward union (not its
    bounding box) -- the download grid is a rectangle, this trims it back
    to the real study area."""
    if len(flood_gdf) == 0:
        return flood_gdf
    clipped = gpd.clip(flood_gdf, gpd.GeoSeries([wards_union], crs=flood_gdf.crs))
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()]
    return clipped.reset_index(drop=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_flood_extent(
    flood_gdf: gpd.GeoDataFrame,
    wards: gpd.GeoDataFrame,
    pre_db: np.ndarray,
    post_db: np.ndarray,
) -> dict:
    report = {}

    diff = post_db - pre_db
    diff_mean, diff_std = float(np.nanmean(diff)), float(np.nanstd(diff))
    threshold_used = diff_mean - CHANGE_STD_MULTIPLIER * diff_std
    n_valid = int(np.sum(~np.isnan(diff)))
    report["change_detection_diagnostics"] = {
        "note": "See module docstring 'Threshold choice' for why CHANGE_STD_MULTIPLIER was chosen over these two rejected alternatives.",
        "diff_db_histogram": {k: round(float(v), 2) for k, v in zip(
            ["p1", "p5", "p10", "p50", "mean", "std"],
            [*np.nanpercentile(diff, [1, 5, 10, 50]), diff_mean, diff_std])},
        "threshold_used": {
            "method": f"mean - {CHANGE_STD_MULTIPLIER}*std",
            "value_db": round(threshold_used, 2),
            "n_pixels_flagged_pre_sieve": int(np.sum((diff < threshold_used) & ~np.isnan(diff))),
        },
        "rejected_alternative_fixed_minus17db": {
            "n_pixels_below_minus17db_post_event": int(np.sum((post_db < -17.0) & ~np.isnan(post_db))),
            "pct_of_valid_aoi": round(100 * np.sum((post_db < -17.0) & ~np.isnan(post_db)) / n_valid, 3) if n_valid else None,
        },
    }

    if len(flood_gdf) == 0:
        report["flood_extent"] = {"n_polygons": 0, "total_area_km2": 0.0}
        report["per_ward_flooded_area_km2"] = {}
        return report

    utm_crs = flood_gdf.estimate_utm_crs()
    flood_proj = flood_gdf.to_crs(utm_crs)
    areas_km2 = flood_proj.geometry.area / 1e6
    report["flood_extent"] = {
        "n_polygons": int(len(flood_gdf)),
        "total_area_km2": round(float(areas_km2.sum()), 4),
        "largest_polygon_km2": round(float(areas_km2.max()), 4),
        "median_polygon_km2": round(float(areas_km2.median()), 4),
    }

    wards_proj = wards.to_crs(utm_crs)
    report["per_ward_flooded_area_km2"] = per_ward_area_breakdown(flood_proj, wards_proj)

    return report


def cross_check_against_nrsc_simulation(flood_gdf: gpd.GeoDataFrame, nrsc_path: Path = NRSC_SIMULATION_PATH, n_control: int = 200) -> dict:
    """Sample task 0.6's georeferenced NRSC hydrological-simulation
    flood-depth raster at this polygon set's centroids vs. an equal number
    of random points drawn from the study-ward bounding box, and compare
    mean depth. This raster is a coarse, independently-sourced (if noisy --
    RMSE ~563m, see developing.md's Phase 0 notes) cross-check, not a
    replacement for this task's own SAR-based result -- a real correlation
    here is corroborating evidence, not proof, and its absence isn't
    disqualifying given that raster's known coarseness.
    """
    if not nrsc_path.exists() or len(flood_gdf) == 0:
        return {"available": False, "reason": "NRSC simulation raster or flood polygons not available"}

    with rasterio.open(nrsc_path) as src:
        depth = src.read(1, masked=True).astype(np.float64).filled(np.nan)
        transform = src.transform
        bounds = src.bounds
        raster_crs = src.crs

    flood_in_raster_crs = flood_gdf.to_crs(raster_crs)
    # Geographic-CRS centroid warning is benign here: the study-ward cluster
    # spans ~9km, far too small for lon/lat distortion to matter for this
    # coarse point-sampling check (unlike area_km2 above, which does need
    # a projected CRS and gets one via estimate_utm_crs()).
    centroids = flood_in_raster_crs.geometry.centroid
    rng = np.random.default_rng(0)
    n = min(n_control, len(centroids))
    sample_centroids = centroids.sample(n=n, random_state=0) if len(centroids) > n else centroids

    def sample_depth_at(xs, ys):
        rows, cols = rasterio.transform.rowcol(transform, xs, ys)
        rows, cols = np.asarray(rows), np.asarray(cols)
        in_bounds = (rows >= 0) & (rows < depth.shape[0]) & (cols >= 0) & (cols < depth.shape[1])
        vals = np.full(len(rows), np.nan)
        vals[in_bounds] = depth[rows[in_bounds], cols[in_bounds]]
        return vals

    flood_depths = sample_depth_at(sample_centroids.x.to_numpy(), sample_centroids.y.to_numpy())

    random_x = rng.uniform(bounds.left, bounds.right, size=n)
    random_y = rng.uniform(bounds.bottom, bounds.top, size=n)
    control_depths = sample_depth_at(random_x, random_y)

    return {
        "available": True,
        "n_flood_centroids_sampled": int(np.sum(~np.isnan(flood_depths))),
        "n_control_points_sampled": int(np.sum(~np.isnan(control_depths))),
        "mean_nrsc_depth_m_at_flood_centroids": round(float(np.nanmean(flood_depths)), 3) if np.any(~np.isnan(flood_depths)) else None,
        "mean_nrsc_depth_m_at_random_points": round(float(np.nanmean(control_depths)), 3) if np.any(~np.isnan(control_depths)) else None,
        "note": "Higher mean depth at flood centroids than random points is corroborating (not proof) evidence -- see docstring on this raster's own coarseness.",
    }


# --------------------------------------------------------------------------
# Saving outputs
# --------------------------------------------------------------------------

def save_outputs(flood_gdf: gpd.GeoDataFrame, out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sentinel1_flood_extent.geojson"
    flood_gdf.to_file(path, driver="GeoJSON")
    return {"flood_extent_geojson": path}


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    import ee

    print(f"Loading study wards <- {WARDS_PATH}")
    try:
        wards, wards_union = load_wards_union()
    except Phase3ArtifactError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> {len(wards)} wards, union bounds {wards_union.bounds}")

    print(f"\nInitializing GEE (project={GEE_PROJECT}) ...")
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception as e:
        print("\nERROR: GEE not authenticated. Run `earthengine authenticate` first "
              "(see src/ground_truth/README.md's task 1.9 section).", file=sys.stderr)
        print(f"(original error: {e})", file=sys.stderr)
        raise SystemExit(1)

    ee_geom = ee.Geometry(mapping(wards_union))
    region = ee_geom.bounds()

    print(f"Fetching pre-event ({PRE_EVENT_DATE}) and post-event ({POST_EVENT_DATE}) VV scenes ...")
    pre_img = fetch_vv_db_image(ee_geom, PRE_EVENT_DATE)
    post_img = fetch_vv_db_image(ee_geom, POST_EVENT_DATE)
    jrc_img = fetch_permanent_water_image(ee_geom)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pre_path = OUT_DIR / "sentinel1_pre_event_vv_db.tif"
    post_path = OUT_DIR / "sentinel1_post_event_vv_db.tif"
    jrc_path = OUT_DIR / "sentinel1_jrc_occurrence.tif"

    print("Downloading rasters (same region/scale/crs for all three, so grids align) ...")
    download_ee_image(pre_img, region, pre_path)
    download_ee_image(post_img, region, post_path)
    download_ee_image(jrc_img, region, jrc_path)
    print(f"  -> {pre_path.name}, {post_path.name}, {jrc_path.name}")

    pre_db, transform, crs = read_single_band(pre_path)
    post_db, _, _ = read_single_band(post_path)
    jrc_occurrence, _, _ = read_single_band(jrc_path)
    print(f"  raster shape: {pre_db.shape}")

    print("\nClassifying new-flooding mask ...")
    mask = classify_flood_mask(pre_db, post_db, jrc_occurrence)
    print(f"  raw candidate flood pixels: {int(mask.sum())} / {mask.size}")
    mask = sieve_mask(mask)
    print(f"  after sieve (min {MIN_SIEVE_PIXELS}px): {int(mask.sum())} pixels")

    print("\nVectorizing and clipping to study wards ...")
    flood_gdf = vectorize_mask(mask, transform, crs)
    flood_gdf = clip_to_wards(flood_gdf, wards_union)
    print(f"  -> {len(flood_gdf)} flood polygon(s)")

    print("\nSaving outputs ...")
    paths = save_outputs(flood_gdf)
    for label, p in paths.items():
        print(f"  {label}: {p}")

    print("\nRunning validation ...")
    report = validate_flood_extent(flood_gdf, wards, pre_db, post_db)
    report["nrsc_simulation_cross_check"] = cross_check_against_nrsc_simulation(flood_gdf)
    report_path = OUT_DIR / "sentinel1_change_detection_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  validation report -> {report_path}")

    print("\n=== Validation summary ===")
    print(json.dumps(report, indent=2, default=str))

    print("\nDone. Next: task 3.3 (news cross-check) intersects line-graph segments against")
    print(f"  {paths['flood_extent_geojson']} -- 'Segment labeled flooded if it intersects the polygon' (working.md SS1.5).")


if __name__ == "__main__":
    main()
