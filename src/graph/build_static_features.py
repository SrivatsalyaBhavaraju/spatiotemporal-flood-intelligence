"""
Task 2.3 -- compute the four MUST-HAVE static node features (working.md
SS1.7) for every line-graph node (= road segment, task 2.2): elevation,
slope, length, distance_to_drain. Output is a lookup table keyed by
`segment_id`, ready for task 2.6 (dynamic features) / 2.7 (model input
schema) to join onto the line graph.

Feature definitions:
  - length_m         -- already computed by task 2.1/2.2 (segment geometry
                         length in a projected/metric CRS). NOT
                         recomputed here -- just carried through from
                         line_graph_nodes.geojson, single source of truth.
  - elevation_m       -- mean SRTM elevation (task 1.5) sampled along the
                         segment's geometry (not just one point -- see
                         sample_rasters_for_segments()).
  - slope_deg         -- mean slope-in-degrees (task 1.6), sampled the same way.
  - distance_to_drain_m -- distance in a projected/metric CRS from the
                         segment's full geometry to the NEAREST drainage
                         feature (task 2.1's drainage.geojson / task 1.2's
                         waterways.geojson) -- minimum line-to-geometry
                         distance, not a point-to-point approximation.

Usage:
    python src/graph/build_static_features.py

Required inputs:
    data/processed/graph/line_graph_nodes.geojson  -- task 2.2
    data/raw/dem/srtm_dem.tif                       -- task 1.5 (elevation, meters, EPSG:4326)
    data/raw/dem/srtm_slope.tif                     -- task 1.6 (slope, degrees, EPSG:4326)

Drainage input (used for distance_to_drain; NOT a hard requirement -- an
empty/missing drainage layer produces NaN distances with a documented
warning, matching how tasks 1.2/1.4/2.1 already treat sparse OSM drainage
tagging as an expected data-quality fact, not a pipeline error):
    data/processed/graph/drainage.geojson  -- preferred (task 2.1's pass-through copy)
    data/raw/osm/waterways.geojson         -- fallback if the above is missing (task 1.2 direct)

Outputs (data/processed/graph/):
    static_features.geojson             -- one row per segment_id: length_m, elevation_m,
                                            slope_deg, distance_to_drain_m, sample-quality
                                            columns, geometry (carried through for reference)
    static_features_validation_report.json -- validation report (see validate_static_features())

Handoff for 2.4/2.6/2.7: join this table onto the line graph / other feature
tables by `segment_id` -- a plain key match, nothing positional.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_GRAPH_DIR = REPO_ROOT / "data" / "processed" / "graph"
LINE_GRAPH_NODES = PROCESSED_GRAPH_DIR / "line_graph_nodes.geojson"
DEM_PATH = REPO_ROOT / "data" / "raw" / "dem" / "srtm_dem.tif"
SLOPE_PATH = REPO_ROOT / "data" / "raw" / "dem" / "srtm_slope.tif"
DRAINAGE_PRIMARY = PROCESSED_GRAPH_DIR / "drainage.geojson"
DRAINAGE_FALLBACK = REPO_ROOT / "data" / "raw" / "osm" / "waterways.geojson"

STORAGE_CRS = "EPSG:4326"

# Sample points along each segment's geometry every SAMPLE_SPACING_M meters
# (roughly half the DEM's 30m SRTM pixel size, so a segment reliably crosses
# into a fresh pixel between samples rather than resampling the same one),
# with at least 2 samples (both endpoints) and a cap so a handful of
# unusually long segments (real data: up to ~1.15km, task 2.1's validation)
# don't blow up runtime.
SAMPLE_SPACING_M = 15.0
MAX_SAMPLES_PER_SEGMENT = 50


class Phase2FeatureError(FileNotFoundError):
    """Raised when a required task 2.2/1.5/1.6 artifact is missing or
    malformed. Always names the exact expected path and which upstream
    script produces it."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase2FeatureError(
            f"Required artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first, or "
            f"place the file there if it was generated elsewhere."
        )
    return path


# --------------------------------------------------------------------------
# Loading inputs
# --------------------------------------------------------------------------

def load_line_graph_nodes(path: Path = LINE_GRAPH_NODES) -> gpd.GeoDataFrame:
    _require_file(path, "src/graph/build_line_graph.py (task 2.2)")
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        raise Phase2FeatureError(f"Failed to parse {path} as a GeoJSON/vector file: {e}") from e

    required_cols = {"segment_id", "length_m", "geometry"}
    missing = required_cols - set(gdf.columns)
    if missing:
        raise Phase2FeatureError(
            f"{path} is missing required column(s) {sorted(missing)}. Expected the schema "
            f"task 2.2's build_line_graph.py produces -- re-run it, or check for a stale file."
        )
    if gdf.crs is None:
        raise Phase2FeatureError(f"{path} has no CRS defined.")
    if len(gdf) == 0:
        raise Phase2FeatureError(f"{path} loaded but contains zero rows.")
    return gdf


def _check_raster(path: Path, produced_by: str, expected_crs: str = STORAGE_CRS) -> None:
    _require_file(path, produced_by)
    try:
        with rasterio.open(path) as src:
            if src.count < 1:
                raise Phase2FeatureError(f"{path} has no raster bands.")
            if src.crs is None:
                raise Phase2FeatureError(f"{path} has no CRS defined.")
    except rasterio.errors.RasterioIOError as e:
        raise Phase2FeatureError(f"Failed to open {path} as a raster: {e}") from e


def load_drainage(primary: Path = DRAINAGE_PRIMARY, fallback: Path = DRAINAGE_FALLBACK):
    """Prefer task 2.1's drainage.geojson (documented pass-through of the
    waterway layer); fall back to task 1.2's waterways.geojson directly if
    2.1 hasn't been run in this environment. Missing entirely, or present
    but empty, is a valid (if limiting) state -- not an error -- since OSM
    drainage tagging in this study area is documented as sparse (task 1.4).
    """
    path = primary if primary.exists() else fallback
    if not path.exists():
        print(f"  WARNING: neither {primary} nor {fallback} found -- distance_to_drain will be NaN for every segment.")
        return None, None
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        print(f"  WARNING: failed to parse {path} ({e}) -- distance_to_drain will be NaN for every segment.")
        return None, path
    return gdf, path


# --------------------------------------------------------------------------
# Elevation / slope sampling
# --------------------------------------------------------------------------

def _segment_sample_points(geom, length_m: float, spacing_m: float, max_samples: int):
    n = max(2, min(max_samples, int(length_m // spacing_m) + 2))
    fracs = np.linspace(0.0, 1.0, n)
    return [geom.interpolate(f, normalized=True) for f in fracs]


def sample_rasters_for_segments(
    segments_gdf: gpd.GeoDataFrame,
    dem_path: Path = DEM_PATH,
    slope_path: Path = SLOPE_PATH,
    spacing_m: float = SAMPLE_SPACING_M,
    max_samples: int = MAX_SAMPLES_PER_SEGMENT,
):
    """Sample the DEM and slope rasters at several points along each
    segment's geometry (not just the midpoint) and return the per-segment
    mean, ignoring nodata pixels. One batched rasterio `.sample()` call
    covers every point across every segment for performance (17k+ segments
    x ~5-50 points each).

    Returns (elevation_m, slope_deg, n_valid_elevation_samples,
    n_valid_slope_samples, n_total_samples) -- all length == len(segments_gdf).
    """
    all_points = []
    counts = []
    for geom, length_m in zip(segments_gdf.geometry, segments_gdf["length_m"]):
        pts = _segment_sample_points(geom, float(length_m), spacing_m, max_samples)
        counts.append(len(pts))
        all_points.extend((p.x, p.y) for p in pts)

    with rasterio.open(dem_path) as dem_src, rasterio.open(slope_path) as slope_src:
        dem_nodata = dem_src.nodata
        slope_nodata = slope_src.nodata
        dem_vals = np.array([v[0] for v in dem_src.sample(all_points)], dtype=float)
        slope_vals = np.array([v[0] for v in slope_src.sample(all_points)], dtype=float)

    if dem_nodata is not None:
        dem_vals[dem_vals == dem_nodata] = np.nan
    if slope_nodata is not None:
        slope_vals[slope_vals == slope_nodata] = np.nan

    elevations, slopes, n_elev_valid, n_slope_valid = [], [], [], []
    idx = 0
    for n in counts:
        chunk_elev = dem_vals[idx: idx + n]
        chunk_slope = slope_vals[idx: idx + n]
        idx += n
        valid_elev = chunk_elev[~np.isnan(chunk_elev)]
        valid_slope = chunk_slope[~np.isnan(chunk_slope)]
        elevations.append(float(valid_elev.mean()) if len(valid_elev) else np.nan)
        slopes.append(float(valid_slope.mean()) if len(valid_slope) else np.nan)
        n_elev_valid.append(int(len(valid_elev)))
        n_slope_valid.append(int(len(valid_slope)))

    return elevations, slopes, n_elev_valid, n_slope_valid, counts


# --------------------------------------------------------------------------
# distance_to_drain
# --------------------------------------------------------------------------

def compute_distance_to_drain(segments_gdf: gpd.GeoDataFrame, drainage_gdf) -> tuple:
    """Distance in meters from each segment's full geometry to the nearest
    drainage feature, computed in an auto-detected local UTM CRS (never in
    degrees) -- same approach task 2.1 uses for length_m. Returns
    (distances_m, utm_crs_used_str). If drainage_gdf is None/empty, returns
    an all-NaN array (not an error -- see load_drainage()).
    """
    if drainage_gdf is None or len(drainage_gdf) == 0:
        return np.full(len(segments_gdf), np.nan), None

    utm_crs = segments_gdf.estimate_utm_crs()
    seg_proj = segments_gdf[["segment_id", "geometry"]].to_crs(utm_crs)
    drain_proj = drainage_gdf[["geometry"]].to_crs(utm_crs)
    drain_proj = drain_proj[drain_proj.geometry.notna() & ~drain_proj.geometry.is_empty]
    if len(drain_proj) == 0:
        return np.full(len(segments_gdf), np.nan), str(utm_crs)

    joined = gpd.sjoin_nearest(seg_proj, drain_proj, how="left", distance_col="distance_to_drain_m")
    # sjoin_nearest can return >1 row per segment on exact ties -- keep one per segment_id
    joined = joined.drop_duplicates(subset="segment_id", keep="first").set_index("segment_id")
    distances = joined.loc[segments_gdf["segment_id"].values, "distance_to_drain_m"].to_numpy()
    return distances, str(utm_crs)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build_static_features(
    line_graph_nodes_path: Path = LINE_GRAPH_NODES,
    dem_path: Path = DEM_PATH,
    slope_path: Path = SLOPE_PATH,
    drainage_primary: Path = DRAINAGE_PRIMARY,
    drainage_fallback: Path = DRAINAGE_FALLBACK,
) -> gpd.GeoDataFrame:
    segments = load_line_graph_nodes(line_graph_nodes_path)
    _check_raster(dem_path, "src/ground_truth/fetch_dem.py (task 1.5)")
    _check_raster(slope_path, "src/ground_truth/fetch_dem.py (task 1.6)")
    drainage_gdf, drainage_path_used = load_drainage(drainage_primary, drainage_fallback)

    elevations, slopes, n_elev_valid, n_slope_valid, n_samples = sample_rasters_for_segments(
        segments, dem_path, slope_path
    )
    distances, utm_crs_used = compute_distance_to_drain(segments, drainage_gdf)

    out = gpd.GeoDataFrame(
        {
            "segment_id": segments["segment_id"].values,
            "length_m": segments["length_m"].values,
            "elevation_m": elevations,
            "slope_deg": slopes,
            "distance_to_drain_m": distances,
            "n_raster_samples": n_samples,
            "n_valid_elevation_samples": n_elev_valid,
            "n_valid_slope_samples": n_slope_valid,
        },
        geometry=segments.geometry.values,
        crs=STORAGE_CRS,
    )
    out = out.set_index("segment_id", drop=False)
    out.attrs["drainage_path_used"] = str(drainage_path_used) if drainage_path_used else None
    out.attrs["utm_crs_used_for_distance_to_drain"] = utm_crs_used
    return out


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_static_features(features_gdf: gpd.GeoDataFrame, sample_size: int = 5) -> dict:
    report = {}
    n = len(features_gdf)
    report["counts"] = {"segments": int(n)}
    report["crs"] = {
        "input_output": str(features_gdf.crs),
        "utm_crs_used_for_distance_to_drain": features_gdf.attrs.get("utm_crs_used_for_distance_to_drain"),
    }
    report["drainage_source_used"] = features_gdf.attrs.get("drainage_path_used")

    for col, unit in [("elevation_m", "m"), ("slope_deg", "deg"), ("distance_to_drain_m", "m"), ("length_m", "m")]:
        series = features_gdf[col]
        missing = series.isna()
        valid = series[~missing]
        report[col] = {
            "missing_count": int(missing.sum()),
            "missing_pct": round(100 * missing.sum() / n, 2) if n else None,
            "missing_segment_ids_sample": features_gdf.loc[missing, "segment_id"].tolist()[:20],
            "min": float(valid.min()) if len(valid) else None,
            "max": float(valid.max()) if len(valid) else None,
            "mean": round(float(valid.mean()), 3) if len(valid) else None,
            "median": round(float(valid.median()), 3) if len(valid) else None,
            "unit": unit,
        }

    low_sample = features_gdf["n_valid_elevation_samples"] == 0
    report["raster_sampling_quality"] = {
        "mean_samples_per_segment": round(float(features_gdf["n_raster_samples"].mean()), 2) if n else None,
        "segments_with_zero_valid_raster_samples": int(low_sample.sum()),
        "segments_with_zero_valid_raster_samples_sample": features_gdf.loc[low_sample, "segment_id"].tolist()[:20],
    }

    report["samples"] = features_gdf.drop(columns="geometry").head(sample_size).to_dict(orient="records")

    return report


# --------------------------------------------------------------------------
# Saving outputs
# --------------------------------------------------------------------------

def save_outputs(features_gdf: gpd.GeoDataFrame, out_dir: Path = PROCESSED_GRAPH_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "static_features.geojson"
    features_gdf.reset_index(drop=True).to_file(path, driver="GeoJSON")
    return {"static_features_geojson": path}


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    print(f"Loading task 2.2 line-graph nodes <- {LINE_GRAPH_NODES}")
    try:
        features = build_static_features()
    except Phase2FeatureError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> {len(features)} segments")
    print(f"  DEM <- {DEM_PATH}")
    print(f"  Slope <- {SLOPE_PATH}")
    print(f"  Drainage <- {features.attrs.get('drainage_path_used')}")

    print("\nSaving outputs ...")
    paths = save_outputs(features)
    for label, p in paths.items():
        print(f"  {label}: {p}")

    print("\nRunning validation ...")
    report = validate_static_features(features)
    report_path = PROCESSED_GRAPH_DIR / "static_features_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  validation report -> {report_path}")

    print("\n=== Validation summary ===")
    print(json.dumps(report, indent=2, default=str))

    print("\nDone. Next: task 2.4 (optional impervious %/ward density) or task 2.6")
    print(f"  (dynamic rainfall features) join onto {paths['static_features_geojson']} by `segment_id`.")


if __name__ == "__main__":
    main()
