"""
Tasks 1.5 / 1.6 -- download SRTM elevation for the study wards and derive a
slope raster from it. These are two of Objective 1's MUST-HAVE static node
features (working.md SS1.7): every road segment gets an elevation and a slope
value, sampled from these rasters once the line-graph exists (task 2.3).

Usage:
    python src/ground_truth/fetch_dem.py

Outputs (data/raw/dem/, gitignored):
    srtm_dem.tif    -- elevation in meters, EPSG:4326, clipped to the study-ward AOI (+buffer)
    srtm_slope.tif  -- slope in degrees, same grid as the DEM
"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
import geopandas as gpd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
WARDS = REPO_ROOT / "data" / "raw" / "wards" / "study_wards.geojson"
OUT_DIR = REPO_ROOT / "data" / "raw" / "dem"

# SRTM GL1 (30m) tiles are named by their lower-left corner. The study AOI
# (~lon 80.18-80.29, lat 12.95-13.04) straddles the 13N tile boundary, so we
# need both tiles covering it -- fetching only N12E080 would clip the
# northernmost wards (142, 170, 171).
SRTM_TILES = ["N12E080", "N13E080"]
SRTM_BASE_URL = "https://opentopography.s3.sdsc.edu/raster/SRTM_GL1/SRTM_GL1_srtm"
AOI_BUFFER_DEG = 0.01  # small margin so edge segments still get real (not nodata) values


def download_tiles():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for tile in SRTM_TILES:
        out_path = OUT_DIR / f"{tile}.tif"
        if not out_path.exists():
            url = f"{SRTM_BASE_URL}/{tile}.tif"
            print(f"Downloading {url} ...")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            print(f"  saved -> {out_path} ({len(r.content)/1e6:.1f} MB)")
        else:
            print(f"  {out_path.name} already downloaded, skipping")
        paths.append(out_path)
    return paths


def mosaic_and_clip(tile_paths):
    print("Mosaicking tiles and clipping to study-ward AOI...")
    srcs = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)

    wards = gpd.read_file(WARDS)
    minx, miny, maxx, maxy = wards.total_bounds
    minx, miny, maxx, maxy = minx - AOI_BUFFER_DEG, miny - AOI_BUFFER_DEG, maxx + AOI_BUFFER_DEG, maxy + AOI_BUFFER_DEG
    aoi_geom = [{"type": "Polygon", "coordinates": [[
        (minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)
    ]]}]

    mosaic_path = OUT_DIR / "_mosaic_tmp.tif"
    with rasterio.open(mosaic_path, "w", **meta) as dst:
        dst.write(mosaic)
    with rasterio.open(mosaic_path) as src:
        clipped, clipped_transform = mask(src, aoi_geom, crop=True)
        clipped_meta = src.meta.copy()
    clipped_meta.update(height=clipped.shape[1], width=clipped.shape[2], transform=clipped_transform, nodata=-32768)

    dem_path = OUT_DIR / "srtm_dem.tif"
    with rasterio.open(dem_path, "w", **clipped_meta) as dst:
        dst.write(clipped)
    mosaic_path.unlink()
    for s in srcs:
        s.close()

    elev = clipped[0].astype(float)
    valid = elev[elev > -1000]
    print(f"  saved -> {dem_path}  shape={clipped.shape[1:]}  "
          f"elevation range: {valid.min():.1f}m - {valid.max():.1f}m (mean {valid.mean():.1f}m)")
    return dem_path, clipped_meta


def compute_slope(dem_path, meta):
    """Slope in degrees via numpy gradient -- avoids depending on the `gdaldem`
    CLI being on PATH (README flags GDAL as finicky on Windows), while giving
    the same result gdaldem would for a simple central-difference slope."""
    print("Computing slope from DEM (numpy gradient, degrees)...")
    with rasterio.open(dem_path) as src:
        elev = src.read(1).astype(float)
        transform = src.transform
        nodata = src.nodata

    elev_masked = np.where(elev == nodata, np.nan, elev)

    # pixel size in meters -- SRTM is in geographic coords (degrees), so
    # convert using local meters-per-degree at the AOI's latitude
    px_deg_x, px_deg_y = transform.a, -transform.e
    lat_mid = (13.0417 + 12.9529) / 2  # study-ward AOI center latitude
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * np.cos(np.radians(lat_mid))
    px_size_x_m = px_deg_x * m_per_deg_lon
    px_size_y_m = px_deg_y * m_per_deg_lat

    dzdy, dzdx = np.gradient(elev_masked, px_size_y_m, px_size_x_m)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad)
    slope_deg = np.where(np.isnan(elev_masked), -9999, slope_deg).astype("float32")

    slope_path = OUT_DIR / "srtm_slope.tif"
    slope_meta = meta.copy()
    slope_meta.update(dtype="float32", nodata=-9999, count=1)
    with rasterio.open(slope_path, "w", **slope_meta) as dst:
        dst.write(slope_deg, 1)

    valid = slope_deg[slope_deg > -9000]
    print(f"  saved -> {slope_path}  slope range: {valid.min():.2f} deg - {valid.max():.2f} deg (mean {valid.mean():.2f} deg)")
    return slope_path


def main():
    tile_paths = download_tiles()
    dem_path, meta = mosaic_and_clip(tile_paths)
    compute_slope(dem_path, meta)


if __name__ == "__main__":
    main()
