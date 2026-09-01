"""
Task 1.8 -- daily IMD gridded rainfall, as an independent cross-check against
the Open-Meteo/ERA5-Land hourly series from task 1.7.

Why this cross-check matters more than a routine QA step: the Open-Meteo
series came back showing 23 Nov and 1 Dec 2015 (the two dates every other
source -- ReliefWeb, World Weather Attribution, the NRSC hydrological report
-- agrees were the rainfall peaks) at only 15mm and 49mm, while flagging
16 Nov (115mm) as its actual maximum. That's a 5-10x mismatch against every
independently-sourced number gathered so far. This script checks whether
IMD's independently-gridded product agrees with Open-Meteo or with the
literature.

Source: IMD gridded daily rainfall, 0.25 deg (~28km), via `imdlib` (pulls the
official IMD binary grids directly, no auth).

Usage:
    python src/ground_truth/fetch_rainfall_imd.py

Output (data/raw/rainfall/, gitignored):
    imd_daily.csv  -- columns: date, precipitation_mm (AOI-centroid nearest grid cell)
"""
from pathlib import Path

import geopandas as gpd
import imdlib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
WARDS = REPO_ROOT / "data" / "raw" / "wards" / "study_wards.geojson"
OUT_DIR = REPO_ROOT / "data" / "raw" / "rainfall"
IMD_CACHE_DIR = OUT_DIR / "imd_raw"

YEAR = 2015  # event falls entirely within this year


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    wards = gpd.read_file(WARDS)
    centroid = wards.union_all().centroid
    lat, lon = centroid.y, centroid.x
    print(f"Study-ward AOI centroid: ({lat:.4f}, {lon:.4f})")

    print(f"Downloading IMD gridded daily rainfall for {YEAR} (whole-India grid, ~28km cells)...")
    data = imdlib.get_data("rain", YEAR, YEAR, fn_format="yearwise", file_dir=str(IMD_CACHE_DIR))

    ds = data.get_xarray()
    print(f"  grid loaded: {dict(ds.sizes)}")

    # IMD's grid masks coastal/ocean cells with -999 for the whole year -- and
    # Chennai's nearest grid cell (13.0, 80.25) IS one of those masked cells
    # (confirmed: 0/365 valid days). .sel(method="nearest") would silently
    # hand back an all-missing series, so find the nearest cell that actually
    # has data instead of trusting proximity alone.
    valid_mask = (ds["rain"] > -900).any(dim="time")
    valid_lats, valid_lons = np.meshgrid(ds["lat"].values, ds["lon"].values, indexing="ij")
    dist = np.sqrt((valid_lats - lat) ** 2 + (valid_lons - lon) ** 2)
    dist = np.where(valid_mask.values, dist, np.inf)
    iy, ix = np.unravel_index(np.argmin(dist), dist.shape)
    cell_lat, cell_lon = float(ds["lat"][iy]), float(ds["lon"][ix])
    print(f"  nearest cell with real data: ({cell_lat}, {cell_lon}), "
          f"~{dist[iy, ix]*111:.0f} km from the AOI centroid (coastal cells right at the AOI are masked)")

    point = ds.sel(lat=cell_lat, lon=cell_lon)
    df = point["rain"].to_dataframe().reset_index()[["time", "rain"]]
    df.columns = ["date", "precipitation_mm"]
    df = df[(df["precipitation_mm"] >= 0)]  # IMD uses negative sentinels for missing/masked cells

    out_path = OUT_DIR / "imd_daily.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path} ({len(df)} daily rows)")

    window = df[(df["date"] >= "2015-11-08") & (df["date"] <= "2015-12-14")]
    top5 = window.sort_values("precipitation_mm", ascending=False).head(8)
    print("\nTop rainiest days in the event window, per IMD (independent of Open-Meteo):")
    for _, row in top5.iterrows():
        print(f"  {row['date'].date()}: {row['precipitation_mm']:.0f} mm")

    nov23 = window[window["date"] == "2015-11-23"]["precipitation_mm"]
    dec01 = window[window["date"] == "2015-12-01"]["precipitation_mm"]
    print(f"\nDirect comparison on the two dates every other source flags as peaks:")
    print(f"  23 Nov 2015 -- IMD: {nov23.values[0]:.0f} mm   |  Open-Meteo (task 1.7): 14.6 mm  |  NRSC report / news: ~227-240 mm")
    print(f"  01 Dec 2015 -- IMD: {dec01.values[0]:.0f} mm   |  Open-Meteo (task 1.7): 48.8 mm  |  NRSC report / news: ~340-490 mm")


if __name__ == "__main__":
    main()
