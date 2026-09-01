"""
Task 1.7 -- pull hourly rainfall for the confirmed event window over the
study wards. This is Objective 1's dynamic driver feature: rainfall_t and
cumulative_rainfall_t (working.md SS1.7) get attached to every graph node at
every timestep (task 2.5/2.6) -- it's what makes the model's input a time
series rather than a single static snapshot.

Source: Open-Meteo Historical Weather API (ERA5-Land reanalysis), free, no
auth. Resolution is coarse (~9-11km grid cell) relative to the ward cluster
(~5km across), so one query point representing the AOI centroid is a
reasonable approximation -- see the note printed at the end of this script.

Usage:
    python src/ground_truth/fetch_rainfall.py

Output (data/raw/rainfall/, gitignored):
    open_meteo_hourly.csv  -- columns: time, precipitation_mm
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
WARDS = REPO_ROOT / "data" / "raw" / "wards" / "study_wards.geojson"
OUT_DIR = REPO_ROOT / "data" / "raw" / "rainfall"

# Confirmed event window (developing.md Phase 0 / working.md SS1.9)
START_DATE = "2015-11-08"
END_DATE = "2015-12-14"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wards = gpd.read_file(WARDS)
    centroid = wards.union_all().centroid
    lat, lon = centroid.y, centroid.x
    print(f"Study-ward AOI centroid: ({lat:.4f}, {lon:.4f})")

    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START_DATE, "end_date": END_DATE,
        "hourly": "precipitation",
        "timezone": "Asia/Kolkata",
    }
    print(f"Querying Open-Meteo archive API for {START_DATE} to {END_DATE} ...")
    r = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "precipitation_mm": data["hourly"]["precipitation"],
    })
    out_path = OUT_DIR / "open_meteo_hourly.csv"
    df.to_csv(out_path, index=False)

    total = df["precipitation_mm"].sum()
    peak_hour = df.loc[df["precipitation_mm"].idxmax()]
    daily = df.set_index("time").resample("D")["precipitation_mm"].sum()
    top5 = daily.sort_values(ascending=False).head(5)

    print(f"\nSaved -> {out_path}  ({len(df)} hourly rows)")
    print(f"Total rainfall over window: {total:.0f} mm")
    print(f"Peak single hour: {peak_hour['precipitation_mm']:.1f} mm at {peak_hour['time']}")
    print("Top 5 rainiest days (cross-check against the confirmed 23 Nov / 1 Dec peaks):")
    for d, v in top5.items():
        print(f"  {d.date()}: {v:.0f} mm")

    print("\nNote: Open-Meteo/ERA5-Land grid resolution (~9-11km) is coarser than the")
    print("~5km study cluster -- this one AOI-centroid series is a reasonable proxy for")
    print("the whole cluster, not a per-ward rainfall estimate. Task 1.8 (IMD daily)")
    print("cross-checks it against a second, independently-gridded source.")


if __name__ == "__main__":
    main()
