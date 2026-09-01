"""
Task 1.9 -- Google Earth Engine access + Sentinel-1 GRD collection query.

This is the entry point for Objective 1's PRIMARY ground truth. The
feasibility check (working.md SS1.5/SS1.9) already confirmed via the live
Copernicus catalog that 4 Sentinel-1A GRD scenes cover the study wards in
the event window -- this script confirms the same collection is reachable
through GEE (which is what task 3.1's actual SAR change-detection will run
on) and prints the exact image IDs to use there.

SETUP (one-time, interactive -- can't be done from an unattended script):
    1. pip install earthengine-api   (already in requirements.txt)
    2. Run this once yourself, in a real terminal:
           earthengine authenticate
       This opens a Google OAuth browser flow tied to a Google account with
       GEE access (sign up free at https://signup.earthengine.google.com if
       your account doesn't have it yet). It stores a token locally after.
    3. Re-run this script -- ee.Initialize() will pick up the stored token.

Usage:
    python src/ground_truth/query_sentinel1_gee.py
"""
from pathlib import Path

import ee
import geopandas as gpd

REPO_ROOT = Path(__file__).resolve().parents[2]
WARDS = REPO_ROOT / "data" / "raw" / "wards" / "study_wards.geojson"

# Confirmed via the live Copernicus catalog (working.md SS1.5) -- the 4 S1A
# passes that actually cover the study wards in the event window.
CONFIRMED_DATES = ["2015-11-12", "2015-11-24", "2015-12-06", "2015-12-18"]
PRE_EVENT_DATE = "2015-11-24"   # pre-peak baseline for change detection (task 3.1)
POST_EVENT_DATE = "2015-12-06"  # closest post-peak pass

# Every GEE API call needs a Cloud Project now -- this is the team's project,
# created during task 1.9's registration (developing.md Phase 1 notes).
GEE_PROJECT = "flood-intelligence-507219"


def main():
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception as e:
        print("GEE not authenticated yet. Run this in a terminal, then re-run this script:")
        print("    earthengine authenticate")
        print(f"\n(original error: {e})")
        return

    print("GEE authenticated OK.\n")

    wards = gpd.read_file(WARDS)
    minx, miny, maxx, maxy = wards.total_bounds
    aoi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    print(f"AOI bbox: ({minx:.4f},{miny:.4f}) - ({maxx:.4f},{maxy:.4f})")

    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate("2015-11-01", "2015-12-20")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    )

    info = collection.getInfo()
    features = info.get("features", [])
    print(f"\nFound {len(features)} Sentinel-1 GRD/IW scenes over the AOI in the event window:")
    found_dates = set()
    for f in features:
        props = f["properties"]
        img_id = f["id"]
        date = props.get("system:time_start")
        date_str = ee.Date(date).format("YYYY-MM-dd").getInfo() if date else "?"
        found_dates.add(date_str)
        print(f"  {date_str}  {img_id}  orbit={props.get('orbitProperties_pass')}")

    print(f"\nExpected (from the Copernicus catalog check): {CONFIRMED_DATES}")
    missing = set(CONFIRMED_DATES) - found_dates
    if missing:
        print(f"  ** WARNING: dates confirmed via Copernicus but not found in GEE: {sorted(missing)} **")
    else:
        print("  All 4 confirmed dates are present in GEE too. Good -- task 3.1 can use this collection directly.")

    print(f"\nFor task 3.1's change detection: pre-event = {PRE_EVENT_DATE}, post-event = {POST_EVENT_DATE}")


if __name__ == "__main__":
    main()
