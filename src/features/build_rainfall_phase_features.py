"""
Task 2.5 -- bias-corrected rainfall_t per phase window.

Per working.md's own recommendation (SS on task 2.5): Open-Meteo's absolute
hourly values are NOT trustworthy on their own (see task 1.7/1.8 comparison --
IMD runs 2-6x higher on most days, and matches the literature's peak values
far better). So instead of feeding Open-Meteo mm directly into rainfall_t:

  1. For each day, compute Open-Meteo's *hourly shape* -- each hour's share
     of that day's Open-Meteo total (a fraction that sums to 1 across the 24
     hours).
  2. Multiply that shape by IMD's independently-sourced DAILY total for the
     same day.
  3. Sum the resulting bias-corrected hourly values within each phase window
     to get that phase's rainfall_t.

This is the standard temporal-downscaling pattern: coarse-but-trusted
magnitude (IMD, ~28km) + fine-but-only-trusted-for-shape temporal
distribution (Open-Meteo, hourly).

Usage:
    python src/features/build_rainfall_phase_features.py

Inputs (already produced by tasks 1.7/1.8):
    data/raw/rainfall/open_meteo_hourly.csv
    data/raw/rainfall/imd_daily.csv

Outputs (data/processed/features/):
    rainfall_bias_corrected_hourly.csv  -- full corrected hourly series, for QA
    rainfall_phase_features.csv         -- one row per phase: rainfall_t (mm)
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OPEN_METEO_CSV = REPO_ROOT / "data" / "raw" / "rainfall" / "open_meteo_hourly.csv"
IMD_CSV = REPO_ROOT / "data" / "raw" / "rainfall" / "imd_daily.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "features"

# ---------------------------------------------------------------------------
# EDIT THIS if your working.md's phase-boundary table differs from what's
# inferred here. Must stay contiguous and non-overlapping (needed for
# cumulative_rainfall_t in task 2.6 to be correct) and should span the full
# event window (2015-11-08 to 2015-12-14).
# ---------------------------------------------------------------------------
PHASES = [
    {"phase_id": 0, "phase_name": "pre_event", "start": "2015-11-08", "end": "2015-11-28"},
    {"phase_id": 1, "phase_name": "rising",    "start": "2015-11-29", "end": "2015-11-30"},
    {"phase_id": 2, "phase_name": "peak",      "start": "2015-12-01", "end": "2015-12-02"},
    {"phase_id": 3, "phase_name": "receding",  "start": "2015-12-03", "end": "2015-12-14"},
]


def bias_correct_hourly(open_meteo: pd.DataFrame, imd: pd.DataFrame) -> pd.DataFrame:
    """Return open_meteo's hourly rows with an added `rainfall_mm_corrected`
    column: IMD's daily total redistributed by Open-Meteo's hourly shape.
    """
    om = open_meteo.copy()
    om["date"] = om["time"].dt.floor("D")

    daily_om_total = om.groupby("date")["precipitation_mm"].transform("sum")
    # Hour's share of that day's Open-Meteo total. If Open-Meteo shows zero
    # rain all day but IMD shows nonzero, fall back to an even 1/24 spread
    # rather than dividing by zero.
    with pd.option_context("mode.use_inf_as_na", True):
        shape = (om["precipitation_mm"] / daily_om_total).fillna(1 / 24)
    shape = shape.where(daily_om_total > 0, 1 / 24)

    imd_lookup = imd.set_index("date")["precipitation_mm"]
    om["imd_daily_total_mm"] = om["date"].map(imd_lookup)
    missing = om["imd_daily_total_mm"].isna().sum()
    if missing:
        print(f"  WARNING: {missing} hourly rows have no matching IMD day "
              f"(outside IMD's coverage?) -- these rows get corrected value 0.")
    om["imd_daily_total_mm"] = om["imd_daily_total_mm"].fillna(0.0)

    om["rainfall_mm_corrected"] = shape * om["imd_daily_total_mm"]
    return om[["time", "date", "precipitation_mm", "imd_daily_total_mm", "rainfall_mm_corrected"]]


def aggregate_to_phases(hourly_corrected: pd.DataFrame, phases: list) -> pd.DataFrame:
    rows = []
    for p in phases:
        start, end = pd.Timestamp(p["start"]), pd.Timestamp(p["end"]) + pd.Timedelta(days=1)
        mask = (hourly_corrected["time"] >= start) & (hourly_corrected["time"] < end)
        window = hourly_corrected.loc[mask]
        rainfall_t = window["rainfall_mm_corrected"].sum()
        rows.append({
            "phase_id": p["phase_id"],
            "phase_name": p["phase_name"],
            "start_date": p["start"],
            "end_date": p["end"],
            "n_hours": len(window),
            "rainfall_t": round(rainfall_t, 3),
        })
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    open_meteo = pd.read_csv(OPEN_METEO_CSV, parse_dates=["time"])
    imd = pd.read_csv(IMD_CSV, parse_dates=["date"])

    print("Bias-correcting Open-Meteo hourly shape against IMD daily magnitude...")
    hourly_corrected = bias_correct_hourly(open_meteo, imd)
    hourly_out = OUT_DIR / "rainfall_bias_corrected_hourly.csv"
    hourly_corrected.to_csv(hourly_out, index=False)
    print(f"Saved -> {hourly_out} ({len(hourly_corrected)} rows)")

    print("\nAggregating corrected hourly series into phase windows...")
    phase_df = aggregate_to_phases(hourly_corrected, PHASES)
    phase_out = OUT_DIR / "rainfall_phase_features.csv"
    phase_df.to_csv(phase_out, index=False)

    print(f"Saved -> {phase_out}\n")
    print(phase_df.to_string(index=False))

    total_event = phase_df["rainfall_t"].sum()
    print(f"\nTotal bias-corrected rainfall across full event window: {total_event:.0f} mm")
    print("Sanity check: Phase 2 (peak) should now be the clear maximum, "
          "consistent with IMD's 2 Dec value and the literature (~340-490mm) "
          "-- unlike raw Open-Meteo, which flagged 16 Nov as the max (task 1.8 docstring).")


if __name__ == "__main__":
    main()