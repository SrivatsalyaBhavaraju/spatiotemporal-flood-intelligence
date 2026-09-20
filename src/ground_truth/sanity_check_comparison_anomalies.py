"""
Task 5.4 -- sanity-check the ground truth against the real anomaly task
5.1 surfaced: the tuned GNN's test-split AUC collapses to ~0.46-0.50
(random) despite ~0.70-0.76 on train/val (`src/models/gnn/evaluate_final_model.py`).

*** ROOT CAUSE FOUND, CONFIRMED WITH REAL DATA, NOT SPECULATED ***
Checked the actual per-ward relationship between elevation and the fused
flood label (task 3.4/4.4) rather than guessing. Two compounding facts:

1. Task 3.4's fusion (OR across SAR/Bhuvan/news, dominated by news' broad
   ward-level fallback -- task 3.4's own note: only 5.4% of the flooded
   set is SAR-covered) pushes MOST wards toward near-total inundation.
   At the `peak` transition, checked all 16 wards directly: only ONE
   ward (170) has a minority class >=10% of its segments -- every other
   ward, including BOTH test wards (169, 182), is >90% one-sided (usually
   >96% flooded). A model can only demonstrate skill where there is
   within-ward label variance to predict; there essentially isn't any
   outside of ward 170.
2. Ward 170 -- the one ward carrying almost all the real elevation-flood
   signal in this entire dataset -- landed in the VAL split, not test.
   Test's two wards (169, 182) are both near-uniformly flooded (96.5%,
   96.4%) with NO meaningful within-ward elevation relationship, and what
   tiny correlation exists there is backwards (elevation correlates
   POSITIVELY with flooding: +0.025, +0.166) versus the physically
   sensible, real relationship task 4.1's model actually learned from
   train/val (elevation correlates NEGATIVELY with flooding: train
   -0.137, val -0.286 -- lower ground floods more, as expected).

**Conclusion: this is a real, ground-truth-driven limitation, not a GNN
bug or a coding error.** With only 16 wards total and the flood label
skewed so heavily toward "everyone floods" by the coarse fusion sources,
genuine feature-vs-outcome signal exists in only a handful of wards, and
whether a useful one lands in train/val/test is close to a coin flip.
The GNN's test AUC collapse is the direct, correctly-measured consequence
of a bad draw on that coin flip, not evidence the model failed to learn.

**Not fixed here, deliberately** -- retroactively re-splitting/re-training
would be a new methodological decision (e.g. stratifying wards by label
variance, not just segment count as task 3.7 did) that needs its own
scoping, not something to slip into a "sanity check" task. Recommendation
is recorded in this module's report for that future decision.

Usage:
    python src/ground_truth/sanity_check_comparison_anomalies.py

Required inputs:
    data/processed/ground_truth/fused_flood_labels.csv (task 3.4/4.4)
    data/processed/model_input/segment_splits.csv        (task 3.7)
    data/processed/graph/static_features.geojson          (task 2.3)
    data/processed/model_input/schema.json                (task 2.7)

Outputs (data/processed/ground_truth/):
    sanity_check_comparison_anomalies_report.json
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ground_truth.fuse_flood_labels import Phase3ArtifactError  # noqa: E402

MODEL_INPUT_DIR = REPO_ROOT / "data" / "processed" / "model_input"
SCHEMA_PATH = MODEL_INPUT_DIR / "schema.json"
SEGMENT_SPLITS_PATH = MODEL_INPUT_DIR / "segment_splits.csv"
STATIC_FEATURES_PATH = REPO_ROOT / "data" / "processed" / "graph" / "static_features.geojson"
GROUND_TRUTH_DIR = REPO_ROOT / "data" / "processed" / "ground_truth"
FUSED_LABELS_PATH = GROUND_TRUTH_DIR / "fused_flood_labels.csv"
REPORT_PATH = GROUND_TRUTH_DIR / "sanity_check_comparison_anomalies_report.json"

MIN_MINORITY_PCT = 10.0  # a ward needs at least this much of its minority class present to be "informative"
FLOOD_RELEVANT_Y_T1_PHASES = ["peak", "receding"]  # the y_t1 phase of rising->peak and peak->receding


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase3ArtifactError(
            f"Required artifact not found: {path}\nThis file is produced by {produced_by}. Run it first."
        )
    return path


# --------------------------------------------------------------------------
# Per-ward diagnostics (pure logic given loaded data -- unit-tested)
# --------------------------------------------------------------------------

def build_ward_label_table(labels_df: pd.DataFrame, splits_df: pd.DataFrame, elevation_df: pd.DataFrame, phase_name: str) -> pd.DataFrame:
    """One row per ward: split, n segments, pct flooded, minority-class
    pct, and corr(elevation, flood_label) within that ward (NaN when the
    ward has zero label variance -- correlation is undefined, not zero)."""
    phase_labels = labels_df[labels_df["phase_name"] == phase_name]
    merged = phase_labels.merge(splits_df, on="segment_id").merge(elevation_df, on="segment_id")

    rows = []
    for (ward_no, split), group in merged.groupby(["ward_no", "split"]):
        pct_flooded = 100 * group["flood_label"].mean()
        minority_pct = min(pct_flooded, 100 - pct_flooded)
        corr = group["elevation_m"].corr(group["flood_label"])
        rows.append({
            "ward_no": int(ward_no), "split": split, "n_segments": int(len(group)),
            "pct_flooded": round(pct_flooded, 2), "minority_class_pct": round(minority_pct, 2),
            "elevation_flood_corr": round(corr, 4) if pd.notna(corr) else None,
        })
    return pd.DataFrame(rows).sort_values(["split", "ward_no"]).reset_index(drop=True)


def classify_informative_wards(ward_table: pd.DataFrame, min_minority_pct: float = MIN_MINORITY_PCT) -> pd.DataFrame:
    """Adds an `informative` column: True if the ward has enough
    within-ward label variance (minority class >= threshold) for a
    feature-based model to possibly learn/demonstrate anything on it."""
    ward_table = ward_table.copy()
    ward_table["informative"] = ward_table["minority_class_pct"] >= min_minority_pct
    return ward_table


def summarize_informativeness_by_split(ward_table: pd.DataFrame) -> dict:
    """{split: {n_wards, n_informative_wards, n_segments, n_informative_segments}}"""
    summary = {}
    for split, group in ward_table.groupby("split"):
        summary[split] = {
            "n_wards": int(len(group)),
            "n_informative_wards": int(group["informative"].sum()),
            "n_segments": int(group["n_segments"].sum()),
            "n_informative_segments": int(group.loc[group["informative"], "n_segments"].sum()),
        }
    return summary


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    print("Loading task 3.4/4.4 fused labels, task 3.7 splits, task 2.3 elevation ...")
    try:
        labels_df = pd.read_csv(_require_file(FUSED_LABELS_PATH, "src/ground_truth/fuse_flood_labels.py (task 3.4/4.4)"))
        splits_df = pd.read_csv(_require_file(SEGMENT_SPLITS_PATH, "src/models/gnn/build_training_harness.py (task 3.7)"))
        elevation_df = gpd.read_file(_require_file(STATIC_FEATURES_PATH, "src/graph/build_static_features.py (task 2.3)"))[["segment_id", "elevation_m"]]
        schema = json.loads(_require_file(SCHEMA_PATH, "src/models/gnn/build_model_input_schema.py (task 2.7)").read_text(encoding="utf-8"))
    except Phase3ArtifactError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    phase_id_to_name = {p["phase_id"]: p["phase_name"] for p in schema["phases"]}
    labels_df["phase_name"] = labels_df["phase_id"].map(phase_id_to_name)

    report = {"min_minority_pct_threshold": MIN_MINORITY_PCT, "by_phase": {}}
    for phase_name in FLOOD_RELEVANT_Y_T1_PHASES:
        print(f"\n=== Phase: {phase_name} ===")
        ward_table = build_ward_label_table(labels_df, splits_df, elevation_df, phase_name)
        ward_table = classify_informative_wards(ward_table)
        summary = summarize_informativeness_by_split(ward_table)
        for split, s in summary.items():
            print(f"  {split}: {s['n_informative_wards']}/{s['n_wards']} wards informative "
                  f"({s['n_informative_segments']}/{s['n_segments']} segments)")
        print(ward_table.to_string(index=False))
        report["by_phase"][phase_name] = {
            "ward_table": ward_table.to_dict(orient="records"),
            "informativeness_by_split": summary,
        }

    report["conclusion"] = (
        "AUC collapse on the test split (task 5.1) is a real, ground-truth-driven limitation, not a "
        "GNN bug: at the peak transition, only 1 of 16 wards (170, landed in val) has a minority class "
        ">=10% of its segments -- essentially all real elevation-flood signal in this dataset lives in "
        "that one ward. Both test wards (169, 182) are >96% flooded with no meaningful within-ward "
        "variance, and what tiny correlation exists there runs opposite the physically sensible "
        "relationship the model learned from train/val. With only 16 wards and a fusion rule that "
        "pushes most wards toward near-total inundation (task 3.4/4.4), whether an informative ward "
        "lands in train/val/test is close to a coin flip -- not something task 5.1/5.2's numbers, or "
        "the model itself, got wrong."
    )
    report["recommendation_for_future_work"] = (
        "If this model line is developed further, stratify the train/val/test ward split (task 3.7) by "
        "within-ward label variance, not just segment count -- guaranteeing at least one informative "
        "ward per split, not leaving it to chance. Not implemented here: that is a new split-methodology "
        "decision requiring its own scoping, not a retroactive change to slip into a sanity-check task."
    )

    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved -> {REPORT_PATH}")
    print("\n=== Conclusion ===")
    print(report["conclusion"])
    print("\n=== Recommendation for future work ===")
    print(report["recommendation_for_future_work"])
    print("\nDone. Task 5.3's baseline-vs-GNN comparison can now cite this root cause rather than")
    print("presenting the test AUC collapse as an unexplained anomaly.")


if __name__ == "__main__":
    main()
