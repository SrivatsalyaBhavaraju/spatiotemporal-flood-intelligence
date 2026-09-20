"""
Task 5.2 -- compute F1/accuracy per phase transition for the baseline
model, across all three splits (train/val/test), matching task 5.1's GNN
report shape exactly so task 5.3 can compare them directly.

Formalizes into its own dedicated, re-runnable report what was previously
only a smoke-test side effect buried inside task 3.7's
`build_training_harness.py` (`baseline_metrics_by_split_and_transition`).
That embedded version is now STALE -- it was computed against the
pre-task-4.4 `baseline_predictions.csv`, before the peak/receding
ground-truth fix. Reusing task 3.7's own `evaluate_split()` (not
reimplemented) against the CURRENT `baseline_predictions.csv` (task 4.3's
last real run, already reflecting task 4.4's fix) and the current
`segment_splits.csv` (task 3.7) gives an up-to-date, dedicated Phase 5
artifact rather than relying on a side effect of a different task's script.

Note the baseline itself isn't "trained" on a split the way the GNN is --
these per-split numbers exist purely so task 5.3 can compare the baseline
against the GNN on the SAME held-out test segments, not because train/val
mean anything for a static rule-based model.

Usage:
    python src/models/baseline/evaluate_per_transition.py

Required inputs:
    data/processed/ground_truth/baseline_predictions.csv (task 3.6/4.3)
    data/processed/model_input/segment_splits.csv          (task 3.7)
    data/processed/model_input/schema.json                  (task 2.7)

Outputs (data/processed/ground_truth/):
    baseline_final_per_transition_report.json  -- F1/accuracy per transition per split (train/val/test)
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline.rule_based_propagation import Phase3BaselineError  # noqa: E402
from src.models.gnn.build_training_harness import evaluate_split  # noqa: E402 -- reused, not reimplemented

MODEL_INPUT_DIR = REPO_ROOT / "data" / "processed" / "model_input"
SCHEMA_PATH = MODEL_INPUT_DIR / "schema.json"
SEGMENT_SPLITS_PATH = MODEL_INPUT_DIR / "segment_splits.csv"
GROUND_TRUTH_DIR = REPO_ROOT / "data" / "processed" / "ground_truth"
BASELINE_PREDICTIONS_PATH = GROUND_TRUTH_DIR / "baseline_predictions.csv"
REPORT_PATH = GROUND_TRUTH_DIR / "baseline_final_per_transition_report.json"


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase3BaselineError(
            f"Required artifact not found: {path}\nThis file is produced by {produced_by}. Run it first."
        )
    return path


def compute_baseline_report(
    baseline_predictions: pd.DataFrame, segment_splits: pd.DataFrame, phase_id_to_name: dict,
) -> dict:
    """{transition_label: {split_name: metrics}} -- same shape as task
    5.1's GNN report. `evaluate_split()` (task 3.7) does the actual
    per-split metric computation; this just groups by transition first."""
    report = {}
    for (x_t, y_t1), group in baseline_predictions.groupby(["x_t_phase_id", "y_t1_phase_id"]):
        group = group.merge(segment_splits[["segment_id", "split"]], on="segment_id", how="left")
        label = f"{phase_id_to_name[x_t]}->{phase_id_to_name[y_t1]}"
        report[label] = evaluate_split(group["y_true"], group["y_pred"], group["split"])
    return report


def main():
    print("Loading task 4.3's current baseline predictions + task 3.7's segment splits ...")
    try:
        baseline_predictions = pd.read_csv(
            _require_file(BASELINE_PREDICTIONS_PATH, "src/models/baseline/rule_based_propagation.py (task 3.6/4.3)")
        )
        segment_splits = pd.read_csv(_require_file(SEGMENT_SPLITS_PATH, "src/models/gnn/build_training_harness.py (task 3.7)"))
        schema = json.loads(_require_file(SCHEMA_PATH, "src/models/gnn/build_model_input_schema.py (task 2.7)").read_text(encoding="utf-8"))
    except Phase3BaselineError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    phase_id_to_name = {p["phase_id"]: p["phase_name"] for p in schema["phases"]}
    print(f"  -> {len(baseline_predictions)} predictions, {len(segment_splits)} split assignments")

    print("\nComputing F1/accuracy per transition per split ...")
    report = compute_baseline_report(baseline_predictions, segment_splits, phase_id_to_name)
    for label, per_split in report.items():
        print(f"  {label}: " + ", ".join(f"{s}_f1={m['f1']}" for s, m in per_split.items()))

    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved -> {REPORT_PATH}")

    print("\n=== Per-transition, per-split summary ===")
    print(json.dumps(report, indent=2, default=str))
    print("\nDone. Next: task 5.3 compares this against task 5.1's GNN report.")


if __name__ == "__main__":
    main()
