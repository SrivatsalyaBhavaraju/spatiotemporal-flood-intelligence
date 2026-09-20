"""
Task 5.1 -- compute F1/accuracy per phase transition for the FINAL, tuned
GNN model, across all three splits (train/val/test) -- task 4.2 itself
only needed and saved the test-split number (that's the one genuine
held-out result; train/val exist purely to select hyperparameters, not to
report against). This task reproduces that same final model and adds the
train/val breakdown, for task 5.3's baseline-vs-GNN comparison table.

Reproduces task 4.2's exact final retrain rather than re-tuning anything:
`train_model()` (task 4.1) is fully deterministic given the same
train_mask/lr/SEED (single fixed seed, full-batch training, no shuffling
or dropout) -- confirmed below, not assumed: this script's own test-split
metrics are checked byte-for-byte against task 4.2's saved
`final_test_evaluation`, and any mismatch raises rather than silently
reporting a different number under the same "final model" label.

Also persists the model weights + feature stats this time -- task 4.2
discarded them in memory after printing metrics, since it never needed to
reuse the model itself.

*** WHY THIS REPORT ALSO INCLUDES AUC-ROC, NOT JUST F1/ACCURACY *** F1 at
a single tuned threshold can look excellent purely from matching a
transition's base rate (rising->peak/peak->receding are 87-96% positive)
even when the model has learned nothing that actually discriminates which
SPECIFIC segments flood -- and that is exactly what a first real run of
this script found: the final model predicts "flooded" for 100% of test
segments on every transition (confirmed directly, not assumed), which is
why AUC-ROC (threshold-independent, the standard way to check for this
exact failure mode) is computed here too. Real, disclosed result: AUC is
~0.70-0.76 on train/val (genuine, real discrimination) but collapses to
~0.46-0.50 on test (statistically indistinguishable from random) for both
flood-relevant transitions. This means the F1=0.982/0.973 test numbers
task 4.2 reported are real arithmetic but do NOT reflect learned
per-segment discrimination transferring to the test wards -- a sharper,
concrete version of the small-N-of-wards variance task 3.7/4.2 already
flagged. Deliberately NOT re-litigated or fixed here -- task 5.4 exists
specifically to sanity-check comparison anomalies like this one; this
report's job is to surface it honestly, not resolve it.

AUC is hand-rolled (`compute_auc()`, the standard tie-aware Mann-Whitney-U
formulation) rather than adding scipy/scikit-learn as a new dependency --
this repo's established convention (see task 3.6's
`binary_classification_metrics()`).

Usage:
    python src/models/gnn/evaluate_final_model.py

Required inputs:
    (everything task 4.1 needs) + data/processed/ground_truth/hyperparameter_tuning_report.json (task 4.2)

Outputs (data/processed/ground_truth/):
    gnn_final_model.pt                    -- final tuned model weights
    gnn_final_model_feature_stats.json    -- normalization stats (mean/std) + selected threshold
    gnn_final_per_transition_report.json  -- F1/accuracy/AUC per transition per split (train/val/test)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.gnn.build_graph_snapshot_dataset import load_dataset  # noqa: E402
from src.models.gnn.train_gnn import (  # noqa: E402
    EPOCHS,
    Phase4TrainingError,
    evaluate_model,
    load_labels_for_attach,
    load_split_masks,
    predict_probabilities,
    train_model,
)

TUNING_REPORT_PATH = REPO_ROOT / "data" / "processed" / "ground_truth" / "hyperparameter_tuning_report.json"
OUT_DIR = REPO_ROOT / "data" / "processed" / "ground_truth"
MODEL_PATH = OUT_DIR / "gnn_final_model.pt"
FEATURE_STATS_PATH = OUT_DIR / "gnn_final_model_feature_stats.json"
REPORT_PATH = OUT_DIR / "gnn_final_per_transition_report.json"


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase4TrainingError(
            f"Required artifact not found: {path}\nThis file is produced by {produced_by}. Run it first."
        )
    return path


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray):
    """AUC-ROC via the tie-aware Mann-Whitney-U formulation (average rank
    for tied scores) -- hand-rolled to avoid adding scipy/scikit-learn as
    a new dependency (see module docstring). Returns None when a split has
    only one class (AUC is undefined, e.g. pre_event->rising is always
    0% positive by construction -- not an error condition)."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(y_prob, kind="mergesort")
    sorted_probs = y_prob[order]
    ranks = np.empty(len(y_prob))
    i = 0
    n = len(sorted_probs)
    while i < n:
        j = i
        while j < n and sorted_probs[j] == sorted_probs[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0  # 1-indexed average rank over the tie block
        i = j

    sum_ranks_pos = ranks[y_true == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def add_auc_to_report(report: dict, probabilities: dict, split_masks: dict) -> dict:
    """Augments evaluate_model()'s per-transition/per-split metrics dict
    with an "auc" key computed from the raw probabilities (evaluate_model()
    only sees post-threshold 0/1 predictions, so AUC has to be computed
    separately from the same probabilities)."""
    for label, (y_true, y_prob) in probabilities.items():
        for split_name, mask in split_masks.items():
            if split_name not in report.get(label, {}):
                continue
            report[label][split_name]["auc"] = compute_auc(
                y_true[mask].squeeze(-1).numpy(), y_prob[mask].squeeze(-1).numpy()
            )
    return report


def load_tuned_hyperparameters(path: Path = TUNING_REPORT_PATH) -> tuple:
    """(learning_rate, threshold) task 4.2 selected -- reused, not
    re-tuned, here."""
    _require_file(path, "src/models/gnn/tune_hyperparameters.py (task 4.2)")
    tuning_report = json.loads(path.read_text(encoding="utf-8"))
    return tuning_report["selected_learning_rate"], tuning_report["selected_threshold"], tuning_report


def check_reproduces_tuning_report(report: dict, tuning_report: dict) -> None:
    """Fails loudly if this script's re-derived test metrics don't
    EXACTLY match task 4.2's own saved final_test_evaluation -- the
    concrete evidence that train_model() is genuinely deterministic here,
    not an assumption. Compares only the keys task 4.2 itself saved (f1/
    precision/recall/accuracy/tp/fp/fn/tn) so this still works if `report`
    already has extra keys added (e.g. `auc`, which task 4.2 never computed)."""
    expected = tuning_report["final_test_evaluation"]
    for label, per_split in expected.items():
        actual_test = report[label]["test"]
        expected_test = per_split["test"]
        mismatched = {k: (actual_test.get(k), v) for k, v in expected_test.items() if actual_test.get(k) != v}
        if mismatched:
            raise Phase4TrainingError(
                f"Reproducibility check failed for {label}: expected {expected_test}, got {actual_test}. "
                f"train_model() may no longer be deterministic, or an input artifact has changed."
            )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("\nLoading task 2.7/2.8/3.4 model input + task 4.2's selected hyperparameters ...")
    try:
        ds = load_dataset()
        labels = load_labels_for_attach()
        ds.attach_labels(labels)
        best_lr, best_threshold, tuning_report = load_tuned_hyperparameters()
        split_masks = load_split_masks(ds.node_order)
    except Phase4TrainingError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> selected lr={best_lr}, threshold={best_threshold} (task 4.2)")

    train_val_mask = split_masks["train"] | split_masks["val"]
    print(f"\nReproducing task 4.2's final retrain on {int(train_val_mask.sum())} train+val segments ...")
    model, curve, feature_stats = train_model(ds, train_val_mask, epochs=EPOCHS, lr=best_lr, device=device)

    phase_id_to_name = {p["phase_id"]: p["phase_name"] for p in ds.phases}
    print("\nEvaluating on ALL three splits (train/val/test) per transition ...")
    report = evaluate_model(model, ds, split_masks, phase_id_to_name, feature_stats, device=device, threshold=best_threshold)
    for label, per_split in report.items():
        print(f"  {label}: " + ", ".join(f"{s}_f1={m['f1']}" for s, m in per_split.items()))

    print("\nChecking this reproduces task 4.2's own saved test-split numbers exactly ...")
    check_reproduces_tuning_report(report, tuning_report)
    print("  -> confirmed: test-split metrics match task 4.2's final_test_evaluation byte-for-byte")

    print("\nComputing AUC-ROC per transition per split (threshold-independent -- see module docstring) ...")
    probabilities = predict_probabilities(model, ds, feature_stats, phase_id_to_name, device=device)
    report = add_auc_to_report(report, probabilities, split_masks)
    for label, per_split in report.items():
        print(f"  {label}: " + ", ".join(f"{s}_auc={m.get('auc')}" for s, m in per_split.items()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    feature_mean, feature_std = feature_stats
    FEATURE_STATS_PATH.write_text(json.dumps({
        "feature_mean": feature_mean.tolist(), "feature_std": feature_std.tolist(),
        "learning_rate": best_lr, "threshold": best_threshold, "epochs": EPOCHS,
    }, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved -> {MODEL_PATH}, {FEATURE_STATS_PATH}, {REPORT_PATH}")

    print("\n=== Per-transition, per-split summary ===")
    print(json.dumps(report, indent=2, default=str))
    print("\nDone. Next: task 5.2 computes the same per-transition/per-split table for the baseline,")
    print("and task 5.3 compares the two.")


if __name__ == "__main__":
    main()
