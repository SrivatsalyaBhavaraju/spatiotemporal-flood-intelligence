"""
Task 4.2 -- hyperparameter tuning. Directly addresses a real limitation
task 3.7 and 4.1 already disclosed: with only 16 study wards, a SINGLE
fixed 70/15/15 ward split gives noisy, high-variance val/test metrics that
depend heavily on which specific wards happened to land where (task 3.7's
own smoke test showed the same baseline model scoring 6.99%/53.54%/3.54%
accuracy on train/val/test for one transition, purely from ward
selection). This task replaces that single point estimate with k-fold
cross-validation over the wards task 3.7 assigned to train+val (the
official TEST wards stay completely untouched until the final step, never
used for model selection).

Method:
  1. Partition task 3.7's train+val wards into K folds (same greedy
     segment-count-balancing idea as build_ward_split(), generalized to K
     groups).
  2. For each candidate learning rate, train K models (one per fold, held
     out in turn) using task 4.1's own train_model()/evaluate_model() --
     not reimplemented -- and aggregate mean+/-std F1 across folds.
  3. Select the learning rate with the best mean F1 averaged over the two
     flood-relevant transitions (rising->peak, peak->receding;
     pre_event->rising is trivially 0 for every config and uninformative
     for selection).
  4. Re-run the K folds once more with the selected learning rate,
     pooling their held-out (y_true, y_prob) pairs to tune the decision
     threshold (task 4.1 used a naive fixed 0.5) via a simple sweep,
     maximizing the same two-transition F1 average.
  5. Retrain a FINAL model on ALL train+val wards with the selected
     learning rate, and evaluate ONCE on task 3.7's official test wards
     using the tuned threshold -- this is the only number that counts as
     a genuine held-out test result; everything in steps 1-4 only ever
     touches train+val wards.

Scope, disclosed: epochs is fixed at task 4.1's 300 (not independently
grid-searched) to keep total runtime bounded (~17 real training runs at
~30s each); the periods=1 architectural framing is unchanged (that was a
design decision confirmed with the user in task 4.1, not a tuning knob).

Usage:
    python src/models/gnn/tune_hyperparameters.py

Required inputs:
    (everything task 4.1 needs) + data/processed/model_input/segment_splits.csv (task 3.7)

Outputs (data/processed/ground_truth/):
    hyperparameter_tuning_report.json  -- full CV grid results, selected
        LR/threshold, and the final tuned test evaluation
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.gnn.build_graph_snapshot_dataset import load_dataset  # noqa: E402
from src.models.gnn.train_gnn import (  # noqa: E402
    FUSED_LABELS_PATH,
    Phase4TrainingError,
    evaluate_model,
    load_labels_for_attach,
    load_split_masks,
    predict_probabilities,
    train_model,
)

SEGMENT_SPLITS_PATH = REPO_ROOT / "data" / "processed" / "model_input" / "segment_splits.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "ground_truth"
REPORT_PATH = OUT_DIR / "hyperparameter_tuning_report.json"

K_FOLDS = 4
LEARNING_RATE_GRID = [0.005, 0.01, 0.02]
EPOCHS = 300  # fixed at task 4.1's value -- see module docstring's "Scope" note
FLOOD_RELEVANT_TRANSITIONS = ["rising->peak", "peak->receding"]  # pre_event->rising is uninformative for selection
THRESHOLD_GRID = np.arange(0.1, 0.95, 0.05)


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase4TrainingError(
            f"Required artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first."
        )
    return path


# --------------------------------------------------------------------------
# Ward-fold construction (pure logic -- unit-tested)
# --------------------------------------------------------------------------

def build_kfold_ward_groups(ward_sizes: dict, k: int, seed: int = 42) -> list:
    """k roughly-segment-count-balanced groups of ward_no -- same greedy
    balancing idea as task 3.7's build_ward_split(), generalized from 3
    groups to k."""
    rng = np.random.default_rng(seed)
    ward_order = list(ward_sizes.keys())
    rng.shuffle(ward_order)
    groups = [[] for _ in range(k)]
    group_totals = [0] * k
    for ward_no in ward_order:
        i = int(np.argmin(group_totals))
        groups[i].append(ward_no)
        group_totals[i] += ward_sizes[ward_no]
    return groups


def build_mask_for_segments(node_order: list, segment_ids: set) -> torch.Tensor:
    return torch.tensor([s in segment_ids for s in node_order])


# --------------------------------------------------------------------------
# Cross-validation (pure-ish aggregation logic -- unit-tested; the actual
# training/eval loop is task 4.1's own code, validated by a real run here)
# --------------------------------------------------------------------------

def aggregate_fold_metrics(fold_reports: list) -> dict:
    """{transition_label: {metric: {"mean", "std", "values"}}} across
    fold_reports (each one evaluate_model()'s output for a single fold,
    using a "heldout" split key)."""
    metric_names = ["f1", "precision", "recall", "accuracy"]
    aggregated = {}
    for label in fold_reports[0]:
        aggregated[label] = {}
        for m in metric_names:
            values = [r[label]["heldout"][m] for r in fold_reports]
            aggregated[label][m] = {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4), "values": values}
    return aggregated


def mean_flood_relevant_f1(aggregated: dict) -> float:
    return float(np.mean([aggregated[label]["f1"]["mean"] for label in FLOOD_RELEVANT_TRANSITIONS if label in aggregated]))


def sweep_threshold(pooled_probs: dict, thresholds=THRESHOLD_GRID) -> tuple:
    """pooled_probs: {transition_label: (y_true concatenated, y_prob concatenated)}.
    Returns (best_threshold, per_threshold_f1) maximizing the flood-relevant
    two-transition mean F1."""
    from src.models.baseline.rule_based_propagation import binary_classification_metrics

    per_threshold = {}
    for t in thresholds:
        f1s = []
        for label in FLOOD_RELEVANT_TRANSITIONS:
            if label not in pooled_probs:
                continue
            y_true, y_prob = pooled_probs[label]
            y_pred = (y_prob > t).float()
            m = binary_classification_metrics(y_true.squeeze(-1).tolist(), y_pred.squeeze(-1).tolist())
            f1s.append(m["f1"])
        per_threshold[round(float(t), 2)] = round(float(np.mean(f1s)), 4)
    best_threshold = max(per_threshold, key=per_threshold.get)
    return best_threshold, per_threshold


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("\nLoading task 2.7/2.8/3.4/3.7 model input + loader ...")
    try:
        ds = load_dataset()
        labels = load_labels_for_attach()
        ds.attach_labels(labels)
        splits_df = pd.read_csv(_require_file(SEGMENT_SPLITS_PATH, "src/models/gnn/build_training_harness.py (task 3.7)"))
    except Phase4TrainingError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    test_segment_ids = set(splits_df[splits_df["split"] == "test"]["segment_id"])
    tune_df = splits_df[splits_df["split"] != "test"]
    ward_sizes = tune_df.groupby("ward_no").size().to_dict()
    folds = build_kfold_ward_groups(ward_sizes, K_FOLDS)
    print(f"  -> tuning over {len(tune_df)} train+val segments across {len(ward_sizes)} wards, "
          f"{K_FOLDS} folds; {len(test_segment_ids)} official test segments held out untouched")

    print(f"\n=== Cross-validated grid search: LR in {LEARNING_RATE_GRID} ===")
    grid_results = {}
    for lr in LEARNING_RATE_GRID:
        fold_reports = []
        for i, fold_wards in enumerate(folds):
            heldout_ids = set(tune_df[tune_df["ward_no"].isin(fold_wards)]["segment_id"])
            train_ids = set(tune_df[~tune_df["ward_no"].isin(fold_wards)]["segment_id"])
            train_mask = build_mask_for_segments(ds.node_order, train_ids)
            heldout_mask = build_mask_for_segments(ds.node_order, heldout_ids)

            model, _, feature_stats = train_model(ds, train_mask, epochs=EPOCHS, lr=lr, device=device)
            phase_id_to_name = {p["phase_id"]: p["phase_name"] for p in ds.phases}
            report = evaluate_model(model, ds, {"heldout": heldout_mask}, phase_id_to_name, feature_stats, device=device)
            fold_reports.append(report)
            print(f"  lr={lr} fold={i + 1}/{K_FOLDS} done")

        aggregated = aggregate_fold_metrics(fold_reports)
        grid_results[lr] = aggregated
        print(f"  lr={lr}: mean flood-relevant F1 = {mean_flood_relevant_f1(aggregated):.4f}")

    best_lr = max(LEARNING_RATE_GRID, key=lambda lr: mean_flood_relevant_f1(grid_results[lr]))
    print(f"\nSelected learning rate: {best_lr}")

    print(f"\n=== Threshold tuning: re-running {K_FOLDS} folds with lr={best_lr} to pool held-out predictions ===")
    pooled = {}
    for i, fold_wards in enumerate(folds):
        heldout_ids = set(tune_df[tune_df["ward_no"].isin(fold_wards)]["segment_id"])
        train_ids = set(tune_df[~tune_df["ward_no"].isin(fold_wards)]["segment_id"])
        train_mask = build_mask_for_segments(ds.node_order, train_ids)
        heldout_mask = build_mask_for_segments(ds.node_order, heldout_ids)

        model, _, feature_stats = train_model(ds, train_mask, epochs=EPOCHS, lr=best_lr, device=device)
        phase_id_to_name = {p["phase_id"]: p["phase_name"] for p in ds.phases}
        probs = predict_probabilities(model, ds, feature_stats, phase_id_to_name, device=device)
        for label, (y_true, y_prob) in probs.items():
            y_true_held, y_prob_held = y_true[heldout_mask], y_prob[heldout_mask]
            if label not in pooled:
                pooled[label] = (y_true_held, y_prob_held)
            else:
                prev_true, prev_prob = pooled[label]
                pooled[label] = (torch.cat([prev_true, y_true_held]), torch.cat([prev_prob, y_prob_held]))
        print(f"  fold {i + 1}/{K_FOLDS} predictions pooled")

    best_threshold, threshold_sweep = sweep_threshold(pooled)
    print(f"Selected decision threshold: {best_threshold} (default was 0.5)")

    print(f"\n=== Final retrain on all train+val wards with lr={best_lr}, evaluate on official test wards ===")
    final_train_mask = build_mask_for_segments(ds.node_order, set(tune_df["segment_id"]))
    final_model, final_curve, final_feature_stats = train_model(ds, final_train_mask, epochs=EPOCHS, lr=best_lr, device=device)
    test_mask = load_split_masks(ds.node_order)["test"]
    phase_id_to_name = {p["phase_id"]: p["phase_name"] for p in ds.phases}
    final_test_report = evaluate_model(
        final_model, ds, {"test": test_mask}, phase_id_to_name, final_feature_stats, device=device, threshold=best_threshold
    )
    for label, per_split in final_test_report.items():
        print(f"  {label}: test_f1={per_split['test']['f1']}")

    report = {
        "k_folds": K_FOLDS,
        "learning_rate_grid": LEARNING_RATE_GRID,
        "epochs_fixed_at": EPOCHS,
        "cv_grid_results": {str(lr): res for lr, res in grid_results.items()},
        "cv_mean_flood_relevant_f1_by_lr": {str(lr): round(mean_flood_relevant_f1(res), 4) for lr, res in grid_results.items()},
        "selected_learning_rate": best_lr,
        "threshold_sweep": threshold_sweep,
        "selected_threshold": best_threshold,
        "final_test_evaluation": final_test_report,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved tuning report -> {REPORT_PATH}")

    print("\nDone. Compare final_test_evaluation against task 4.1's single-split test numbers --")
    print("this one is backed by cross-validated hyperparameter selection, not one arbitrary split.")


if __name__ == "__main__":
    main()
