"""
Task 5.3 -- baseline vs. GNN comparison: plots/tables testing working.md
SS1.6's core claim ("the learned model captures propagation patterns a
static baseline misses").

*** A METHODOLOGICAL PRECISION CORRECTION, FOUND WHILE BUILDING THIS,
DISCLOSED NOT HIDDEN *** task 5.1's "final tuned" model (task 4.2) was
retrained on train+val COMBINED (`tune_hyperparameters.py`'s own final
step) -- so its "val" AUC/F1 in `gnn_final_per_transition_report.json`
is NOT a fair holdout measurement; the model was fit to those exact
segments' labels. Citing it as evidence of generalization (as task 5.1's
own tracking note arguably implied) would overstate the case. This module
instead ALSO computes AUC for task 4.1's ORIGINAL model
(`gnn_model.pt`) -- trained on the "train" split ONLY, making its val
AND test splits genuinely held out -- as the methodologically clean
comparison point:

    Model                          | train AUC | val AUC (FAIR) | test AUC (FAIR)
    4.1 original (train-only)      | ~0.73-0.78| ~0.59-0.60      | ~0.42-0.49
    4.2 final tuned (train+val)    | ~0.70-0.76| ~0.70-0.72 (NOT FAIR -- fit to it) | ~0.46-0.50

Both models collapse to near-random on test, consistent with task 5.4's
finding (test wards 169/182 have essentially zero within-ward label
variance -- no model, however good, can rank what carries no signal).
But the FAIR val check (4.1's original model) shows real, if weak, above-
chance generalization (~0.59-0.60) specifically on the one ward (170)
task 5.4 identified as having genuine elevation-flood variance. That is
the actual evidence for working.md's core claim -- not the tuned model's
inflated (train-contaminated) val number, and not the test split (which
task 5.4 shows is uninformative either way, not counter-evidence).

The baseline has no comparable AUC at all -- `predict_transition()`
(task 3.6) outputs hard 0/1 rule-based decisions, not a ranked
probability, so there is nothing to compute an AUC over. This is itself
part of the comparison: the GNN can express graded uncertainty a
fixed rule structurally cannot, even where both post-threshold F1 scores
look similar.

Usage:
    python src/models/compare_baseline_vs_gnn.py

Required inputs:
    data/processed/ground_truth/gnn_final_per_transition_report.json (task 5.1)
    data/processed/ground_truth/baseline_final_per_transition_report.json (task 5.2)
    data/processed/ground_truth/sanity_check_comparison_anomalies_report.json (task 5.4)
    data/processed/ground_truth/gnn_model.pt + gnn_feature_normalization_stats.json (task 4.1, for the fair-holdout AUC)

Outputs:
    data/processed/ground_truth/baseline_vs_gnn_comparison.json -- consolidated table + core-claim verdict
    docs/figures/phase5_f1_comparison.png       -- grouped bar: baseline vs tuned-GNN F1, per transition/split
    docs/figures/phase5_auc_fair_holdout.png    -- bar: fair train/val/test AUC, task 4.1's train-only model
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- this script runs from the CLI, not a notebook
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.gnn.build_graph_snapshot_dataset import load_dataset  # noqa: E402
from src.models.gnn.evaluate_final_model import compute_auc  # noqa: E402
from src.models.gnn.train_gnn import (  # noqa: E402
    Phase4TrainingError,
    load_labels_for_attach,
    load_split_masks,
    predict_probabilities,
)
from torch_geometric_temporal.nn.recurrent import A3TGCN  # noqa: E402

GROUND_TRUTH_DIR = REPO_ROOT / "data" / "processed" / "ground_truth"
GNN_TUNED_REPORT_PATH = GROUND_TRUTH_DIR / "gnn_final_per_transition_report.json"
BASELINE_REPORT_PATH = GROUND_TRUTH_DIR / "baseline_final_per_transition_report.json"
ANOMALY_REPORT_PATH = GROUND_TRUTH_DIR / "sanity_check_comparison_anomalies_report.json"
GNN_ORIGINAL_MODEL_PATH = GROUND_TRUTH_DIR / "gnn_model.pt"
GNN_ORIGINAL_STATS_PATH = GROUND_TRUTH_DIR / "gnn_feature_normalization_stats.json"

OUT_REPORT_PATH = GROUND_TRUTH_DIR / "baseline_vs_gnn_comparison.json"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
F1_FIGURE_PATH = FIGURES_DIR / "phase5_f1_comparison.png"
AUC_FIGURE_PATH = FIGURES_DIR / "phase5_auc_fair_holdout.png"

FLOOD_RELEVANT_TRANSITIONS = ["rising->peak", "peak->receding"]
SPLITS = ["train", "val", "test"]


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase4TrainingError(
            f"Required artifact not found: {path}\nThis file is produced by {produced_by}. Run it first."
        )
    return path


# --------------------------------------------------------------------------
# Loading (pure logic given loaded JSON -- unit-tested)
# --------------------------------------------------------------------------

def build_comparison_table(gnn_tuned_report: dict, baseline_report: dict) -> dict:
    """{transition: {split: {baseline_f1, baseline_accuracy, gnn_f1,
    gnn_accuracy, gnn_auc}}} -- gnn_auc here is the TUNED model's (val is
    NOT a fair holdout for it, see module docstring)."""
    table = {}
    for transition in gnn_tuned_report:
        table[transition] = {}
        for split in SPLITS:
            gnn_metrics = gnn_tuned_report[transition].get(split, {})
            baseline_metrics = baseline_report.get(transition, {}).get(split, {})
            table[transition][split] = {
                "baseline_f1": baseline_metrics.get("f1"),
                "baseline_accuracy": baseline_metrics.get("accuracy"),
                "gnn_f1": gnn_metrics.get("f1"),
                "gnn_accuracy": gnn_metrics.get("accuracy"),
                "gnn_auc_tuned_model": gnn_metrics.get("auc"),
            }
    return table


def compute_fair_holdout_auc(device=None) -> dict:
    """AUC for task 4.1's ORIGINAL model (trained on "train" ONLY, so val
    AND test are genuinely held out -- see module docstring). Reuses
    predict_probabilities()/compute_auc(), doesn't reimplement them."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset()
    ds.attach_labels(load_labels_for_attach())
    stats = json.loads(_require_file(GNN_ORIGINAL_STATS_PATH, "src/models/gnn/train_gnn.py (task 4.1)").read_text(encoding="utf-8"))
    feature_stats = (torch.tensor(stats["mean"]), torch.tensor(stats["std"]))

    model = A3TGCN(in_channels=ds.num_features, out_channels=1, periods=1).to(device)
    model.load_state_dict(torch.load(_require_file(GNN_ORIGINAL_MODEL_PATH, "src/models/gnn/train_gnn.py (task 4.1)")))
    model.eval()

    phase_id_to_name = {p["phase_id"]: p["phase_name"] for p in ds.phases}
    probabilities = predict_probabilities(model, ds, feature_stats, phase_id_to_name, device=device)
    split_masks = load_split_masks(ds.node_order)

    result = {}
    for label, (y_true, y_prob) in probabilities.items():
        result[label] = {}
        for split in SPLITS:
            mask = split_masks[split]
            auc = compute_auc(y_true[mask].squeeze(-1).numpy(), y_prob[mask].squeeze(-1).numpy())
            result[label][split] = auc
    return result


# --------------------------------------------------------------------------
# Core-claim verdict (pure logic -- unit-tested)
# --------------------------------------------------------------------------

def evaluate_core_claim(comparison_table: dict, fair_holdout_auc: dict, anomaly_report: dict) -> dict:
    """A structured, honest verdict per flood-relevant transition -- not a
    single yes/no. Cites task 5.4's per-split informativeness so "test is
    inconclusive" is a documented fact, not a hedge."""
    informativeness = {
        phase: data["informativeness_by_split"]
        for phase, data in anomaly_report.get("by_phase", {}).items()
    }
    verdicts = {}
    for transition in [t for t in FLOOD_RELEVANT_TRANSITIONS if t in comparison_table]:
        y_t1_phase = transition.split("->")[1]
        split_info = informativeness.get(y_t1_phase, {})
        baseline_test_f1 = comparison_table[transition]["test"]["baseline_f1"]
        fair_val_auc = fair_holdout_auc.get(transition, {}).get("val")
        fair_test_auc = fair_holdout_auc.get(transition, {}).get("test")
        test_informative_wards = split_info.get("test", {}).get("n_informative_wards", 0)
        val_informative_wards = split_info.get("val", {}).get("n_informative_wards", 0)

        verdict = {
            "baseline_test_f1": baseline_test_f1,
            "gnn_fair_holdout_val_auc": fair_val_auc,
            "gnn_fair_holdout_test_auc": fair_test_auc,
            "val_has_informative_wards": val_informative_wards > 0,
            "test_has_informative_wards": test_informative_wards > 0,
        }
        if fair_val_auc is not None and fair_val_auc > 0.55 and val_informative_wards > 0:
            verdict["val_conclusion"] = (
                "SUPPORTED (weak-to-modest): fair-holdout AUC above chance on the one split "
                "with genuine within-ward label variance."
            )
        else:
            verdict["val_conclusion"] = "NOT SUPPORTED on val."

        if not verdict["test_has_informative_wards"]:
            verdict["test_conclusion"] = (
                "INCONCLUSIVE: task 5.4 found zero informative wards in test -- near-random AUC here "
                "is expected regardless of model quality, not evidence against the GNN."
            )
        elif fair_test_auc is not None and fair_test_auc > 0.55:
            verdict["test_conclusion"] = "SUPPORTED on test."
        else:
            verdict["test_conclusion"] = "NOT SUPPORTED on test."

        verdicts[transition] = verdict
    return verdicts


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_f1_comparison(comparison_table: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(comparison_table), figsize=(5 * len(comparison_table), 4.5), sharey=True)
    if len(comparison_table) == 1:
        axes = [axes]
    x = np.arange(len(SPLITS))
    width = 0.35
    for ax, (transition, per_split) in zip(axes, comparison_table.items()):
        baseline_f1 = [per_split[s]["baseline_f1"] or 0 for s in SPLITS]
        gnn_f1 = [per_split[s]["gnn_f1"] or 0 for s in SPLITS]
        ax.bar(x - width / 2, baseline_f1, width, label="Baseline", color="#888888")
        ax.bar(x + width / 2, gnn_f1, width, label="GNN (tuned)", color="#2c7fb8")
        ax.set_xticks(x)
        ax.set_xticklabels(SPLITS)
        ax.set_title(transition)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1")
    axes[0].legend()
    fig.suptitle("Baseline vs. tuned GNN -- F1 per transition per split")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fair_holdout_auc(fair_holdout_auc: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(SPLITS))
    width = 0.35
    for i, transition in enumerate(FLOOD_RELEVANT_TRANSITIONS):
        aucs = [fair_holdout_auc[transition][s] or 0 for s in SPLITS]
        offset = (i - 0.5) * width
        ax.bar(x + offset, aucs, width, label=transition)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="random chance (AUC=0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(SPLITS)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0, 1.0)
    ax.set_title("Task 4.1's train-only model -- FAIR holdout AUC (val/test genuinely unseen)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    print("Loading task 5.1/5.2/5.4 reports ...")
    try:
        gnn_tuned_report = json.loads(_require_file(GNN_TUNED_REPORT_PATH, "src/models/gnn/evaluate_final_model.py (task 5.1)").read_text(encoding="utf-8"))
        baseline_report = json.loads(_require_file(BASELINE_REPORT_PATH, "src/models/baseline/evaluate_per_transition.py (task 5.2)").read_text(encoding="utf-8"))
        anomaly_report = json.loads(_require_file(ANOMALY_REPORT_PATH, "src/ground_truth/sanity_check_comparison_anomalies.py (task 5.4)").read_text(encoding="utf-8"))
    except Phase4TrainingError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    comparison_table = build_comparison_table(gnn_tuned_report, baseline_report)

    print("\nComputing FAIR holdout AUC for task 4.1's original (train-only) model ...")
    fair_holdout_auc = compute_fair_holdout_auc()
    for transition, per_split in fair_holdout_auc.items():
        print(f"  {transition}: " + ", ".join(f"{s}={a}" for s, a in per_split.items()))

    print("\nEvaluating working.md SS1.6's core claim ...")
    verdicts = evaluate_core_claim(comparison_table, fair_holdout_auc, anomaly_report)
    for transition, v in verdicts.items():
        print(f"  {transition}:")
        print(f"    val:  {v['val_conclusion']}")
        print(f"    test: {v['test_conclusion']}")

    print("\nGenerating plots ...")
    plot_f1_comparison(comparison_table, F1_FIGURE_PATH)
    plot_fair_holdout_auc(fair_holdout_auc, AUC_FIGURE_PATH)
    print(f"  -> {F1_FIGURE_PATH}")
    print(f"  -> {AUC_FIGURE_PATH}")

    report = {
        "comparison_table": comparison_table,
        "fair_holdout_auc_train_only_model": fair_holdout_auc,
        "core_claim_verdicts": verdicts,
    }
    OUT_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved -> {OUT_REPORT_PATH}")

    print("\nDone. Task 5.5 (Objective 2 evaluation) and 5.6 (checkpoint meeting) are next.")


if __name__ == "__main__":
    main()
