"""
Task 3.6 -- rule-based baseline propagation model (working.md SS1.6): the
static, non-learned counterpart the GNN (task 4.1) has to beat. Two rules,
combined with OR, exactly as working.md's own diagram specifies:

    Rule A (direct rainfall trigger):
        segment floods if the CURRENT phase's rainfall intensity exceeds
        RAINFALL_INTENSITY_THRESHOLD_MM_PER_DAY.
    Rule B (neighbor cascading):
        segment floods if it has at least one graph-adjacent neighbor that
        IS flooded (in the current phase, per task 3.4's real fused
        ground truth -- not the baseline's own prior prediction, see
        "Why ground truth as the neighbor-state input" below) AND its own
        elevation is at/below ELEVATION_THRESHOLD_M.

Why rainfall INTENSITY, not the rainfall_t already in task 2.7's X_t:
checked the real numbers first, not assumed -- 2.7's rainfall_t is a raw
phase-WINDOW total, and developing.md's task 2.5 notes already flag that
pre_event's total (1055.7mm over its 20-day window) exceeds peak's
(410.1mm over 2 days) purely because the windows are wildly different
lengths, not because pre_event was wetter. Using that raw total for Rule A
would make the "dry baseline" phase spuriously trigger the flood rule.
Recomputed from task 2.5's own n_hours column instead: intensity (mm/day)
is pre_event=50.3, rising=15.6, peak=205.1, receding=10.5 -- now peak is
unambiguously the outlier (4x the next-highest), which a raw-total
comparison could not show. RAINFALL_INTENSITY_THRESHOLD_MM_PER_DAY=100 is
grounded in IMD's own rainfall-intensity classification (Heavy: 64.5-119.5
mm/day, Very Heavy: 119.6-244.4mm/day) -- peak's 205.1 falls solidly in
"Very Heavy," every other phase falls below "Heavy" -- not a value fit to
produce a particular-looking result.

ELEVATION_THRESHOLD_M=8.5 is the real median segment elevation (task 2.3's
own validation report) -- a data-driven default, disclosed as such (no
literature reference grounds a specific "flood-prone elevation" for this
AOI the way IMD's rainfall bands do for Rule A).

Adjacency for Rule B is treated as UNDIRECTED, unlike task 2.2/2.7's
edge_index (deliberately directed there, respecting OSM one-way traversal
-- see that module's docstring): physical floodwater does not respect
traffic direction, so "neighbor" here means graph-adjacent either way.
This is a deliberate choice specific to this hand-written physical rule,
not a change to the GNN's own edge_index.

Why ground truth as the neighbor-state input, not the baseline's own prior
prediction: working.md SS1.6 frames evaluation as "F1/accuracy PER PHASE
TRANSITION," matching task 2.8's own transition_pairs() design (independent
(X_t, Y_t+1) pairs, not a chained rollout). Using real Y_t for Rule B's
neighbor check evaluates each transition on equal footing with however
the GNN (task 4.1) gets evaluated, rather than compounding the baseline's
own errors across phases the way an autoregressive simulation would.

Usage:
    python src/models/baseline/rule_based_propagation.py

Required inputs:
    data/processed/model_input/node_order.json        (task 2.7)
    data/processed/model_input/edge_index.npy           (task 2.7)
    data/processed/model_input/schema.json                (task 2.7)
    data/processed/graph/static_features.geojson           (task 2.3)
    data/processed/features/rainfall_phase_features.csv      (task 2.5)
    data/processed/ground_truth/fused_flood_labels.csv        (task 3.4)

Outputs (data/processed/ground_truth/):
    baseline_predictions.csv               -- segment_id, phase_id, y_true, y_pred
    baseline_evaluation_report.json         -- per-transition + overall metrics
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_INPUT_DIR = REPO_ROOT / "data" / "processed" / "model_input"
NODE_ORDER_PATH = MODEL_INPUT_DIR / "node_order.json"
EDGE_INDEX_PATH = MODEL_INPUT_DIR / "edge_index.npy"
SCHEMA_PATH = MODEL_INPUT_DIR / "schema.json"
STATIC_FEATURES = REPO_ROOT / "data" / "processed" / "graph" / "static_features.geojson"
RAINFALL_PHASE_FEATURES = REPO_ROOT / "data" / "processed" / "features" / "rainfall_phase_features.csv"
FUSED_LABELS_PATH = REPO_ROOT / "data" / "processed" / "ground_truth" / "fused_flood_labels.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "ground_truth"

RAINFALL_INTENSITY_THRESHOLD_MM_PER_DAY = 100.0  # IMD "Heavy" rainfall boundary -- see module docstring
ELEVATION_THRESHOLD_M = 8.5  # real median segment elevation (task 2.3) -- data-driven, disclosed


class Phase3BaselineError(FileNotFoundError):
    """Raised when a required task 2.3/2.5/2.7/3.4 artifact is missing.
    Matches the *ArtifactError convention used across src/graph,
    src/ground_truth, src/nlp, src/models/gnn."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase3BaselineError(
            f"Required artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first."
        )
    return path


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_rainfall_intensity(path: Path = RAINFALL_PHASE_FEATURES) -> dict:
    """{phase_id: rainfall intensity in mm/day} -- see module docstring for
    why this is recomputed from n_hours rather than reusing the raw
    phase-window rainfall_t total already in task 2.7's X_t."""
    _require_file(path, "src/features/build_rainfall_phase_features.py (task 2.5)")
    df = pd.read_csv(path)
    df["intensity_mm_per_day"] = df["rainfall_t"] / df["n_hours"] * 24
    return dict(zip(df["phase_id"], df["intensity_mm_per_day"]))


def load_elevation(path: Path = STATIC_FEATURES) -> pd.Series:
    _require_file(path, "src/graph/build_static_features.py (task 2.3)")
    import geopandas as gpd
    gdf = gpd.read_file(path)
    return gdf.set_index("segment_id")["elevation_m"]


def load_ground_truth(path: Path = FUSED_LABELS_PATH) -> pd.DataFrame:
    _require_file(path, "src/ground_truth/fuse_flood_labels.py (task 3.4)")
    return pd.read_csv(path)


def build_undirected_adjacency(edge_index: np.ndarray, node_order: list) -> dict:
    """{segment_id: set(neighbor segment_ids)} -- undirected (see module
    docstring for why this differs from task 2.2/2.7's directed
    edge_index)."""
    adjacency = {seg_id: set() for seg_id in node_order}
    for src_pos, dst_pos in zip(edge_index[0], edge_index[1]):
        src_id, dst_id = node_order[src_pos], node_order[dst_pos]
        adjacency[src_id].add(dst_id)
        adjacency[dst_id].add(src_id)
    return adjacency


# --------------------------------------------------------------------------
# The baseline rule itself (pure logic -- unit-tested)
# --------------------------------------------------------------------------

def predict_transition(
    node_order: list, adjacency: dict, elevation: pd.Series, rainfall_intensity_mm_per_day: float,
    y_t: dict, rainfall_threshold: float = RAINFALL_INTENSITY_THRESHOLD_MM_PER_DAY,
    elevation_threshold: float = ELEVATION_THRESHOLD_M,
) -> dict:
    """One transition's prediction: {segment_id: 0 or 1}.
    y_t: {segment_id: 0 or 1} -- the CURRENT phase's real ground-truth
    label, used for Rule B's neighbor check (see module docstring).
    """
    rainfall_trigger = rainfall_intensity_mm_per_day > rainfall_threshold
    predictions = {}
    for seg_id in node_order:
        if rainfall_trigger:
            predictions[seg_id] = 1
            continue
        low_elevation = elevation.get(seg_id, np.inf) <= elevation_threshold
        neighbor_flooded = any(y_t.get(n, 0) == 1 for n in adjacency.get(seg_id, ()))
        predictions[seg_id] = 1 if (low_elevation and neighbor_flooded) else 0
    return predictions


# --------------------------------------------------------------------------
# Evaluation (pure logic -- unit-tested)
# --------------------------------------------------------------------------

def binary_classification_metrics(y_true: list, y_pred: list) -> dict:
    """Precision/recall/F1/accuracy from a hand-rolled confusion matrix --
    trivial for binary labels, no need for a new dependency (task 3.7's
    training/eval harness is where general metrics utilities belong)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "accuracy": round(accuracy, 4),
    }


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    print(f"Loading task 2.7 model input <- {MODEL_INPUT_DIR}")
    try:
        node_order = json.loads(_require_file(NODE_ORDER_PATH, "task 2.7").read_text(encoding="utf-8"))
        edge_index = np.load(_require_file(EDGE_INDEX_PATH, "task 2.7"))
        schema = json.loads(_require_file(SCHEMA_PATH, "task 2.7").read_text(encoding="utf-8"))
        elevation = load_elevation()
        rainfall_intensity = load_rainfall_intensity()
        ground_truth = load_ground_truth()
    except Phase3BaselineError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> {len(node_order)} segments, {edge_index.shape[1]} directed edges")

    print("\nBuilding undirected adjacency (see module docstring for why) ...")
    adjacency = build_undirected_adjacency(edge_index, node_order)

    phase_id_to_name = {p["phase_id"]: p["phase_name"] for p in schema["phases"]}
    gt_by_phase = {
        phase_id: dict(zip(group["segment_id"], group["flood_label"]))
        for phase_id, group in ground_truth.groupby("phase_id")
    }

    print(f"\nRunning baseline over {len(schema['y_t1_contract']['usable_transitions'])} usable transitions ...")
    all_predictions = []
    per_transition_metrics = {}
    for transition in schema["y_t1_contract"]["usable_transitions"]:
        x_t_phase_id, y_t1_phase_id = transition["x_t_phase_id"], transition["y_t1_phase_id"]
        y_pred = predict_transition(
            node_order, adjacency, elevation, rainfall_intensity[x_t_phase_id], gt_by_phase[x_t_phase_id]
        )
        y_true = gt_by_phase[y_t1_phase_id]

        y_true_list = [y_true[s] for s in node_order]
        y_pred_list = [y_pred[s] for s in node_order]
        metrics = binary_classification_metrics(y_true_list, y_pred_list)
        label = f"{phase_id_to_name[x_t_phase_id]}->{phase_id_to_name[y_t1_phase_id]}"
        per_transition_metrics[label] = metrics
        print(f"  {label}: F1={metrics['f1']}  accuracy={metrics['accuracy']}  "
              f"precision={metrics['precision']}  recall={metrics['recall']}")

        for seg_id in node_order:
            all_predictions.append({
                "segment_id": seg_id, "x_t_phase_id": x_t_phase_id, "y_t1_phase_id": y_t1_phase_id,
                "y_true": y_true[seg_id], "y_pred": y_pred[seg_id],
            })

    predictions_df = pd.DataFrame(all_predictions)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = OUT_DIR / "baseline_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nSaved predictions -> {predictions_path}")

    overall_metrics = binary_classification_metrics(predictions_df["y_true"], predictions_df["y_pred"])
    report = {
        "rainfall_intensity_threshold_mm_per_day": RAINFALL_INTENSITY_THRESHOLD_MM_PER_DAY,
        "elevation_threshold_m": ELEVATION_THRESHOLD_M,
        "rainfall_intensity_by_phase_mm_per_day": rainfall_intensity,
        "per_transition_metrics": per_transition_metrics,
        "overall_metrics": overall_metrics,
    }
    report_path = OUT_DIR / "baseline_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Saved evaluation report -> {report_path}")

    print("\n=== Validation summary ===")
    print(json.dumps(report, indent=2, default=str))

    print("\nDone. Next: task 4.1 (GNN training) evaluated against the same transitions/metrics")
    print("is the comparison working.md SS1.6 calls the core experiment of this whole project.")


if __name__ == "__main__":
    main()
