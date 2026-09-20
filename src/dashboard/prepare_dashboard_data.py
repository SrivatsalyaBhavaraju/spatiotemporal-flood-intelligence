"""
Task 6.1 support script -- consolidates everything the dashboard needs
into fast-loading, pre-joined files, so the Streamlit app (`app.py`)
never has to load a model or run inference at request time. Reuses
already-persisted artifacts from tasks 2.2/2.3/3.4/3.7/4.3/5.1 --
nothing is recomputed except the GNN's forward pass (loading the
already-trained task 5.1 model, not retraining it).

Per-phase segment layer: for each of the 4 phases, one GeoJSON with
segment_id, geometry, ward_no, split, elevation_m, true flood_label,
baseline_pred, gnn_pred, gnn_prob. `pre_event` has no baseline/GNN
prediction (it's never a y_t1 target, task 2.7's schema) -- both are
fixed at 0/0.0 and the app labels this explicitly rather than implying a
model ran there.

Usage:
    python src/dashboard/prepare_dashboard_data.py

Outputs (data/processed/dashboard/):
    segments_<phase_name>.geojson  (4 files, one per phase)
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.gnn.build_graph_snapshot_dataset import load_dataset  # noqa: E402
from src.models.gnn.train_gnn import load_labels_for_attach, predict_probabilities  # noqa: E402
from torch_geometric_temporal.nn.recurrent import A3TGCN  # noqa: E402

MODEL_INPUT_DIR = REPO_ROOT / "data" / "processed" / "model_input"
SCHEMA_PATH = MODEL_INPUT_DIR / "schema.json"
SEGMENT_SPLITS_PATH = MODEL_INPUT_DIR / "segment_splits.csv"
LINE_GRAPH_NODES = REPO_ROOT / "data" / "processed" / "graph" / "line_graph_nodes.geojson"
STATIC_FEATURES = REPO_ROOT / "data" / "processed" / "graph" / "static_features.geojson"
GROUND_TRUTH_DIR = REPO_ROOT / "data" / "processed" / "ground_truth"
FUSED_LABELS_PATH = GROUND_TRUTH_DIR / "fused_flood_labels.csv"
BASELINE_PREDICTIONS_PATH = GROUND_TRUTH_DIR / "baseline_predictions.csv"
GNN_MODEL_PATH = GROUND_TRUTH_DIR / "gnn_final_model.pt"
GNN_STATS_PATH = GROUND_TRUTH_DIR / "gnn_final_model_feature_stats.json"
TUNING_REPORT_PATH = GROUND_TRUTH_DIR / "hyperparameter_tuning_report.json"

OUT_DIR = REPO_ROOT / "data" / "processed" / "dashboard"


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}\nProduced by {produced_by}. Run it first.")
    return path


def load_gnn_predictions() -> dict:
    """{transition_label: (y_true[N,1], y_prob[N,1])} for all 3 usable
    transitions, from the already-trained task 5.1 model -- no retraining."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset()
    ds.attach_labels(load_labels_for_attach())
    stats = json.loads(_require_file(GNN_STATS_PATH, "src/models/gnn/evaluate_final_model.py (task 5.1)").read_text(encoding="utf-8"))
    feature_stats = (torch.tensor(stats["feature_mean"]), torch.tensor(stats["feature_std"]))
    model = A3TGCN(in_channels=ds.num_features, out_channels=1, periods=1).to(device)
    model.load_state_dict(torch.load(_require_file(GNN_MODEL_PATH, "src/models/gnn/evaluate_final_model.py (task 5.1)")))
    model.eval()
    phase_id_to_name = {p["phase_id"]: p["phase_name"] for p in ds.phases}
    return predict_probabilities(model, ds, feature_stats, phase_id_to_name, device=device), ds.node_order


def main():
    print("Loading real committed geometry/features/labels/predictions ...")
    schema = json.loads(_require_file(SCHEMA_PATH, "task 2.7").read_text(encoding="utf-8"))
    phase_names = [p["phase_name"] for p in schema["phases"]]
    line_graph = gpd.read_file(_require_file(LINE_GRAPH_NODES, "task 2.2"))[["segment_id", "geometry"]]
    static_features = gpd.read_file(_require_file(STATIC_FEATURES, "task 2.3"))[["segment_id", "elevation_m"]]
    splits = pd.read_csv(_require_file(SEGMENT_SPLITS_PATH, "task 3.7"))
    fused_labels = pd.read_csv(_require_file(FUSED_LABELS_PATH, "task 3.4/4.4"))
    fused_labels["phase_name"] = fused_labels["phase_id"].map({p["phase_id"]: p["phase_name"] for p in schema["phases"]})
    baseline_preds = pd.read_csv(_require_file(BASELINE_PREDICTIONS_PATH, "task 3.6/4.3"))
    tuning_report = json.loads(_require_file(TUNING_REPORT_PATH, "task 4.2").read_text(encoding="utf-8"))
    threshold = tuning_report["selected_threshold"]

    print(f"\nLoading task 5.1's trained GNN model (threshold={threshold}, not retrained) ...")
    gnn_probs, node_order = load_gnn_predictions()

    phase_id_to_name = {p["phase_id"]: p["phase_name"] for p in schema["phases"]}
    name_to_phase_id = {v: k for k, v in phase_id_to_name.items()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for phase_name in phase_names:
        print(f"\nBuilding layer for phase: {phase_name}")
        phase_id = name_to_phase_id[phase_name]
        truth = fused_labels[fused_labels["phase_name"] == phase_name][["segment_id", "flood_label"]].rename(
            columns={"flood_label": "true_flood"}
        )

        baseline_match = baseline_preds[baseline_preds["y_t1_phase_id"] == phase_id]
        if len(baseline_match) > 0:
            baseline_col = baseline_match[["segment_id", "y_pred"]].rename(columns={"y_pred": "baseline_pred"})
        else:
            baseline_col = pd.DataFrame({"segment_id": node_order, "baseline_pred": 0})

        gnn_col = pd.DataFrame({"segment_id": node_order, "gnn_prob": 0.0, "gnn_pred": 0})
        # Match this phase as the y_t1 of whichever transition targets it (pre_event is never a
        # target -- gnn_col stays at the 0/0.0 default above, and the app labels this explicitly).
        for transition in schema["y_t1_contract"]["usable_transitions"]:
            if transition["y_t1_phase_id"] == phase_id:
                trans_label = f"{phase_id_to_name[transition['x_t_phase_id']]}->{phase_id_to_name[transition['y_t1_phase_id']]}"
                _, y_prob_t = gnn_probs[trans_label]
                gnn_col = pd.DataFrame({
                    "segment_id": node_order,
                    "gnn_prob": y_prob_t.squeeze(-1).numpy(),
                    "gnn_pred": (y_prob_t.squeeze(-1).numpy() >= threshold).astype(int),
                })
                break

        merged = (
            line_graph.merge(static_features, on="segment_id", how="left")
            .merge(splits[["segment_id", "ward_no", "split"]], on="segment_id", how="left")
            .merge(truth, on="segment_id", how="left")
            .merge(baseline_col, on="segment_id", how="left")
            .merge(gnn_col, on="segment_id", how="left")
        )
        merged["true_flood"] = merged["true_flood"].fillna(0).astype(int)
        merged["baseline_pred"] = merged["baseline_pred"].fillna(0).astype(int)
        merged["gnn_pred"] = merged["gnn_pred"].fillna(0).astype(int)
        merged["gnn_prob"] = merged["gnn_prob"].fillna(0.0).round(4)
        merged["elevation_m"] = merged["elevation_m"].round(2)

        out_path = OUT_DIR / f"segments_{phase_name}.geojson"
        merged.to_file(out_path, driver="GeoJSON")
        print(f"  -> {out_path} ({len(merged)} segments, {out_path.stat().st_size / 1e6:.1f} MB)")

    print("\nDone. Dashboard data ready in data/processed/dashboard/.")


if __name__ == "__main__":
    main()
