"""
Task 6.1 -- data loading + pure presentation logic for the Streamlit
dashboard (`app.py`). Kept separate from `app.py` so the color/parsing/
KPI logic is unit-testable without a Streamlit runtime.

All data is real, pre-joined by `prepare_dashboard_data.py` (per-phase
segment layers) or read directly from already-committed/generated task
outputs (comparison reports, Objective 2 pipeline output). Nothing here
recomputes a model or fabricates a number.
"""
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DATA_DIR = REPO_ROOT / "data" / "processed" / "dashboard"
GROUND_TRUTH_DIR = REPO_ROOT / "data" / "processed" / "ground_truth"
NLP_DIR = REPO_ROOT / "data" / "processed" / "nlp"
WARDS_PATH = REPO_ROOT / "data" / "raw" / "wards" / "study_wards.geojson"

PHASES = ["pre_event", "rising", "peak", "receding"]
PHASE_LABELS = {"pre_event": "Pre-Event", "rising": "Rising", "peak": "Peak", "receding": "Receding"}

# Validated palette (dataviz skill) -- status roles, fixed, never themed.
COLOR_DRY = "#2a78d6"       # categorical slot 1 (blue) -- reads as "baseline/normal state", not alarming
COLOR_FLOODED = "#d03b3b"   # status: critical
COLOR_BASELINE = "#2a78d6"  # categorical slot 1
COLOR_GNN = "#eb6834"       # categorical slot 2
COLOR_GOOD = "#0ca30c"
COLOR_WARNING = "#fab219"
COLOR_MUTED = "#898781"


def flood_fill_color(value: int) -> str:
    """0/1 flood value -> hex fill color (status palette)."""
    return COLOR_FLOODED if int(value) == 1 else COLOR_DRY


def load_phase_layer(phase_name: str) -> gpd.GeoDataFrame:
    path = DASHBOARD_DATA_DIR / f"segments_{phase_name}.geojson"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run src/dashboard/prepare_dashboard_data.py first.")
    return gpd.read_file(path)


def load_wards() -> gpd.GeoDataFrame:
    return gpd.read_file(WARDS_PATH)


def parse_resolved_locations(raw: str) -> list:
    """Objective 2 pipeline output stores `resolved_locations` as a JSON
    string (task 4.5's CSV serialization) -- parse back to a list of
    {"place", "lon", "lat", "score"} dicts. Empty/invalid input -> []."""
    if not raw or pd.isna(raw):
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def load_distress_markers() -> pd.DataFrame:
    """Distress posts that ALSO resolved to >=1 coordinate -- working.md
    SS2.4's actual Objective 2 deliverable (task 4.5's own definition),
    exploded one row per resolved location so each gets its own marker."""
    path = NLP_DIR / "objective2_pipeline_output.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run src/nlp/run_objective2_pipeline.py first.")
    df = pd.read_csv(path)
    df["resolved_locations"] = df["resolved_locations"].apply(parse_resolved_locations)
    distress = df[(df["distress_label"] == "distress") & (df["resolved_locations"].apply(len) > 0)]
    rows = []
    for _, row in distress.iterrows():
        for loc in row["resolved_locations"]:
            rows.append({
                "text": row["text"], "probability": row["distress_probability"],
                "place": loc["place"], "lon": loc["lon"], "lat": loc["lat"],
            })
    return pd.DataFrame(rows)


def load_json_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def compute_overview_kpis() -> dict:
    """Real numbers pulled from already-generated reports (tasks 3.1-5.5)
    for the Overview tab's hero stat tiles -- nothing computed fresh."""
    comparison = load_json_report(GROUND_TRUTH_DIR / "baseline_vs_gnn_comparison.json")
    anomaly = load_json_report(GROUND_TRUTH_DIR / "sanity_check_comparison_anomalies_report.json")
    obj2 = load_json_report(NLP_DIR / "objective2_precision_recall_report.json")
    fused = pd.read_csv(GROUND_TRUTH_DIR / "fused_flood_labels.csv")
    peak_pct = round(100 * fused[fused["phase_name"] == "peak"]["flood_label"].mean(), 1)

    return {
        "n_segments": 17195,
        "n_wards": 16,
        "peak_flooded_pct": peak_pct,
        "gnn_val_auc": comparison["fair_holdout_auc_train_only_model"]["rising->peak"]["val"],
        "gnn_test_auc": comparison["fair_holdout_auc_train_only_model"]["rising->peak"]["test"],
        "gnn_tuned_test_f1": comparison["comparison_table"]["rising->peak"]["test"]["gnn_f1"],
        "baseline_test_f1": comparison["comparison_table"]["rising->peak"]["test"]["baseline_f1"],
        "n_informative_wards_test": anomaly["by_phase"]["peak"]["informativeness_by_split"]["test"]["n_informative_wards"],
        "classifier_f1": obj2["classification_precision_recall"]["f1"],
        "geoparse_precision": obj2["geoparsing_precision_new"]["precision"],
        "geoparse_recall": obj2["geoparsing_recall_task_2_10"]["recall"],
    }
