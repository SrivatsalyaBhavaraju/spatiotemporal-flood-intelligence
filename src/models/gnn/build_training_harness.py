"""
Task 3.7 -- training/eval harness skeleton: a phase-based train/val/test
split plus reusable metrics, ready for task 4.1's GNN training. No spec
for this exists in working.md (checked directly) -- the design below is
this task's own judgment call, documented so it can be revisited.

Why a WARD-level (spatial) split, not a temporal (transition) one: there
are only 3 usable transitions total (task 2.7's schema.json), and each
transition's dynamic features (rainfall_t) are broadcast identically to
every segment within a phase (task 2.6) -- holding out a whole transition
for testing would remove an entire feature-context from training, which
is a bigger loss for a dataset this small than it looks. A ward-level
split instead uses ALL 3 transitions for training AND evaluation, holding
out geographically contiguous groups of segments -- the standard mitigation
for spatial autocorrelation leakage that a per-node random split would
have (neighboring segments are highly correlated; a purely random split
would put many of a test segment's own graph neighbors in the training
set). "Phase-based" in this task's name refers to how results are
REPORTED, not how segments are split: task 5.1/5.2 explicitly ask for
F1/accuracy PER PHASE TRANSITION, so this harness evaluates each of the
3 transitions separately, using the SAME fixed ward split throughout.

Split target: ~70/15/15 by SEGMENT COUNT (not ward count, since ward
sizes range 639-1,830 segments) -- a greedy assignment (shuffle wards with
a fixed seed, assign each to whichever bucket is currently furthest below
its target share) gets close without needing an optimizer for just 16
wards. ~2.7% of segments (465/17,195) don't fall strictly inside any ward
polygon (boundary-adjacent, same class of edge case build_gazetteer.py's
README already documents for gazetteer points) -- assigned to their
nearest ward by centroid distance rather than left out.

Usage:
    python src/models/gnn/build_training_harness.py

Required inputs:
    data/processed/model_input/node_order.json        (task 2.7)
    data/processed/model_input/schema.json              (task 2.7)
    data/processed/graph/line_graph_nodes.geojson        (task 2.2)
    data/raw/wards/study_wards.geojson                    (task 1.3)
    data/processed/ground_truth/fused_flood_labels.csv     (task 3.4)
    data/processed/ground_truth/baseline_predictions.csv    (task 3.6, optional
        -- used only to smoke-test this harness against a real model's
        real predictions, not required to build the split itself)

Outputs (data/processed/model_input/):
    segment_splits.csv               -- segment_id, ward_no, split (train/val/test)
    training_harness_validation_report.json
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline.rule_based_propagation import binary_classification_metrics  # noqa: E402

MODEL_INPUT_DIR = REPO_ROOT / "data" / "processed" / "model_input"
NODE_ORDER_PATH = MODEL_INPUT_DIR / "node_order.json"
SCHEMA_PATH = MODEL_INPUT_DIR / "schema.json"
LINE_GRAPH_NODES = REPO_ROOT / "data" / "processed" / "graph" / "line_graph_nodes.geojson"
WARDS_PATH = REPO_ROOT / "data" / "raw" / "wards" / "study_wards.geojson"
FUSED_LABELS_PATH = REPO_ROOT / "data" / "processed" / "ground_truth" / "fused_flood_labels.csv"
BASELINE_PREDICTIONS_PATH = REPO_ROOT / "data" / "processed" / "ground_truth" / "baseline_predictions.csv"

OUT_SPLITS_PATH = MODEL_INPUT_DIR / "segment_splits.csv"
REPORT_PATH = MODEL_INPUT_DIR / "training_harness_validation_report.json"

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42


class Phase3HarnessError(FileNotFoundError):
    """Raised when a required task 1.3/2.2/2.7/3.4 artifact is missing.
    Matches the *ArtifactError convention used across src/graph,
    src/ground_truth, src/nlp, src/models/gnn."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase3HarnessError(
            f"Required artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first."
        )
    return path


# --------------------------------------------------------------------------
# Ward assignment (real geospatial join -- validated by a real run)
# --------------------------------------------------------------------------

def assign_wards_to_segments(line_graph_nodes: gpd.GeoDataFrame, wards: gpd.GeoDataFrame) -> tuple:
    """(ward_series, n_reassigned): `ward_series` is {segment_id: ward_no}.
    Segments whose representative point doesn't fall strictly inside any
    ward polygon (boundary-adjacent, ~2.7% of the graph -- see module
    docstring) are assigned to their nearest ward by centroid distance
    instead of left unassigned; `n_reassigned` counts those.
    """
    pts_gdf = gpd.GeoDataFrame(
        {"segment_id": line_graph_nodes["segment_id"]},
        geometry=line_graph_nodes.geometry.representative_point(), crs=line_graph_nodes.crs,
    )
    joined = gpd.sjoin(pts_gdf, wards[["Ward_No", "geometry"]], how="left", predicate="within")
    joined = joined.drop_duplicates(subset="segment_id", keep="first").set_index("segment_id")["Ward_No"]

    unmatched = joined[joined.isna()].index
    if len(unmatched):
        ward_centroids = wards.set_index("Ward_No").geometry.centroid
        unmatched_pts = pts_gdf.set_index("segment_id").loc[unmatched].geometry
        for seg_id, pt in unmatched_pts.items():
            nearest_ward = ward_centroids.distance(pt).idxmin()
            joined[seg_id] = nearest_ward

    return joined.astype(int), len(unmatched)


# --------------------------------------------------------------------------
# Ward-level split (pure logic -- unit-tested)
# --------------------------------------------------------------------------

def build_ward_split(ward_sizes: dict, fractions: dict = SPLIT_FRACTIONS, seed: int = SPLIT_SEED) -> dict:
    """{ward_no: split_name}. Shuffles wards (fixed seed) then greedily
    assigns each to whichever bucket is currently furthest below its
    target share of TOTAL segment count -- balances by data volume, not
    just ward count, without needing a real optimizer for ~16 wards.
    """
    total = sum(ward_sizes.values())
    targets = {name: frac * total for name, frac in fractions.items()}
    current = {name: 0 for name in fractions}

    rng = np.random.default_rng(seed)
    ward_order = list(ward_sizes.keys())
    rng.shuffle(ward_order)

    assignment = {}
    for ward_no in ward_order:
        deficit = {name: targets[name] - current[name] for name in fractions}
        chosen = max(deficit, key=deficit.get)
        assignment[ward_no] = chosen
        current[chosen] += ward_sizes[ward_no]
    return assignment


def assign_segment_splits(segment_ward: pd.Series, ward_split: dict) -> pd.Series:
    return segment_ward.map(ward_split)


# --------------------------------------------------------------------------
# Metrics (reused from task 3.6, not reimplemented)
# --------------------------------------------------------------------------

def evaluate_split(y_true: pd.Series, y_pred: pd.Series, split_labels: pd.Series) -> dict:
    """binary_classification_metrics() (task 3.6) computed separately for
    each split ("train"/"val"/"test") over the same (y_true, y_pred)
    pair."""
    results = {}
    for split_name in ["train", "val", "test"]:
        mask = split_labels == split_name
        if mask.sum() == 0:
            continue
        results[split_name] = binary_classification_metrics(y_true[mask].tolist(), y_pred[mask].tolist())
    return results


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    print(f"Loading task 2.2/1.3 geometries <- {LINE_GRAPH_NODES}, {WARDS_PATH}")
    try:
        node_order = json.loads(_require_file(NODE_ORDER_PATH, "task 2.7").read_text(encoding="utf-8"))
        line_graph_nodes = gpd.read_file(_require_file(LINE_GRAPH_NODES, "task 2.2"))
        wards = gpd.read_file(_require_file(WARDS_PATH, "task 1.3"))
    except Phase3HarnessError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> {len(node_order)} segments, {len(wards)} wards")

    print("\nAssigning each segment to its ward (nearest-centroid fallback for boundary cases) ...")
    segment_ward, n_reassigned = assign_wards_to_segments(line_graph_nodes, wards)
    print(f"  -> {len(segment_ward)}/{len(node_order)} segments assigned ({n_reassigned} via nearest-ward fallback)")

    ward_sizes = segment_ward.value_counts().to_dict()
    ward_split = build_ward_split(ward_sizes)
    segment_split = assign_segment_splits(segment_ward, ward_split)
    print(f"\nWard-level split (target {SPLIT_FRACTIONS}): {ward_split}")

    splits_df = pd.DataFrame({
        "segment_id": segment_ward.index, "ward_no": segment_ward.values, "split": segment_split.values,
    })
    MODEL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(OUT_SPLITS_PATH, index=False)
    print(f"Saved segment splits -> {OUT_SPLITS_PATH}")

    split_counts = splits_df["split"].value_counts()
    split_pct = (100 * split_counts / len(splits_df)).round(2)
    report = {
        "split_fractions_target": SPLIT_FRACTIONS,
        "split_seed": SPLIT_SEED,
        "ward_assignment": {str(k): v for k, v in ward_split.items()},
        "achieved_segment_counts": split_counts.to_dict(),
        "achieved_segment_pct": split_pct.to_dict(),
        "unmatched_segments_reassigned_by_nearest_ward": n_reassigned,
    }

    print("\nSmoke-testing the harness against task 3.6's real baseline predictions ...")
    if BASELINE_PREDICTIONS_PATH.exists() and FUSED_LABELS_PATH.exists():
        baseline_preds = pd.read_csv(BASELINE_PREDICTIONS_PATH)
        schema = json.loads(_require_file(SCHEMA_PATH, "task 2.7").read_text(encoding="utf-8"))
        phase_id_to_name = {p["phase_id"]: p["phase_name"] for p in schema["phases"]}

        per_transition = {}
        for (x_t, y_t1), group in baseline_preds.groupby(["x_t_phase_id", "y_t1_phase_id"]):
            group = group.merge(splits_df[["segment_id", "split"]], on="segment_id", how="left")
            label = f"{phase_id_to_name[x_t]}->{phase_id_to_name[y_t1]}"
            per_transition[label] = evaluate_split(group["y_true"], group["y_pred"], group["split"])
            print(f"  {label}: " + ", ".join(f"{s}_f1={m['f1']}" for s, m in per_transition[label].items()))
        report["baseline_metrics_by_split_and_transition"] = per_transition
    else:
        print("  (skipped -- task 3.6's baseline_predictions.csv not found; harness itself is still built)")
        report["baseline_metrics_by_split_and_transition"] = None

    report_path = REPORT_PATH
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved validation report -> {report_path}")

    print("\n=== Validation summary ===")
    print(json.dumps({k: v for k, v in report.items() if k != "ward_assignment"}, indent=2, default=str))

    print("\nDone. Next: task 4.1 (GNN training) trains only on segment_splits.csv's 'train' rows,")
    print("tunes on 'val', and reports final per-transition metrics on 'test' -- same split every run.")


if __name__ == "__main__":
    main()
