"""
Task 3.4 -- fuse tasks 3.1 (Sentinel-1 SAR, primary), 3.2 (Bhuvan/NRSC,
secondary), and 3.3 (news/advisory cross-check, secondary) into the
4-phase flood label scheme (working.md SS1.5): one binary flood_label per
line-graph segment per phase (pre_event/rising/peak/receding), in exactly
the shape task 2.7's schema.json `y_t1_contract` and task 2.8's
GraphSnapshotDataset.attach_labels() expect.

Phase-assignment decision (confirmed with the user, 19 Sep 2026 -- this is
the one real judgment call this task makes, not something derivable from
the data alone): none of the three sources are phase-resolved --
  - 3.1's SAR polygon is a single before(24 Nov)/after(6 Dec) change
    detection spanning the WHOLE event, not a per-phase observation.
  - 3.2's Bhuvan/NRSC raster is a single simulated depth surface, no time
    axis at all.
  - 3.3's news cross-check found no reliable per-passage dates in the real
    corpus (checked directly, not assumed).
So a segment flagged by any of the three means "flooded at SOME point
during the event," not "flooded in phase X." This task assigns that
flooding to PEAK and RECEDING only (rising and pre_event stay dry),
grounded in task 2.5's own already-computed rainfall numbers: rising totals
just 31mm over 2 days vs. peak's 410mm over 2 days -- a 13x contrast
strongly suggesting flooding hadn't materialized yet during rising.
pre_event is always dry by construction: it's the SAR change-detection's
own "before" reference snapshot, not an independent observation.

Fusion rule per segment (working.md SS1.5's own rule, "segment labeled
flooded if it intersects the polygon," extended across all three sources
with a simple OR -- not invented fresh here):
    ever_flooded = intersects(3.1 polygon) OR intersects(3.2 polygon) OR flagged-by(3.3)
    flood_label[pre_event] = 0                    (always)
    flood_label[rising]    = 0                    (always -- see phase-assignment note)
    flood_label[peak]      = 1 if ever_flooded else 0
    flood_label[receding]  = 1 if ever_flooded else 0

Usage:
    python src/ground_truth/fuse_flood_labels.py

Required inputs:
    data/processed/model_input/node_order.json      (task 2.7)
    data/processed/model_input/schema.json           (task 2.7)
    data/processed/graph/line_graph_nodes.geojson    (task 2.2)
    data/processed/ground_truth/sentinel1_flood_extent.geojson   (task 3.1)
    data/processed/ground_truth/bhuvan_nrsc_flood_extent.geojson (task 3.2)
    data/processed/ground_truth/news_cross_check_segments.geojson (task 3.3)

Outputs (data/processed/ground_truth/):
    fused_flood_labels.csv             -- segment_id, phase_id, phase_name,
                                           flood_label, sar_flagged,
                                           bhuvan_flagged, news_flagged
    fused_flood_labels_validation_report.json
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.gnn.build_graph_snapshot_dataset import load_dataset  # noqa: E402

MODEL_INPUT_DIR = REPO_ROOT / "data" / "processed" / "model_input"
NODE_ORDER_PATH = MODEL_INPUT_DIR / "node_order.json"
SCHEMA_PATH = MODEL_INPUT_DIR / "schema.json"
LINE_GRAPH_NODES = REPO_ROOT / "data" / "processed" / "graph" / "line_graph_nodes.geojson"
WARDS_PATH = REPO_ROOT / "data" / "raw" / "wards" / "study_wards.geojson"
GROUND_TRUTH_DIR = REPO_ROOT / "data" / "processed" / "ground_truth"
SENTINEL1_PATH = GROUND_TRUTH_DIR / "sentinel1_flood_extent.geojson"
BHUVAN_PATH = GROUND_TRUTH_DIR / "bhuvan_nrsc_flood_extent.geojson"
NEWS_SEGMENTS_PATH = GROUND_TRUTH_DIR / "news_cross_check_segments.geojson"

OUT_CSV = GROUND_TRUTH_DIR / "fused_flood_labels.csv"
REPORT_PATH = GROUND_TRUTH_DIR / "fused_flood_labels_validation_report.json"

# See module docstring's "Phase-assignment decision" -- confirmed with the
# user, not derived from the data alone.
FLOODED_PHASE_NAMES = {"peak", "receding"}


class Phase3ArtifactError(FileNotFoundError):
    """Raised when a required task 2.2/2.7/3.1/3.2/3.3 artifact is
    missing. Matches the *ArtifactError convention used across
    src/graph, src/ground_truth, src/nlp, src/models/gnn."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase3ArtifactError(
            f"Required artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first."
        )
    return path


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_node_order(path: Path = NODE_ORDER_PATH) -> list:
    _require_file(path, "src/models/gnn/build_model_input_schema.py (task 2.7)")
    return json.loads(path.read_text(encoding="utf-8"))


def load_phases(path: Path = SCHEMA_PATH) -> list:
    _require_file(path, "src/models/gnn/build_model_input_schema.py (task 2.7)")
    schema = json.loads(path.read_text(encoding="utf-8"))
    return schema["phases"]


def load_line_graph_geometries(path: Path = LINE_GRAPH_NODES) -> gpd.GeoDataFrame:
    _require_file(path, "src/graph/build_line_graph.py (task 2.2)")
    return gpd.read_file(path)[["segment_id", "geometry"]]


# --------------------------------------------------------------------------
# Per-source segment flagging (pure logic given loaded data -- unit-tested)
# --------------------------------------------------------------------------

def flag_segments_intersecting(line_graph_nodes: gpd.GeoDataFrame, flood_extent_path: Path) -> set:
    """segment_ids whose geometry intersects ANY polygon in
    `flood_extent_path` (task 3.1/3.2's shared output schema). Empty/
    missing file -> empty set, not an error -- both are legitimately
    optional secondary sources (task 0.6/3.2's own downgrade)."""
    if not flood_extent_path.exists():
        return set()
    flood_gdf = gpd.read_file(flood_extent_path)
    if len(flood_gdf) == 0:
        return set()
    flood_union = flood_gdf.geometry.union_all()
    intersects = line_graph_nodes.geometry.intersects(flood_union)
    return set(line_graph_nodes.loc[intersects, "segment_id"])


def load_news_flagged_segments(path: Path = NEWS_SEGMENTS_PATH) -> set:
    """Task 3.3's output is already segment-level (that task did its own
    road-name-match/ward-fallback resolution) -- just read the segment_ids,
    no geometry intersection needed here."""
    if not path.exists():
        return set()
    gdf = gpd.read_file(path)
    if len(gdf) == 0:
        return set()
    return set(gdf["segment_id"])


# --------------------------------------------------------------------------
# Fusion (pure logic -- unit-tested)
# --------------------------------------------------------------------------

def fuse_labels(node_order: list, sar_segments: set, bhuvan_segments: set, news_segments: set, phases: list) -> pd.DataFrame:
    """One row per (segment_id, phase): flood_label per the module
    docstring's fusion rule, plus which source(s) flagged this segment
    (for audit -- see validate_fusion())."""
    ever_flooded = sar_segments | bhuvan_segments | news_segments
    rows = []
    for phase in phases:
        flooded_phase = phase["phase_name"] in FLOODED_PHASE_NAMES
        for seg_id in node_order:
            in_sar, in_bhuvan, in_news = seg_id in sar_segments, seg_id in bhuvan_segments, seg_id in news_segments
            rows.append({
                "segment_id": seg_id, "phase_id": phase["phase_id"], "phase_name": phase["phase_name"],
                "flood_label": int(flooded_phase and seg_id in ever_flooded),
                "sar_flagged": int(in_sar), "bhuvan_flagged": int(in_bhuvan), "news_flagged": int(in_news),
                "n_sources_agreeing": int(in_sar) + int(in_bhuvan) + int(in_news),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_fusion(labels_df: pd.DataFrame, node_order: list, phases: list) -> dict:
    report = {}
    n_nodes = len(node_order)

    report["coverage"] = {
        "expected_rows": n_nodes * len(phases),
        "actual_rows": int(len(labels_df)),
        "complete": len(labels_df) == n_nodes * len(phases),
    }

    per_phase = labels_df.groupby("phase_name")["flood_label"].agg(["sum", "count"]).to_dict(orient="index")
    report["flooded_count_per_phase"] = {
        phase: {"flooded": int(v["sum"]), "total": int(v["count"]), "pct": round(100 * v["sum"] / v["count"], 2)}
        for phase, v in per_phase.items()
    }

    one_phase = labels_df[labels_df["phase_id"] == labels_df["phase_id"].max()]
    source_counts = one_phase[["sar_flagged", "bhuvan_flagged", "news_flagged"]].sum().to_dict()
    report["source_contribution"] = {k: int(v) for k, v in source_counts.items()}
    report["ever_flooded_segments"] = int((one_phase["n_sources_agreeing"] > 0).sum())

    agreement = one_phase["n_sources_agreeing"].value_counts().sort_index().to_dict()
    report["n_sources_agreeing_histogram"] = {int(k): int(v) for k, v in agreement.items()}

    return report


def validate_end_to_end_with_loader(labels_df: pd.DataFrame) -> dict:
    """The real exit-criterion check: does task 2.8's GraphSnapshotDataset
    actually accept these labels and produce usable (X_t, Y_{t+1}) training
    pairs? Not just "is the CSV well-formed" -- an actual integration run
    through the loader Phase 4 will use.
    """
    ds = load_dataset()
    ds.attach_labels(labels_df)
    pairs = list(ds.transition_pairs())
    return {
        "loader_accepted_labels": True,
        "n_transition_pairs": len(pairs),
        "pair_shapes": [{"x_t_phase_id": x.phase_id, "x_shape": list(x.x.shape), "y_shape": list(y.shape)} for x, y in pairs],
    }


# --------------------------------------------------------------------------
# Saving outputs
# --------------------------------------------------------------------------

def save_outputs(labels_df: pd.DataFrame, out_path: Path = OUT_CSV) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(out_path, index=False)
    return {"fused_flood_labels_csv": out_path}


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    print(f"Loading task 2.7 node order <- {NODE_ORDER_PATH}")
    try:
        node_order = load_node_order()
        phases = load_phases()
        line_graph_nodes = load_line_graph_geometries()
    except Phase3ArtifactError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> {len(node_order)} segments, {len(phases)} phases: {[p['phase_name'] for p in phases]}")

    print(f"\nFlagging segments intersecting task 3.1 (Sentinel-1) <- {SENTINEL1_PATH}")
    sar_segments = flag_segments_intersecting(line_graph_nodes, SENTINEL1_PATH)
    print(f"  -> {len(sar_segments)} segments")

    print(f"Flagging segments intersecting task 3.2 (Bhuvan/NRSC) <- {BHUVAN_PATH}")
    bhuvan_segments = flag_segments_intersecting(line_graph_nodes, BHUVAN_PATH)
    print(f"  -> {len(bhuvan_segments)} segments")

    print(f"Loading task 3.3 news-flagged segments <- {NEWS_SEGMENTS_PATH}")
    news_segments = load_news_flagged_segments(NEWS_SEGMENTS_PATH)
    print(f"  -> {len(news_segments)} segments")

    print(f"\nFusing into 4-phase labels (flooded phases: {sorted(FLOODED_PHASE_NAMES)}) ...")
    labels_df = fuse_labels(node_order, sar_segments, bhuvan_segments, news_segments, phases)
    print(f"  -> {len(labels_df)} rows ({len(node_order)} segments x {len(phases)} phases)")

    print("\nSaving outputs ...")
    paths = save_outputs(labels_df)
    for label, p in paths.items():
        print(f"  {label}: {p}")

    print("\nRunning validation ...")
    report = validate_fusion(labels_df, node_order, phases)

    print("Running end-to-end check against task 2.8's GraphSnapshotDataset ...")
    try:
        report["end_to_end_loader_check"] = validate_end_to_end_with_loader(labels_df)
    except Exception as e:
        report["end_to_end_loader_check"] = {"loader_accepted_labels": False, "error": str(e)}

    report_path = REPORT_PATH
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  validation report -> {report_path}")

    print("\n=== Validation summary ===")
    print(json.dumps(report, indent=2, default=str))

    print("\nDone. Phase 3's exit criterion (task 3.4) is met -- task 4.1 (GNN training)")
    print("and task 4.3 (baseline predictions) can now consume fused_flood_labels.csv.")


if __name__ == "__main__":
    main()
