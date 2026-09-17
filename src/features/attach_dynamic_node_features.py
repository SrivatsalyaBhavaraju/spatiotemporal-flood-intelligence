"""
Task 2.6 -- derive cumulative_rainfall_t and attach both dynamic features
(rainfall_t, cumulative_rainfall_t) to every line-graph node, per phase.

Per working.md SS1 (dynamic node features): rainfall is a GLOBAL time-varying
driver, not spatially differentiating at this grid resolution -- so every
segment (line-graph node) gets the SAME rainfall_t / cumulative_rainfall_t
value at a given phase (working.md line 81's "honest caveat"). This script
does that broadcast: it does not compute anything per-segment.

cumulative_rainfall_t = running total of rainfall_t since event start
(working.md line 78), i.e. cumsum() over the phases in order -- valid only
because task 2.5's PHASES are contiguous and non-overlapping.

Usage:
    python src/features/attach_dynamic_node_features.py

Inputs:
    data/processed/features/rainfall_phase_features.csv   (task 2.5)
    data/processed/graph/line_graph_nodes.geojson          (task 2.2)

Output (data/processed/features/):
    dynamic_node_features.csv  -- long format: segment_id, phase_id,
                                   phase_name, rainfall_t, cumulative_rainfall_t
                                   One row per (segment, phase) -- ready for
                                   task 2.7 to pivot/merge into X_t alongside
                                   task 2.3's static features and task 2.4's
                                   labels.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE_FEATURES_CSV = REPO_ROOT / "data" / "processed" / "features" / "rainfall_phase_features.csv"
LINE_GRAPH_NODES = REPO_ROOT / "data" / "processed" / "graph" / "line_graph_nodes.geojson"
OUT_DIR = REPO_ROOT / "data" / "processed" / "features"


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first."
        )
    return path


def load_segment_ids(path: Path) -> list:
    """Segment ids = line-graph node ids, same key task 2.3/2.7 join on."""
    _require_file(path, "src/graph/build_line_graph.py (task 2.2)")
    nodes = gpd.read_file(path)
    if "segment_id" not in nodes.columns:
        raise ValueError(f"{path} has no `segment_id` column -- unexpected schema.")
    return sorted(nodes["segment_id"].unique())


def add_cumulative(phase_df: pd.DataFrame) -> pd.DataFrame:
    df = phase_df.sort_values("phase_id").copy()
    df["cumulative_rainfall_t"] = df["rainfall_t"].cumsum().round(3)
    return df


def broadcast_to_nodes(phase_df: pd.DataFrame, segment_ids: list) -> pd.DataFrame:
    """Cartesian product of every segment x every phase, with rainfall_t /
    cumulative_rainfall_t broadcast identically to all segments in a phase --
    this IS the deliberate design (see module docstring), not a placeholder.
    """
    rows = []
    for _, phase_row in phase_df.iterrows():
        for seg_id in segment_ids:
            rows.append({
                "segment_id": seg_id,
                "phase_id": phase_row["phase_id"],
                "phase_name": phase_row["phase_name"],
                "rainfall_t": phase_row["rainfall_t"],
                "cumulative_rainfall_t": phase_row["cumulative_rainfall_t"],
            })
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _require_file(PHASE_FEATURES_CSV, "src/features/build_rainfall_phase_features.py (task 2.5)")
    phase_df = pd.read_csv(PHASE_FEATURES_CSV)

    print("Deriving cumulative_rainfall_t (running total since event start)...")
    phase_df = add_cumulative(phase_df)
    print(phase_df[["phase_id", "phase_name", "rainfall_t", "cumulative_rainfall_t"]].to_string(index=False))

    print(f"\nLoading line-graph node ids from {LINE_GRAPH_NODES} ...")
    segment_ids = load_segment_ids(LINE_GRAPH_NODES)
    print(f"  {len(segment_ids)} segments (line-graph nodes) found.")

    print("\nBroadcasting rainfall_t / cumulative_rainfall_t to every segment, per phase...")
    dynamic_df = broadcast_to_nodes(phase_df, segment_ids)

    out_path = OUT_DIR / "dynamic_node_features.csv"
    dynamic_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}  ({len(dynamic_df)} rows = {len(segment_ids)} segments x {len(phase_df)} phases)")

    # Quick sanity check: every phase should have exactly len(segment_ids) rows,
    # and within a phase every row should share the same rainfall_t value.
    check = dynamic_df.groupby("phase_id")["rainfall_t"].nunique()
    assert (check == 1).all(), "BUG: rainfall_t is not uniform within a phase!"
    print("Sanity check passed: rainfall_t is uniform across all segments within each phase.")


if __name__ == "__main__":
    main()