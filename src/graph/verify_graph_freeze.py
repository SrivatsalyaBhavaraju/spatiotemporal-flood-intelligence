"""
Task 3.5 -- freeze the graph structure: from this point forward, no more
topology changes (developing.md's own Phase 3 plan). Ground truth (task
3.4) and the model input schema (task 2.7) are already keyed off the EXACT
segment_id set and ordering tasks 2.1/2.2 produced -- any future topology
change (a different OSM pull, an edited AOI, changed primal->line-graph
logic) would silently break that alignment without this check.

This is not a "protect against non-determinism" concern -- tasks 2.1/2.2's
pipeline is deterministic and was re-run 5+ times over the course of this
project's session, producing the identical 6,971/17,195 primal and
17,195/48,732 line-graph counts every single time. It's a PROCESS
commitment: nobody edits build_primal_graph.py/build_line_graph.py's
logic, the AOI, or re-pulls OSM data from this point on without
deliberately updating FROZEN_FINGERPRINT below and re-verifying every
downstream task (2.3+, 3.1-3.4, eventually 4.x) still holds against the
new topology.

Freeze contract (see FROZEN_FINGERPRINT below): node_count, edge_count,
and a SHA256 hash of the sorted segment_id list -- catches same-count-
but-different-IDs drift a bare count comparison would miss (e.g. an OSM
re-pull that happens to return the same number of segments but different
ones). Also checks that every downstream table already built on top of
the graph (task 2.3's static features, task 2.7's node_order.json) still
carries the EXACT same segment_id set -- a real cross-check that the
already-built pipeline hasn't drifted internally, not just a re-statement
of 2.1/2.2's own already-known numbers.

Usage:
    python src/graph/verify_graph_freeze.py

Required inputs:
    data/processed/graph/line_graph_nodes.geojson   (task 2.2)
    data/processed/graph/line_graph_edges.geojson    (task 2.2)
    data/processed/graph/static_features.geojson      (task 2.3, optional -- checked if present)
    data/processed/model_input/node_order.json         (task 2.7, optional -- checked if present)
"""
import hashlib
import json
import sys
from pathlib import Path

import geopandas as gpd

REPO_ROOT = Path(__file__).resolve().parents[2]
LINE_GRAPH_NODES = REPO_ROOT / "data" / "processed" / "graph" / "line_graph_nodes.geojson"
LINE_GRAPH_EDGES = REPO_ROOT / "data" / "processed" / "graph" / "line_graph_edges.geojson"
STATIC_FEATURES = REPO_ROOT / "data" / "processed" / "graph" / "static_features.geojson"
NODE_ORDER_PATH = REPO_ROOT / "data" / "processed" / "model_input" / "node_order.json"
REPORT_PATH = REPO_ROOT / "data" / "processed" / "graph" / "graph_freeze_verification_report.json"

# The frozen contract -- computed once (19 Sep 2026) from the real,
# already-validated graph (tasks 2.1/2.2's own recorded numbers,
# developing.md SS6) and committed here as code so it persists across
# sessions without needing a committed data file (data/processed/ is
# deliberately regenerable-only throughout this repo -- see .gitignore).
FROZEN_FINGERPRINT = {
    "node_count": 17195,
    "edge_count": 48732,
    "segment_id_set_sha256": "ca3ad911e59f0bad1fdcc712d110ce42e6e7ec19a5ad350861e1970eb9c7f256",
}


class Phase3FreezeError(FileNotFoundError):
    """Raised when a required task 2.2 artifact is missing. Matches the
    *ArtifactError convention used across src/graph, src/ground_truth,
    src/nlp, src/models/gnn."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase3FreezeError(
            f"Required artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first."
        )
    return path


# --------------------------------------------------------------------------
# Fingerprinting (pure logic -- unit-tested)
# --------------------------------------------------------------------------

def hash_segment_ids(segment_ids) -> str:
    """SHA256 of the sorted, pipe-joined segment_id list -- deterministic
    regardless of input ordering or container type."""
    return hashlib.sha256("|".join(sorted(str(s) for s in segment_ids)).encode()).hexdigest()


def compute_fingerprint(node_count: int, edge_count: int, segment_ids) -> dict:
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "segment_id_set_sha256": hash_segment_ids(segment_ids),
    }


def compare_fingerprints(actual: dict, frozen: dict = FROZEN_FINGERPRINT) -> dict:
    """Field-by-field comparison. Returns {"match": bool, "mismatches": {...}}."""
    mismatches = {k: {"frozen": frozen[k], "actual": actual.get(k)} for k in frozen if actual.get(k) != frozen[k]}
    return {"match": len(mismatches) == 0, "mismatches": mismatches}


def check_segment_id_consistency(reference_ids: set, other_ids: set, other_label: str) -> dict:
    """Does `other_ids` (e.g. task 2.3's static_features.geojson) carry
    EXACTLY the frozen segment_id set -- not just the same count."""
    missing = reference_ids - other_ids
    extra = other_ids - reference_ids
    return {
        "label": other_label,
        "consistent": len(missing) == 0 and len(extra) == 0,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing_sample": sorted(missing)[:10],
        "extra_sample": sorted(extra)[:10],
    }


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    print(f"Loading task 2.2 line graph <- {LINE_GRAPH_NODES}")
    try:
        _require_file(LINE_GRAPH_NODES, "src/graph/build_line_graph.py (task 2.2)")
        _require_file(LINE_GRAPH_EDGES, "src/graph/build_line_graph.py (task 2.2)")
    except Phase3FreezeError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    nodes = gpd.read_file(LINE_GRAPH_NODES)
    edges = gpd.read_file(LINE_GRAPH_EDGES)
    segment_ids = set(nodes["segment_id"])
    print(f"  -> {len(nodes)} nodes, {len(edges)} edges")

    actual_fingerprint = compute_fingerprint(len(nodes), len(edges), segment_ids)
    comparison = compare_fingerprints(actual_fingerprint)

    report = {"actual_fingerprint": actual_fingerprint, "frozen_fingerprint": FROZEN_FINGERPRINT, "comparison": comparison}

    print("\nChecking downstream tables for segment_id consistency with the frozen set ...")
    consistency_checks = []
    if STATIC_FEATURES.exists():
        static_ids = set(gpd.read_file(STATIC_FEATURES)["segment_id"])
        check = check_segment_id_consistency(segment_ids, static_ids, "task 2.3 static_features.geojson")
        consistency_checks.append(check)
        print(f"  {check['label']}: {'OK' if check['consistent'] else 'MISMATCH'}")
    if NODE_ORDER_PATH.exists():
        order_ids = set(json.loads(NODE_ORDER_PATH.read_text(encoding="utf-8")))
        check = check_segment_id_consistency(segment_ids, order_ids, "task 2.7 node_order.json")
        consistency_checks.append(check)
        print(f"  {check['label']}: {'OK' if check['consistent'] else 'MISMATCH'}")
    report["downstream_consistency_checks"] = consistency_checks

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved verification report -> {REPORT_PATH}")

    all_consistent = all(c["consistent"] for c in consistency_checks)
    if comparison["match"] and all_consistent:
        print("\n=== GRAPH TOPOLOGY FROZEN -- VERIFIED ===")
        print(json.dumps(actual_fingerprint, indent=2))
        print("\nNo topology drift since tasks 2.1/2.2. Downstream tables are consistent.")
        print("From this point forward: no changes to build_primal_graph.py/build_line_graph.py's")
        print("logic, the study-ward AOI, or the source OSM pull without updating FROZEN_FINGERPRINT")
        print("here and re-verifying everything built on top of the graph (2.3+, 3.1-3.4, 4.x).")
    else:
        print("\n=== GRAPH TOPOLOGY DRIFT DETECTED ===", file=sys.stderr)
        print(json.dumps(report, indent=2, default=str), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
