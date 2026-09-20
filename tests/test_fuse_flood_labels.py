"""
Unit tests for src/ground_truth/fuse_flood_labels.py (task 3.4).

Covers the pure fusion/validation logic (per-source segment flagging,
label fusion, coverage/agreement validation) with small synthetic data.
The end-to-end check against task 2.8's real GraphSnapshotDataset
(validate_end_to_end_with_loader()) is exercised by an actual run against
real data (see developing.md's task 3.4 notes), matching this repo's
convention of not mocking real integration points.

Run with:
    pytest tests/test_fuse_flood_labels.py -v
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, box

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ground_truth.fuse_flood_labels import (  # noqa: E402
    FLOODED_PHASE_NAMES,
    flag_segments_intersecting,
    fuse_labels,
    load_news_flagged_segments,
    load_recovered_segments,
    validate_fusion,
)

PHASES = [
    {"phase_id": 0, "phase_name": "pre_event"},
    {"phase_id": 1, "phase_name": "rising"},
    {"phase_id": 2, "phase_name": "peak"},
    {"phase_id": 3, "phase_name": "receding"},
]
NODE_ORDER = ["s1", "s2", "s3", "s4"]


def _line_graph_nodes():
    return gpd.GeoDataFrame(
        {
            "segment_id": NODE_ORDER,
            "geometry": [
                LineString([(0.2, 0.2), (0.8, 0.2)]),
                LineString([(1.2, 0.2), (1.8, 0.2)]),
                LineString([(5.2, 5.2), (5.8, 5.2)]),
                LineString([(10.2, 10.2), (10.8, 10.2)]),
            ],
        },
        crs="EPSG:4326",
    )


class TestFlagSegmentsIntersecting:
    def test_missing_file_returns_empty_set(self, tmp_path):
        nodes = _line_graph_nodes()
        assert flag_segments_intersecting(nodes, tmp_path / "nope.geojson") == set()

    def test_empty_geojson_returns_empty_set(self, tmp_path):
        nodes = _line_graph_nodes()
        p = tmp_path / "empty.geojson"
        gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326").to_file(p, driver="GeoJSON")
        assert flag_segments_intersecting(nodes, p) == set()

    def test_intersecting_segments_flagged(self, tmp_path):
        nodes = _line_graph_nodes()
        p = tmp_path / "flood.geojson"
        gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326").to_file(p, driver="GeoJSON")  # covers s1 only
        assert flag_segments_intersecting(nodes, p) == {"s1"}

    def test_multiple_polygons_union_correctly(self, tmp_path):
        nodes = _line_graph_nodes()
        p = tmp_path / "flood.geojson"
        gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1), box(5, 5, 6, 6)]}, crs="EPSG:4326").to_file(p, driver="GeoJSON")  # covers s1, s3
        assert flag_segments_intersecting(nodes, p) == {"s1", "s3"}


class TestLoadNewsFlaggedSegments:
    def test_missing_file_returns_empty_set(self, tmp_path):
        assert load_news_flagged_segments(tmp_path / "nope.geojson") == set()

    def test_reads_segment_ids_directly(self, tmp_path):
        p = tmp_path / "news.geojson"
        gpd.GeoDataFrame(
            {"segment_id": ["s2", "s3"], "geometry": [LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)])]},
            crs="EPSG:4326",
        ).to_file(p, driver="GeoJSON")
        assert load_news_flagged_segments(p) == {"s2", "s3"}


class TestFuseLabels:
    def test_pre_event_and_rising_always_dry(self):
        labels = fuse_labels(NODE_ORDER, sar_segments={"s1"}, bhuvan_segments=set(), news_segments=set(), phases=PHASES)
        pre_event = labels[labels["phase_name"] == "pre_event"]
        rising = labels[labels["phase_name"] == "rising"]
        assert (pre_event["flood_label"] == 0).all()
        assert (rising["flood_label"] == 0).all()

    def test_peak_and_receding_flooded_for_ever_flooded_segment(self):
        labels = fuse_labels(NODE_ORDER, sar_segments={"s1"}, bhuvan_segments=set(), news_segments=set(), phases=PHASES)
        s1_peak = labels[(labels["segment_id"] == "s1") & (labels["phase_name"] == "peak")].iloc[0]
        s1_receding = labels[(labels["segment_id"] == "s1") & (labels["phase_name"] == "receding")].iloc[0]
        assert s1_peak["flood_label"] == 1
        assert s1_receding["flood_label"] == 1

    def test_never_flagged_segment_stays_dry_everywhere(self):
        labels = fuse_labels(NODE_ORDER, sar_segments={"s1"}, bhuvan_segments=set(), news_segments=set(), phases=PHASES)
        s4 = labels[labels["segment_id"] == "s4"]
        assert (s4["flood_label"] == 0).all()

    def test_union_across_sources(self):
        labels = fuse_labels(NODE_ORDER, sar_segments={"s1"}, bhuvan_segments={"s2"}, news_segments={"s3"}, phases=PHASES)
        peak = labels[labels["phase_name"] == "peak"].set_index("segment_id")
        assert peak.loc["s1", "flood_label"] == 1
        assert peak.loc["s2", "flood_label"] == 1
        assert peak.loc["s3", "flood_label"] == 1
        assert peak.loc["s4", "flood_label"] == 0

    def test_source_flags_and_agreement_count(self):
        labels = fuse_labels(NODE_ORDER, sar_segments={"s1"}, bhuvan_segments={"s1"}, news_segments=set(), phases=PHASES)
        s1_peak = labels[(labels["segment_id"] == "s1") & (labels["phase_name"] == "peak")].iloc[0]
        assert s1_peak["sar_flagged"] == 1
        assert s1_peak["bhuvan_flagged"] == 1
        assert s1_peak["news_flagged"] == 0
        assert s1_peak["n_sources_agreeing"] == 2

    def test_row_count_matches_segments_times_phases(self):
        labels = fuse_labels(NODE_ORDER, sar_segments=set(), bhuvan_segments=set(), news_segments=set(), phases=PHASES)
        assert len(labels) == len(NODE_ORDER) * len(PHASES)

    def test_flooded_phase_names_are_peak_and_receding(self):
        assert FLOODED_PHASE_NAMES == {"peak", "receding"}


class TestLoadRecoveredSegments:
    def test_missing_file_returns_empty_set(self, tmp_path):
        assert load_recovered_segments(tmp_path / "nope.json") == set()

    def test_reads_segment_ids(self, tmp_path):
        p = tmp_path / "recovered.json"
        p.write_text('["s1", "s3"]', encoding="utf-8")
        assert load_recovered_segments(p) == {"s1", "s3"}


class TestFuseLabelsRecovery:
    """Task 4.4 -- recovered_segments demotes flood_label to 0 in receding
    only, never in peak, and never for segments not marked recovered."""

    def test_recovered_segment_demoted_only_in_receding(self):
        labels = fuse_labels(
            NODE_ORDER, sar_segments={"s1"}, bhuvan_segments=set(), news_segments=set(), phases=PHASES,
            recovered_segments={"s1"},
        )
        s1_peak = labels[(labels["segment_id"] == "s1") & (labels["phase_name"] == "peak")].iloc[0]
        s1_receding = labels[(labels["segment_id"] == "s1") & (labels["phase_name"] == "receding")].iloc[0]
        assert s1_peak["flood_label"] == 1        # peak untouched
        assert s1_receding["flood_label"] == 0    # receding demoted
        assert s1_receding["recovered_by_18dec"] == 1

    def test_non_recovered_segment_unaffected(self):
        labels = fuse_labels(
            NODE_ORDER, sar_segments={"s1", "s2"}, bhuvan_segments=set(), news_segments=set(), phases=PHASES,
            recovered_segments={"s1"},
        )
        s2_receding = labels[(labels["segment_id"] == "s2") & (labels["phase_name"] == "receding")].iloc[0]
        assert s2_receding["flood_label"] == 1
        assert s2_receding["recovered_by_18dec"] == 0

    def test_recovery_only_applies_to_segments_actually_flooded(self):
        # s4 was never flagged flooded at all -- marking it "recovered" (a
        # nonsensical input, but the function shouldn't newly flag it)
        labels = fuse_labels(
            NODE_ORDER, sar_segments={"s1"}, bhuvan_segments=set(), news_segments=set(), phases=PHASES,
            recovered_segments={"s4"},
        )
        s4_receding = labels[(labels["segment_id"] == "s4") & (labels["phase_name"] == "receding")].iloc[0]
        assert s4_receding["flood_label"] == 0

    def test_default_empty_recovered_matches_original_behavior(self):
        with_default = fuse_labels(NODE_ORDER, sar_segments={"s1"}, bhuvan_segments=set(), news_segments=set(), phases=PHASES)
        with_empty = fuse_labels(
            NODE_ORDER, sar_segments={"s1"}, bhuvan_segments=set(), news_segments=set(), phases=PHASES,
            recovered_segments=set(),
        )
        pd.testing.assert_frame_equal(with_default.drop(columns=["recovered_by_18dec"]),
                                       with_empty.drop(columns=["recovered_by_18dec"]))


class TestValidateFusionRecovery:
    def test_reports_demoted_count(self):
        labels = fuse_labels(
            NODE_ORDER, sar_segments={"s1", "s2"}, bhuvan_segments=set(), news_segments=set(), phases=PHASES,
            recovered_segments={"s1"},
        )
        report = validate_fusion(labels, NODE_ORDER, PHASES)
        assert report["task_4_4_recovered_by_18dec"]["segments_demoted_in_receding"] == 1
        assert report["task_4_4_recovered_by_18dec"]["peak_receding_now_identical"] is False

    def test_no_recovery_reports_identical(self):
        labels = fuse_labels(NODE_ORDER, sar_segments={"s1"}, bhuvan_segments=set(), news_segments=set(), phases=PHASES)
        report = validate_fusion(labels, NODE_ORDER, PHASES)
        assert report["task_4_4_recovered_by_18dec"]["segments_demoted_in_receding"] == 0
        assert report["task_4_4_recovered_by_18dec"]["peak_receding_now_identical"] is True


class TestValidateFusion:
    def test_coverage_complete(self):
        labels = fuse_labels(NODE_ORDER, sar_segments={"s1"}, bhuvan_segments=set(), news_segments=set(), phases=PHASES)
        report = validate_fusion(labels, NODE_ORDER, PHASES)
        assert report["coverage"]["complete"] is True
        assert report["coverage"]["actual_rows"] == len(NODE_ORDER) * len(PHASES)

    def test_flooded_count_per_phase(self):
        labels = fuse_labels(NODE_ORDER, sar_segments={"s1", "s2"}, bhuvan_segments=set(), news_segments=set(), phases=PHASES)
        report = validate_fusion(labels, NODE_ORDER, PHASES)
        assert report["flooded_count_per_phase"]["pre_event"]["flooded"] == 0
        assert report["flooded_count_per_phase"]["rising"]["flooded"] == 0
        assert report["flooded_count_per_phase"]["peak"]["flooded"] == 2
        assert report["flooded_count_per_phase"]["receding"]["flooded"] == 2

    def test_source_contribution_and_ever_flooded(self):
        labels = fuse_labels(NODE_ORDER, sar_segments={"s1"}, bhuvan_segments={"s2"}, news_segments={"s1"}, phases=PHASES)
        report = validate_fusion(labels, NODE_ORDER, PHASES)
        assert report["source_contribution"] == {"sar_flagged": 1, "bhuvan_flagged": 1, "news_flagged": 1}
        assert report["ever_flooded_segments"] == 2  # s1 (sar+news) and s2 (bhuvan)
        assert report["n_sources_agreeing_histogram"][2] == 1  # s1 has 2 sources agreeing
        assert report["n_sources_agreeing_histogram"][1] == 1  # s2 has 1
