"""
Unit tests for src/ground_truth/cross_check_news_advisories.py (task 3.3).

Covers the pure resolution/cross-check logic (name normalization, segment
lookup by road name, ward-fallback lookup, place-to-segment resolution,
satellite overlap cross-check) with small synthetic data. The network-
dependent part (geoparse_corpus(), which re-fetches real text) is
validated by an actual run against real data (see developing.md's task
3.3 notes), matching this repo's convention of not mocking external
services.

Run with:
    pytest tests/test_cross_check_news_advisories.py -v
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ground_truth.cross_check_news_advisories import (  # noqa: E402
    cross_check_against_satellite,
    find_segments_by_road_name,
    find_segments_in_ward,
    normalize_name_field,
    resolve_matches_to_segments,
)


class TestNormalizeNameField:
    def test_plain_string(self):
        assert normalize_name_field("Gandhi Road") == ["Gandhi Road"]

    def test_none_returns_empty(self):
        assert normalize_name_field(None) == []

    def test_nan_returns_empty(self):
        assert normalize_name_field(float("nan")) == []

    def test_real_list(self):
        assert normalize_name_field(["Anna Salai", "Mount Road"]) == ["Anna Salai", "Mount Road"]

    def test_numpy_array(self):
        assert normalize_name_field(np.array(["Anna Salai", "Mount Road"])) == ["Anna Salai", "Mount Road"]

    def test_stringified_list_repr(self):
        assert normalize_name_field("['Anna Salai', 'Mount Road']") == ["Anna Salai", "Mount Road"]

    def test_single_bracketed_string(self):
        assert normalize_name_field("[Velachery Road]") == ["Velachery Road"] or normalize_name_field("['Velachery Road']") == ["Velachery Road"]


def _line_graph_nodes():
    # All segments placed with a clear margin from ward/test-box boundaries
    # (a boundary-touching point is a real, separately-disclosed edge case
    # in this repo's spatial joins -- see build_gazetteer.py's README note
    # on boundary-adjacent segments -- not what these tests exercise).
    return gpd.GeoDataFrame(
        {
            "segment_id": ["s1", "s2", "s3", "s4"],
            "name": ["Gandhi Road", "['Anna Salai', 'Mount Road']", None, "Gandhi Road"],
            "geometry": [
                LineString([(0.2, 0.2), (0.8, 0.2)]),   # s1: inside ward 1
                LineString([(1.2, 0.2), (1.8, 0.2)]),   # s2: inside ward 1, away from s1
                LineString([(5.2, 5.2), (5.8, 5.2)]),   # s3: inside ward 2
                LineString([(0.5, 0.5), (1.5, 0.5)]),   # s4: inside ward 1
            ],
        },
        crs="EPSG:4326",
    )


def _wards():
    return gpd.GeoDataFrame({"Ward_No": [1, 2], "geometry": [box(0, 0, 2, 2), box(5, 5, 7, 7)]}, crs="EPSG:4326")


class TestFindSegmentsByRoadName:
    def test_exact_match_case_insensitive(self):
        nodes = _line_graph_nodes()
        result = find_segments_by_road_name("gandhi road", nodes)
        assert set(result) == {"s1", "s4"}

    def test_matches_within_multi_name_list(self):
        nodes = _line_graph_nodes()
        result = find_segments_by_road_name("Mount Road", nodes)
        assert result == ["s2"]

    def test_no_match_returns_empty(self):
        nodes = _line_graph_nodes()
        assert find_segments_by_road_name("Nonexistent Road", nodes) == []


class TestFindSegmentsInWard:
    def test_segments_within_ward_boundary(self):
        nodes = _line_graph_nodes()
        wards = _wards()
        result = find_segments_in_ward(1, nodes, wards)
        assert set(result) == {"s1", "s2", "s4"}

    def test_different_ward(self):
        nodes = _line_graph_nodes()
        wards = _wards()
        result = find_segments_in_ward(2, nodes, wards)
        assert result == ["s3"]

    def test_unknown_ward_returns_empty(self):
        nodes = _line_graph_nodes()
        wards = _wards()
        assert find_segments_in_ward(999, nodes, wards) == []

    def test_float_ward_no_matches_int_ward_column(self):
        # Regression: the real gazetteer's ward_no column is float64 (pandas'
        # default for a column with any NaN), while wards.geojson's Ward_No
        # is int32 -- a naive str(1.0)=="1" comparison would never match.
        nodes = _line_graph_nodes()
        wards = _wards()
        result = find_segments_in_ward(1.0, nodes, wards)
        assert set(result) == {"s1", "s2", "s4"}


class TestResolveMatchesToSegments:
    def test_road_type_resolves_by_name_match(self):
        nodes = _line_graph_nodes()
        wards = _wards()
        gaz = pd.DataFrame({"name": ["Gandhi Road"], "type": ["road"], "ward_no": [1.0]})
        geoparsed = pd.DataFrame({
            "matched_name": ["Gandhi Road"], "mention_text": ["Gandhi Road"], "score": [100.0],
            "lon": [0.5], "lat": [0.0], "source_url": ["u1"], "source_label": ["news"],
        })
        resolved = resolve_matches_to_segments(geoparsed, gaz, nodes, wards)
        assert len(resolved) == 1
        row = resolved.iloc[0]
        assert row["resolution_method"] == "segment_name_match"
        assert set(row["segment_ids"].split(";")) == {"s1", "s4"}

    def test_locality_type_falls_back_to_ward(self):
        nodes = _line_graph_nodes()
        wards = _wards()
        gaz = pd.DataFrame({"name": ["Velachery"], "type": ["locality"], "ward_no": [2.0]})
        geoparsed = pd.DataFrame({
            "matched_name": ["Velachery"], "mention_text": ["Velachery"], "score": [100.0],
            "lon": [6.0], "lat": [6.0], "source_url": ["u1"], "source_label": ["news"],
        })
        resolved = resolve_matches_to_segments(geoparsed, gaz, nodes, wards)
        assert len(resolved) == 1
        row = resolved.iloc[0]
        assert row["resolution_method"] == "ward_fallback"
        assert row["segment_ids"] == "s3"

    def test_road_with_no_segment_match_falls_back_to_ward(self):
        nodes = _line_graph_nodes()
        wards = _wards()
        gaz = pd.DataFrame({"name": ["Unknown Road"], "type": ["road"], "ward_no": [1.0]})
        geoparsed = pd.DataFrame({
            "matched_name": ["Unknown Road"], "mention_text": ["Unknown Road"], "score": [90.0],
            "lon": [0.5], "lat": [0.5], "source_url": ["u1"], "source_label": ["news"],
        })
        resolved = resolve_matches_to_segments(geoparsed, gaz, nodes, wards)
        row = resolved.iloc[0]
        assert row["resolution_method"] == "ward_fallback"
        assert set(row["segment_ids"].split(";")) == {"s1", "s2", "s4"}

    def test_non_road_type_still_gets_precise_segment_match_when_name_matches(self):
        # Regression: task 2.9 tags all its OSM-verified additions
        # "distress_text_candidate" even when the matched feature is
        # actually a road -- the segment-name match must still be tried.
        nodes = _line_graph_nodes()
        wards = _wards()
        gaz = pd.DataFrame({"name": ["Gandhi Road"], "type": ["distress_text_candidate"], "ward_no": [1.0]})
        geoparsed = pd.DataFrame({
            "matched_name": ["Gandhi Road"], "mention_text": ["Gandhi Road"], "score": [100.0],
            "lon": [0.5], "lat": [0.0], "source_url": ["u1"], "source_label": ["news"],
        })
        resolved = resolve_matches_to_segments(geoparsed, gaz, nodes, wards)
        row = resolved.iloc[0]
        assert row["resolution_method"] == "segment_name_match"
        assert set(row["segment_ids"].split(";")) == {"s1", "s4"}

    def test_aggregates_multiple_mentions_of_same_place(self):
        nodes = _line_graph_nodes()
        wards = _wards()
        gaz = pd.DataFrame({"name": ["Velachery"], "type": ["locality"], "ward_no": [2.0]})
        geoparsed = pd.DataFrame({
            "matched_name": ["Velachery", "Velachery"], "mention_text": ["Velachery", "velachery"], "score": [100.0, 90.0],
            "lon": [6.0, 6.0], "lat": [6.0, 6.0], "source_url": ["u1", "u2"], "source_label": ["news", "news"],
        })
        resolved = resolve_matches_to_segments(geoparsed, gaz, nodes, wards)
        assert len(resolved) == 1
        assert resolved.iloc[0]["n_mentions"] == 2

    def test_unmatched_gazetteer_name_skipped(self):
        nodes = _line_graph_nodes()
        wards = _wards()
        gaz = pd.DataFrame({"name": ["Something Else"], "type": ["locality"], "ward_no": [1.0]})
        geoparsed = pd.DataFrame({
            "matched_name": ["Not In Gazetteer"], "mention_text": ["x"], "score": [90.0],
            "lon": [0.0], "lat": [0.0], "source_url": ["u1"], "source_label": ["news"],
        })
        resolved = resolve_matches_to_segments(geoparsed, gaz, nodes, wards)
        assert len(resolved) == 0


class TestCrossCheckAgainstSatellite:
    def test_no_news_segments_returns_zero(self, tmp_path):
        nodes = _line_graph_nodes()
        resolved = pd.DataFrame({"segment_ids": []})
        report = cross_check_against_satellite(resolved, nodes, sentinel1_path=tmp_path / "a.geojson", bhuvan_path=tmp_path / "b.geojson")
        assert report["n_news_flagged_segments"] == 0

    def test_unavailable_when_satellite_files_missing(self, tmp_path):
        nodes = _line_graph_nodes()
        resolved = pd.DataFrame({"segment_ids": ["s1;s2"]})
        report = cross_check_against_satellite(resolved, nodes, sentinel1_path=tmp_path / "a.geojson", bhuvan_path=tmp_path / "b.geojson")
        assert report["n_news_flagged_segments"] == 2
        assert report["sentinel1_overlap"]["available"] is False
        assert report["bhuvan_nrsc_overlap"]["available"] is False

    def test_overlap_and_news_only_computed(self, tmp_path):
        nodes = _line_graph_nodes()
        resolved = pd.DataFrame({"segment_ids": ["s1;s2;s3"]})

        s1_path = tmp_path / "s1.geojson"
        gpd.GeoDataFrame({"geometry": [box(-1, -1, 1.0, 1.0)]}, crs="EPSG:4326").to_file(s1_path, driver="GeoJSON")  # covers s1 (0.2-0.8) only
        bh_path = tmp_path / "bh.geojson"
        gpd.GeoDataFrame({"geometry": [box(5.0, 5.0, 6.0, 6.0)]}, crs="EPSG:4326").to_file(bh_path, driver="GeoJSON")  # covers s3 (5.2-5.8)

        report = cross_check_against_satellite(resolved, nodes, sentinel1_path=s1_path, bhuvan_path=bh_path)
        assert report["sentinel1_overlap"]["available"] is True
        assert report["bhuvan_nrsc_overlap"]["available"] is True
        # s2 (1.2-1.8) falls outside both boxes -> news-only
        assert report["news_only_segments"]["count"] == 1
        assert report["news_only_segments"]["segment_ids_sample"] == ["s2"]
