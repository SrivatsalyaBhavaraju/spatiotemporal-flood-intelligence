"""
Unit tests for src/nlp/finalize_gazetteer.py (task 2.9).

Covers the pure text/table logic (candidate extraction, classification,
merging, the name->coordinate flattening) with small synthetic inputs. The
network-dependent parts (re-fetching source pages, OSM Overpass resolution)
are validated by an actual live run against real data (see developing.md's
task 2.9 notes), matching this repo's convention of not mocking external
services.

Run with:
    pytest tests/test_finalize_gazetteer.py -v
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nlp.finalize_gazetteer import (  # noqa: E402
    MIN_CANDIDATE_WORDS,
    build_new_rows,
    classify_candidates,
    extract_candidate_phrases,
    geocode_candidate,
    to_name_coordinate_dict,
)


class TestExtractCandidatePhrases:
    def test_two_word_place_name_extracted(self):
        text = "Residents of Tansi Nagar in Velachery were evacuated by boat."
        candidates = extract_candidate_phrases(text)
        assert "Tansi Nagar" in candidates

    def test_leading_stopword_stripped_not_whole_phrase_dropped(self):
        text = "The Adyar River overflowed its banks near the bridge."
        candidates = extract_candidate_phrases(text)
        assert "Adyar River" in candidates
        assert "The Adyar River" not in candidates

    def test_single_word_never_extracted(self):
        text = "Chennai experienced severe flooding across the city."
        candidates = extract_candidate_phrases(text)
        assert all(len(c.split()) >= MIN_CANDIDATE_WORDS for c in candidates)
        assert "Chennai" not in candidates

    def test_phrase_reduced_to_single_word_after_stripping_is_dropped(self):
        # "According Officials" -> strip "According" -> "Officials" (1 word) -> dropped
        text = "According Officials confirmed the death toll had risen."
        candidates = extract_candidate_phrases(text)
        assert "Officials" not in candidates
        assert "According Officials" not in candidates

    def test_no_capitalized_phrases_returns_empty(self):
        assert extract_candidate_phrases("the rain fell heavily all night long") == []


class TestClassifyCandidates:
    def test_covered_vs_novel_case_insensitive(self):
        existing = {"Velachery", "Adyar River"}
        candidates = ["Velachery", "velachery", "Tansi Nagar", "Adyar River", "Tansi Nagar"]
        novel, covered_count = classify_candidates(candidates, existing)
        assert novel == ["Tansi Nagar"]  # deduplicated, sorted
        assert covered_count == 3  # "Velachery", "velachery", "Adyar River"

    def test_all_novel_when_nothing_matches(self):
        novel, covered_count = classify_candidates(["Foo Bar", "Baz Qux"], {"Something Else"})
        assert novel == ["Baz Qux", "Foo Bar"]
        assert covered_count == 0

    def test_empty_candidates(self):
        novel, covered_count = classify_candidates([], {"Anything"})
        assert novel == []
        assert covered_count == 0


class TestGeocodeCandidate:
    def test_exact_case_insensitive_lookup(self):
        index = {"tansi nagar": (80.22, 12.98)}
        assert geocode_candidate("Tansi Nagar", index) == (80.22, 12.98)
        assert geocode_candidate("TANSI NAGAR", index) == (80.22, 12.98)

    def test_no_match_returns_none(self):
        index = {"tansi nagar": (80.22, 12.98)}
        assert geocode_candidate("Prime Minister", index) is None

    def test_strips_whitespace(self):
        index = {"tansi nagar": (80.22, 12.98)}
        assert geocode_candidate("  Tansi Nagar  ", index) == (80.22, 12.98)


class TestBuildNewRows:
    def test_empty_resolved_returns_empty_frame(self):
        wards = gpd.GeoDataFrame({"Ward_No": [1], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
        result = build_new_rows([], wards)
        assert len(result) == 0
        assert list(result.columns) == ["name", "name_ta", "type", "subtype", "ward_no", "lon", "lat"]

    def test_resolved_rows_get_ward_tagged(self):
        wards = gpd.GeoDataFrame(
            {"Ward_No": [1, 2], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]}, crs="EPSG:4326"
        )
        resolved = [{"name": "Tansi Nagar", "lon": 0.5, "lat": 0.5}]  # inside ward 1
        result = build_new_rows(resolved, wards)
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Tansi Nagar"
        assert result.iloc[0]["type"] == "distress_text_candidate"
        assert result.iloc[0]["ward_no"] == 1


class TestToNameCoordinateDict:
    def test_flattens_to_name_lon_lat(self):
        gaz = pd.DataFrame({
            "name": ["Velachery"], "type": ["locality"], "lon": [80.22], "lat": [12.98],
        })
        result = to_name_coordinate_dict(gaz)
        assert result == {"Velachery": [80.22, 12.98]}

    def test_prefers_locality_over_road_for_same_name(self):
        gaz = pd.DataFrame({
            "name": ["Velachery", "Velachery"],
            "type": ["road", "locality"],
            "lon": [80.10, 80.22],
            "lat": [12.90, 12.98],
        })
        result = to_name_coordinate_dict(gaz)
        assert result["Velachery"] == [80.22, 12.98]  # the locality row, not the road row

    def test_one_entry_per_unique_name(self):
        gaz = pd.DataFrame({
            "name": ["A", "A", "B"], "type": ["road", "road", "landmark"],
            "lon": [1.0, 2.0, 3.0], "lat": [1.0, 2.0, 3.0],
        })
        result = to_name_coordinate_dict(gaz)
        assert set(result.keys()) == {"A", "B"}
