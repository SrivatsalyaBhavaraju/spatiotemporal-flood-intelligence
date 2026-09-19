"""
Unit tests for src/nlp/fuzzy_geoparse.py (task 2.10).

Covers the pure text logic (tokenizing, windowing, fuzzy matching, overlap
resolution, end-to-end geoparse_text) with a small synthetic gazetteer --
no real data needed. The real-corpus validation (validate_against_corpus())
is exercised by an actual run against committed data (see developing.md's
task 2.10 notes), matching this repo's convention of not mocking real data
sources.

Run with:
    pytest tests/test_fuzzy_geoparse.py -v
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nlp.fuzzy_geoparse import (  # noqa: E402
    MAX_NGRAM,
    Phase2GeoparseError,
    SCORE_CUTOFF,
    _is_substring_of_any,
    fuzzy_match_windows,
    generate_candidate_windows,
    geoparse_text,
    load_gazetteer,
    resolve_overlaps,
    tokenize,
    validate_against_corpus,
)


def _gazetteer():
    return {
        "tansi nagar": {"name": "Tansi Nagar", "lon": 80.21, "lat": 12.97},
        "velachery": {"name": "Velachery", "lon": 80.22, "lat": 12.98},
        "adyar": {"name": "Adyar", "lon": 80.25, "lat": 13.00},
        "adyar river": {"name": "Adyar River", "lon": 80.26, "lat": 13.01},
        "gandhi road": {"name": "Gandhi Road", "lon": 80.23, "lat": 12.99},
    }


class TestTokenize:
    def test_words_with_offsets(self):
        tokens = tokenize("Tansi Nagar, Velachery!")
        words = [t[0] for t in tokens]
        assert words == ["Tansi", "Nagar", "Velachery"]
        assert tokens[0] == ("Tansi", 0, 5)

    def test_empty_text(self):
        assert tokenize("") == []


class TestGenerateCandidateWindows:
    def test_window_counts_and_longest_first(self):
        text = "Tansi Nagar Velachery"
        tokens = tokenize(text)
        windows = generate_candidate_windows(tokens, text, max_ngram=3)
        # 3 tokens -> n=3:1, n=2:2, n=1:3 windows = 6 total, longest-n first
        assert len(windows) == 6
        assert windows[0]["n_words"] == 3
        assert windows[-1]["n_words"] == 1

    def test_max_ngram_capped_by_token_count(self):
        text = "Adyar"
        tokens = tokenize(text)
        windows = generate_candidate_windows(tokens, text, max_ngram=4)
        assert len(windows) == 1
        assert windows[0]["n_words"] == 1

    def test_window_text_matches_original_span(self):
        text = "Tansi Nagar Velachery"
        tokens = tokenize(text)
        windows = generate_candidate_windows(tokens, text, max_ngram=2)
        two_word = next(w for w in windows if w["n_words"] == 2 and w["start"] == 0)
        assert two_word["text"] == "Tansi Nagar"


class TestFuzzyMatchWindows:
    def test_exact_match_scores_100(self):
        gaz = _gazetteer()
        windows = [{"text": "Velachery", "start": 0, "end": 9, "n_words": 1}]
        matches = fuzzy_match_windows(windows, gaz)
        assert len(matches) == 1
        assert matches[0]["matched_name"] == "Velachery"
        assert matches[0]["score"] == 100.0
        assert matches[0]["lon"] == 80.22

    def test_typo_still_matches_above_cutoff(self):
        gaz = _gazetteer()
        windows = [{"text": "velacherry", "start": 0, "end": 10, "n_words": 1}]  # typo
        matches = fuzzy_match_windows(windows, gaz)
        assert len(matches) == 1
        assert matches[0]["matched_name"] == "Velachery"

    def test_generic_word_does_not_falsely_match_longer_name(self):
        # "road" should NOT fuzzy-match "Gandhi Road" -- see module docstring's
        # scorer-choice note on why fuzz.ratio (not WRatio) is used.
        gaz = _gazetteer()
        windows = [{"text": "road", "start": 0, "end": 4, "n_words": 1}]
        matches = fuzzy_match_windows(windows, gaz)
        assert matches == []

    def test_unrelated_word_does_not_match(self):
        gaz = _gazetteer()
        windows = [{"text": "flooding", "start": 0, "end": 8, "n_words": 1}]
        assert fuzzy_match_windows(windows, gaz) == []


class TestResolveOverlaps:
    def test_prefers_exact_short_match_over_padded_longer_one(self):
        matches = [
            {"text": "Tansi Nagar in", "start": 0, "end": 14, "n_words": 3, "score": 88.0, "matched_name": "Tansi Nagar", "lon": 0, "lat": 0},
            {"text": "Tansi Nagar", "start": 0, "end": 11, "n_words": 2, "score": 100.0, "matched_name": "Tansi Nagar", "lon": 0, "lat": 0},
        ]
        resolved = resolve_overlaps(matches)
        assert len(resolved) == 1
        assert resolved[0]["text"] == "Tansi Nagar"

    def test_prefers_longer_match_when_scores_tie(self):
        matches = [
            {"text": "Adyar", "start": 0, "end": 5, "n_words": 1, "score": 100.0, "matched_name": "Adyar", "lon": 0, "lat": 0},
            {"text": "Adyar River", "start": 0, "end": 11, "n_words": 2, "score": 100.0, "matched_name": "Adyar River", "lon": 0, "lat": 0},
        ]
        resolved = resolve_overlaps(matches)
        assert len(resolved) == 1
        assert resolved[0]["matched_name"] == "Adyar River"

    def test_non_overlapping_matches_both_kept_in_position_order(self):
        matches = [
            {"text": "Velachery", "start": 20, "end": 29, "n_words": 1, "score": 100.0, "matched_name": "Velachery", "lon": 0, "lat": 0},
            {"text": "Tansi Nagar", "start": 0, "end": 11, "n_words": 2, "score": 100.0, "matched_name": "Tansi Nagar", "lon": 0, "lat": 0},
        ]
        resolved = resolve_overlaps(matches)
        assert [m["matched_name"] for m in resolved] == ["Tansi Nagar", "Velachery"]

    def test_empty_input(self):
        assert resolve_overlaps([]) == []


class TestGeoparseTextEndToEnd:
    def test_real_example_from_task_1_13_corpus(self):
        gaz = _gazetteer()
        text = ("Tansi Nagar in Velachery was severely inundated during the incessant rains. "
                "Families living on interior roads had to be evacuated by boat.")
        results = geoparse_text(text, gaz)
        names = [r["matched_name"] for r in results]
        assert names == ["Tansi Nagar", "Velachery"]
        assert results[0]["score"] == 100.0
        assert results[1]["score"] == 100.0

    def test_no_places_mentioned_returns_empty(self):
        gaz = _gazetteer()
        results = geoparse_text("The rain continued for several days without stopping.", gaz)
        assert results == []

    def test_typo_and_case_insensitive(self):
        gaz = _gazetteer()
        results = geoparse_text("stuck near VELACHERRY signal need help", gaz)
        assert len(results) == 1
        assert results[0]["matched_name"] == "Velachery"


class TestLoadGazetteer:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Phase2GeoparseError, match="finalize_gazetteer.py"):
            load_gazetteer(tmp_path / "does_not_exist.json")

    def test_round_trip(self, tmp_path):
        import json
        p = tmp_path / "gaz.json"
        p.write_text(json.dumps({"Velachery": [80.22, 12.98]}), encoding="utf-8")
        gaz = load_gazetteer(p)
        assert gaz == {"velachery": {"name": "Velachery", "lon": 80.22, "lat": 12.98}}


def test_module_constants():
    assert MAX_NGRAM == 4
    assert SCORE_CUTOFF == 85.0


class TestIsSubstringOfAny:
    def test_true_when_strict_substring_of_a_different_candidate(self):
        assert _is_substring_of_any("Adyar", {"Adyar River"}) is True

    def test_false_when_no_containing_candidate(self):
        assert _is_substring_of_any("Adyar", {"Velachery"}) is False

    def test_false_for_itself_only(self):
        assert _is_substring_of_any("Adyar", {"Adyar"}) is False

    def test_respects_word_boundaries(self):
        # "Ram" should not be considered a substring of "Ramnagar" (no boundary)
        assert _is_substring_of_any("Ram", {"Ramnagar"}) is False


class TestValidateAgainstCorpus:
    def test_absorbed_miss_not_counted_as_real_miss(self, tmp_path):
        gaz = _gazetteer()
        csv_path = tmp_path / "corpus.csv"
        # baseline double-counts "Adyar" (substring of "Adyar River") as its own hit;
        # our overlap resolution correctly keeps only the more specific "Adyar River".
        csv_path.write_text(
            "source_url,source_label,paragraph_text,matched_locations\n"
            'u,l,"families along the Adyar River were evacuated","Adyar; Adyar River"\n',
            encoding="utf-8",
        )
        report = validate_against_corpus(gaz, corpus_path=csv_path)
        assert report["total_absorbed_by_more_specific_match"] == 1
        assert report["recall_excluding_absorbed"] == 1.0
        assert report["per_passage_examples"][0]["real_misses"] == []
        assert report["per_passage_examples"][0]["absorbed_by_more_specific_match"] == ["Adyar"]

    def test_missing_corpus_file_raises(self, tmp_path):
        gaz = _gazetteer()
        with pytest.raises(Phase2GeoparseError, match="collect_distress_text.py"):
            validate_against_corpus(gaz, corpus_path=tmp_path / "nope.csv")
