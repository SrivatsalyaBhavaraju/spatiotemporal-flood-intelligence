"""
Unit tests for src/nlp/run_objective2_pipeline.py (task 4.5).

Covers the pure composition logic (run_pipeline, filter_resolved_distress_posts,
build_report) with injected fake classify_fn/geoparse_fn -- neither real
MuRIL nor the real gazetteer is needed, since task 3.8/2.10 already tested
those independently. This module's real job (composing them correctly) is
validated by an actual run against real data (see developing.md's task 4.5
notes), matching this repo's convention elsewhere.

Run with:
    pytest tests/test_run_objective2_pipeline.py -v
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nlp.run_objective2_pipeline import (  # noqa: E402
    build_report,
    filter_resolved_distress_posts,
    run_pipeline,
)

GAZETTEER = {"velachery": {"name": "Velachery", "lon": 80.22, "lat": 12.98}}


def _fake_classify(labels_by_text):
    def classify_fn(text):
        return labels_by_text[text]
    return classify_fn


def _fake_geoparse(locations_by_text):
    def geoparse_fn(text, gazetteer):
        return locations_by_text.get(text, [])
    return geoparse_fn


class TestRunPipeline:
    def test_distress_with_resolved_location(self):
        text = "Velachery is flooded"
        classify_fn = _fake_classify({text: {"label": "distress", "probability": 0.9}})
        geoparse_fn = _fake_geoparse({text: [{"matched_name": "Velachery", "lon": 80.22, "lat": 12.98, "score": 100.0}]})
        results = run_pipeline([text], classify_fn, geoparse_fn, GAZETTEER)
        assert results[0]["distress_label"] == "distress"
        assert results[0]["resolved_locations"] == [{"place": "Velachery", "lon": 80.22, "lat": 12.98, "score": 100.0}]

    def test_geoparse_still_runs_for_not_distress_posts(self):
        # both branches run independently on every post (see module docstring)
        text = "routine announcement about Velachery"
        calls = []

        def geoparse_fn(text, gazetteer):
            calls.append(text)
            return [{"matched_name": "Velachery", "lon": 80.22, "lat": 12.98, "score": 100.0}]

        classify_fn = _fake_classify({text: {"label": "not_distress", "probability": 0.1}})
        results = run_pipeline([text], classify_fn, geoparse_fn, GAZETTEER)
        assert calls == [text]  # geoparse was called even though not_distress
        assert results[0]["resolved_locations"] != []

    def test_distress_with_no_resolvable_location(self):
        text = "we are stranded please help"
        classify_fn = _fake_classify({text: {"label": "distress", "probability": 0.95}})
        geoparse_fn = _fake_geoparse({text: []})
        results = run_pipeline([text], classify_fn, geoparse_fn, GAZETTEER)
        assert results[0]["distress_label"] == "distress"
        assert results[0]["resolved_locations"] == []

    def test_result_count_matches_input_count(self):
        texts = ["a", "b", "c"]
        classify_fn = _fake_classify({t: {"label": "not_distress", "probability": 0.0} for t in texts})
        geoparse_fn = _fake_geoparse({})
        results = run_pipeline(texts, classify_fn, geoparse_fn, GAZETTEER)
        assert len(results) == 3


class TestFilterResolvedDistressPosts:
    def test_keeps_only_distress_and_resolved(self):
        results = [
            {"text": "a", "distress_label": "distress", "distress_probability": 0.9, "resolved_locations": [{"place": "X"}]},
            {"text": "b", "distress_label": "distress", "distress_probability": 0.9, "resolved_locations": []},
            {"text": "c", "distress_label": "not_distress", "distress_probability": 0.1, "resolved_locations": [{"place": "X"}]},
            {"text": "d", "distress_label": "not_distress", "distress_probability": 0.1, "resolved_locations": []},
        ]
        filtered = filter_resolved_distress_posts(results)
        assert len(filtered) == 1
        assert filtered[0]["text"] == "a"

    def test_empty_input_returns_empty(self):
        assert filter_resolved_distress_posts([]) == []


class TestBuildReport:
    def test_counts_correct(self):
        results = [
            {"text": "a", "distress_label": "distress", "distress_probability": 0.9, "resolved_locations": [{"place": "X"}]},
            {"text": "b", "distress_label": "distress", "distress_probability": 0.9, "resolved_locations": []},
            {"text": "c", "distress_label": "not_distress", "distress_probability": 0.1, "resolved_locations": [{"place": "X"}]},
        ]
        report = build_report(results, "test_run")
        assert report["n_total_posts"] == 3
        assert report["n_classified_distress"] == 2
        assert report["n_classified_not_distress"] == 1
        assert report["n_distress_with_resolved_location"] == 1
        assert report["n_distress_without_resolved_location"] == 1
        assert report["pct_distress_geolocated"] == 50.0

    def test_no_distress_posts_gives_none_pct_not_a_crash(self):
        results = [{"text": "a", "distress_label": "not_distress", "distress_probability": 0.1, "resolved_locations": []}]
        report = build_report(results, "test_run")
        assert report["pct_distress_geolocated"] is None
