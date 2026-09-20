"""
Unit tests for src/nlp/evaluate_objective2_precision_recall.py (task 5.5).

Covers the pure precision/recall computation logic with small synthetic
data and injected fake geoparse functions. The real, headline finding
(one confirmed false positive, Nandambakkam->Adambakkam) is validated
against the actual real corpus (see developing.md's task 5.5 notes), not
reproduced here with fake data.

Run with:
    pytest tests/test_evaluate_objective2_precision_recall.py -v
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nlp.evaluate_objective2_precision_recall import (  # noqa: E402
    compute_end_to_end_metrics,
    compute_geoparse_precision,
)

GAZETTEER = {"velachery": {"name": "Velachery", "lon": 80.22, "lat": 12.98}}


def _fake_geoparse(matches_by_text):
    def geoparse_fn(text, gazetteer):
        return matches_by_text.get(text, [])
    return geoparse_fn


class TestComputeGeoparsePrecision:
    def test_all_correct_gives_precision_one(self):
        texts = ["a", "b"]
        matches = {
            "a": [{"text": "Velachery", "matched_name": "Velachery"}],
            "b": [{"text": "Velachery", "matched_name": "Velachery"}],
        }
        result = compute_geoparse_precision(texts, GAZETTEER, geoparse_fn=_fake_geoparse(matches))
        assert result["precision"] == 1.0
        assert result["total_matches"] == 2
        assert result["false_positives"] == 0

    def test_known_false_positive_excluded_from_true_positives(self):
        texts = ["a"]
        matches = {"a": [{"text": "Nandambakkam", "matched_name": "Adambakkam"}]}
        known_fp = {("nandambakkam", "Adambakkam"): "wrong place, absent from gazetteer"}
        result = compute_geoparse_precision(texts, GAZETTEER, geoparse_fn=_fake_geoparse(matches), known_false_positives=known_fp)
        assert result["precision"] == 0.0
        assert result["false_positives"] == 1
        assert result["false_positive_examples"][0]["matched_text"] == "Nandambakkam"

    def test_mixed_correct_and_incorrect(self):
        texts = ["a"]
        matches = {"a": [
            {"text": "Velachery", "matched_name": "Velachery"},
            {"text": "Nandambakkam", "matched_name": "Adambakkam"},
        ]}
        known_fp = {("nandambakkam", "Adambakkam"): "wrong place"}
        result = compute_geoparse_precision(texts, GAZETTEER, geoparse_fn=_fake_geoparse(matches), known_false_positives=known_fp)
        assert result["total_matches"] == 2
        assert result["true_positives"] == 1
        assert result["false_positives"] == 1
        assert result["precision"] == 0.5

    def test_no_matches_gives_none_precision_not_a_crash(self):
        result = compute_geoparse_precision(["a"], GAZETTEER, geoparse_fn=_fake_geoparse({}))
        assert result["total_matches"] == 0
        assert result["precision"] is None


class TestComputeEndToEndMetrics:
    def test_true_positive_when_distress_and_resolved(self):
        fold_results = [{"held_out_index": 0, "y_true": 1, "y_pred": 1}]
        texts_by_index = {0: "flood in Velachery"}
        geoparse_fn = _fake_geoparse({"flood in Velachery": [{"text": "Velachery", "matched_name": "Velachery"}]})
        metrics = compute_end_to_end_metrics(fold_results, texts_by_index, GAZETTEER, geoparse_fn=geoparse_fn)
        assert metrics["tp"] == 1
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_false_negative_when_distress_but_unresolvable(self):
        # classifier correctly says distress, but no location -> pipeline can't deliver it -> counted against recall
        fold_results = [{"held_out_index": 0, "y_true": 1, "y_pred": 1}]
        texts_by_index = {0: "we are stranded, please help"}
        geoparse_fn = _fake_geoparse({"we are stranded, please help": []})
        metrics = compute_end_to_end_metrics(fold_results, texts_by_index, GAZETTEER, geoparse_fn=geoparse_fn)
        assert metrics["fn"] == 1
        assert metrics["recall"] == 0.0

    def test_true_negative_when_not_distress_and_unresolved(self):
        fold_results = [{"held_out_index": 0, "y_true": 0, "y_pred": 0}]
        texts_by_index = {0: "routine announcement"}
        geoparse_fn = _fake_geoparse({"routine announcement": []})
        metrics = compute_end_to_end_metrics(fold_results, texts_by_index, GAZETTEER, geoparse_fn=geoparse_fn)
        assert metrics["tn"] == 1

    def test_false_positive_when_classifier_wrong_but_resolved(self):
        fold_results = [{"held_out_index": 0, "y_true": 0, "y_pred": 1}]
        texts_by_index = {0: "storm water drains in Velachery"}
        geoparse_fn = _fake_geoparse({"storm water drains in Velachery": [{"text": "Velachery", "matched_name": "Velachery"}]})
        metrics = compute_end_to_end_metrics(fold_results, texts_by_index, GAZETTEER, geoparse_fn=geoparse_fn)
        assert metrics["fp"] == 1
        assert metrics["precision"] == 0.0

    def test_per_example_rows_include_held_out_index(self):
        fold_results = [{"held_out_index": 5, "y_true": 1, "y_pred": 1}]
        texts_by_index = {5: "flood in Velachery"}
        geoparse_fn = _fake_geoparse({"flood in Velachery": [{"text": "Velachery", "matched_name": "Velachery"}]})
        metrics = compute_end_to_end_metrics(fold_results, texts_by_index, GAZETTEER, geoparse_fn=geoparse_fn)
        assert metrics["per_example"][0]["held_out_index"] == 5
        assert metrics["per_example"][0]["n_resolved_locations"] == 1
