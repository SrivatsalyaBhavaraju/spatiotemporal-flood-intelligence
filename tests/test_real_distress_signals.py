"""
Unit tests for src/nlp/real_distress_signals.py and its use in the
dashboard's load_distress_markers() -- the shared "is this a REAL distress
signal?" filter (synthetic smoke-test rows and task 5.5's known geoparsing
false positive excluded).

Run with:
    pytest tests/test_real_distress_signals.py -v
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.dashboard.data_loader as data_loader  # noqa: E402
from src.nlp.real_distress_signals import (  # noqa: E402
    KNOWN_FALSE_POSITIVES,
    is_known_false_positive,
    is_real_corpus_text,
    load_real_corpus_texts,
)


def test_nandambakkam_false_positive_is_registered():
    assert ("nandambakkam", "Adambakkam") in KNOWN_FALSE_POSITIVES


def test_false_positive_needs_both_span_and_place():
    assert is_known_false_positive("Roads to Nandambakkam were cut off.", "Adambakkam")
    assert not is_known_false_positive("Adambakkam lake overflowed.", "Adambakkam")   # genuine mention
    assert not is_known_false_positive("Roads to Nandambakkam were cut off.", "Guindy")


def test_real_corpus_text_matching_strips_whitespace():
    corpus = {"A real passage."}
    assert is_real_corpus_text("  A real passage.\n", corpus)
    assert not is_real_corpus_text("A synthetic smoke test.", corpus)


def test_real_corpus_loads_from_committed_file():
    texts = load_real_corpus_texts()
    assert len(texts) == 19  # task 1.13's real corpus


def test_dashboard_markers_exclude_synthetic_rows_and_false_positive(tmp_path, monkeypatch):
    loc = lambda place: {"place": place, "lon": 80.2, "lat": 13.0, "score": 100.0}  # noqa: E731
    pd.DataFrame([
        {"text": "Real: Velachery residents stranded.", "distress_label": "distress",
         "distress_probability": 0.99, "resolved_locations": json.dumps([loc("Velachery")])},
        {"text": "Real: bridge cut off Guindy and Nandambakkam.", "distress_label": "distress",
         "distress_probability": 0.98, "resolved_locations": json.dumps([loc("Guindy"), loc("Adambakkam")])},
        {"text": "Synthetic: families in Velachery trapped.", "distress_label": "distress",
         "distress_probability": 0.97, "resolved_locations": json.dumps([loc("Velachery")])},
    ]).to_csv(tmp_path / "objective2_pipeline_output.csv", index=False)
    monkeypatch.setattr(data_loader, "NLP_DIR", tmp_path)

    markers = data_loader.load_distress_markers(
        corpus_texts={"Real: Velachery residents stranded.", "Real: bridge cut off Guindy and Nandambakkam."})
    assert sorted(markers["place"]) == ["Guindy", "Velachery"]
    assert not markers["text"].str.startswith("Synthetic").any()
