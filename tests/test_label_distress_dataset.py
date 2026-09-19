"""
Unit tests for src/nlp/label_distress_dataset.py (task 2.11).

Simulates a human annotator by injecting a fake `input_fn` into
interactive_session() -- this tests the tool's logic (resumability,
incremental saving, quit/skip/relabel behavior) without needing a real
terminal or, crucially, without this test suite fabricating any actual
judgment about the real corpus content.

Run with:
    pytest tests/test_label_distress_dataset.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nlp.label_distress_dataset import (  # noqa: E402
    Phase2LabelingError,
    build_report,
    interactive_session,
    load_corpus,
    load_existing_labels,
    passage_id,
)


def _make_corpus_csv(tmp_path, rows=None):
    if rows is None:
        rows = [
            {"source_url": "http://a", "source_label": "news", "paragraph_text": "Families were evacuated by boat.", "matched_locations": "Velachery"},
            {"source_url": "http://b", "source_label": "encyclopedia", "paragraph_text": "The PM announced ex gratia payments.", "matched_locations": "Adyar"},
            {"source_url": "http://c", "source_label": "academic", "paragraph_text": "Rainfall patterns in the northeast monsoon.", "matched_locations": "Adyar"},
        ]
    p = tmp_path / "corpus_draft.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


class FakeInput:
    """Feeds a fixed sequence of responses to interactive_session()'s
    input_fn calls, in order, and records every prompt it was asked."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("FakeInput ran out of scripted responses")
        return self.responses.pop(0)


class TestPassageId:
    def test_deterministic_across_calls(self):
        text = "Families were evacuated by boat."
        assert passage_id(text) == passage_id(text)

    def test_different_text_different_id(self):
        assert passage_id("text A") != passage_id("text B")

    def test_stable_regardless_of_surrounding_whitespace(self):
        assert passage_id("hello world") == passage_id("  hello world  ")


class TestLoadCorpus:
    def test_missing_file_raises_actionable_error(self, tmp_path):
        with pytest.raises(Phase2LabelingError, match="collect_distress_text.py"):
            load_corpus(tmp_path / "nope.csv")

    def test_missing_required_column_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        pd.DataFrame({"source_url": ["a"], "paragraph_text": ["x"]}).to_csv(p, index=False)
        with pytest.raises(Phase2LabelingError, match="matched_locations|source_label"):
            load_corpus(p)

    def test_valid_corpus_gets_passage_ids(self, tmp_path):
        p = _make_corpus_csv(tmp_path)
        df = load_corpus(p)
        assert "passage_id" in df.columns
        assert df["passage_id"].is_unique

    def test_duplicate_text_warns_not_raises(self, tmp_path, capsys):
        rows = [
            {"source_url": "http://a", "source_label": "news", "paragraph_text": "same text", "matched_locations": "X"},
            {"source_url": "http://b", "source_label": "news", "paragraph_text": "same text", "matched_locations": "Y"},
        ]
        p = _make_corpus_csv(tmp_path, rows)
        df = load_corpus(p)  # must not raise
        assert len(df) == 2
        captured = capsys.readouterr()
        assert "WARNING" in captured.out


class TestLoadExistingLabels:
    def test_missing_file_returns_empty_with_right_columns(self, tmp_path):
        df = load_existing_labels(tmp_path / "nope.csv")
        assert len(df) == 0
        assert set(["passage_id", "label", "annotator_notes", "labeled_at"]).issubset(df.columns)


class TestInteractiveSession:
    def test_labels_every_passage_in_one_pass(self, tmp_path):
        corpus_path = _make_corpus_csv(tmp_path)
        corpus_df = load_corpus(corpus_path)
        labels_df = load_existing_labels(tmp_path / "labels.csv")

        # 3 passages -> d, n, u ; each followed by an empty notes response
        fake_input = FakeInput(["d", "", "n", "", "u", "note for the ambiguous one"])
        result = interactive_session(corpus_df, labels_df, input_fn=fake_input, output_path=tmp_path / "labels.csv")

        assert len(result) == 3
        assert set(result["label"]) == {"distress", "not_distress", "uncertain"}

    def test_saves_incrementally_not_just_at_the_end(self, tmp_path):
        corpus_path = _make_corpus_csv(tmp_path)
        corpus_df = load_corpus(corpus_path)
        labels_df = load_existing_labels(tmp_path / "labels.csv")
        output_path = tmp_path / "labels.csv"

        seen_row_counts_on_disk = []

        class RecordingFakeInput(FakeInput):
            def __call__(self, prompt):
                if output_path.exists():
                    seen_row_counts_on_disk.append(len(pd.read_csv(output_path)))
                return super().__call__(prompt)

        fake_input = RecordingFakeInput(["d", "", "n", "", "u", ""])
        interactive_session(corpus_df, labels_df, input_fn=fake_input, output_path=output_path)
        # by the time we're asked for the 2nd passage's label, the 1st must already be on disk
        assert any(c >= 1 for c in seen_row_counts_on_disk)
        assert len(pd.read_csv(output_path)) == 3

    def test_already_labeled_passages_are_skipped_on_second_run(self, tmp_path):
        corpus_path = _make_corpus_csv(tmp_path)
        corpus_df = load_corpus(corpus_path)
        output_path = tmp_path / "labels.csv"

        labels_df = load_existing_labels(output_path)
        fake_input = FakeInput(["d", "", "n", "", "u", ""])
        labels_df = interactive_session(corpus_df, labels_df, input_fn=fake_input, output_path=output_path)

        # second run: nothing left to label, input_fn should never be called
        def fail_if_called(prompt):
            raise AssertionError("input_fn should not be called -- everything is already labeled")

        result2 = interactive_session(corpus_df, labels_df, input_fn=fail_if_called, output_path=output_path)
        assert len(result2) == 3

    def test_quit_mid_session_preserves_labels_so_far(self, tmp_path):
        corpus_path = _make_corpus_csv(tmp_path)
        corpus_df = load_corpus(corpus_path)
        labels_df = load_existing_labels(tmp_path / "labels.csv")
        output_path = tmp_path / "labels.csv"

        fake_input = FakeInput(["d", "", "q"])  # label 1st, then quit before 2nd
        result = interactive_session(corpus_df, labels_df, input_fn=fake_input, output_path=output_path)
        assert len(result) == 1
        assert len(pd.read_csv(output_path)) == 1

    def test_skip_leaves_passage_unlabeled(self, tmp_path):
        corpus_path = _make_corpus_csv(tmp_path)
        corpus_df = load_corpus(corpus_path)
        labels_df = load_existing_labels(tmp_path / "labels.csv")
        output_path = tmp_path / "labels.csv"

        fake_input = FakeInput(["s", "d", "", "n", ""])
        result = interactive_session(corpus_df, labels_df, input_fn=fake_input, output_path=output_path)
        assert len(result) == 2  # the skipped one never got a row

    def test_invalid_input_is_rejected_and_reprompts(self, tmp_path):
        corpus_path = _make_corpus_csv(tmp_path, rows=[
            {"source_url": "http://a", "source_label": "news", "paragraph_text": "only one passage", "matched_locations": "X"},
        ])
        corpus_df = load_corpus(corpus_path)
        labels_df = load_existing_labels(tmp_path / "labels.csv")
        output_path = tmp_path / "labels.csv"

        fake_input = FakeInput(["banana", "d", ""])  # garbage first, then valid
        result = interactive_session(corpus_df, labels_df, input_fn=fake_input, output_path=output_path)
        assert len(result) == 1
        assert result.iloc[0]["label"] == "distress"

    def test_relabel_overwrites_existing_label(self, tmp_path):
        corpus_path = _make_corpus_csv(tmp_path)
        corpus_df = load_corpus(corpus_path)
        output_path = tmp_path / "labels.csv"
        labels_df = load_existing_labels(output_path)

        fake_input = FakeInput(["d", "", "n", "", "u", ""])
        labels_df = interactive_session(corpus_df, labels_df, input_fn=fake_input, output_path=output_path)
        target_id = corpus_df.iloc[0]["passage_id"]
        assert labels_df.set_index("passage_id").loc[target_id, "label"] == "distress"

        relabel_input = FakeInput(["n", "changed my mind"])
        labels_df = interactive_session(
            corpus_df, labels_df, only_passage_id=target_id, input_fn=relabel_input, output_path=output_path
        )
        assert labels_df.set_index("passage_id").loc[target_id, "label"] == "not_distress"
        assert len(labels_df) == 3  # still 3 rows total, not 4


class TestBuildReport:
    def test_counts_correct(self, tmp_path):
        corpus_path = _make_corpus_csv(tmp_path)
        corpus_df = load_corpus(corpus_path)
        output_path = tmp_path / "labels.csv"
        labels_df = load_existing_labels(output_path)
        fake_input = FakeInput(["d", "", "n", "", "u", ""])
        labels_df = interactive_session(corpus_df, labels_df, input_fn=fake_input, output_path=output_path)

        report = build_report(corpus_df, labels_df)
        assert report["corpus_total_passages"] == 3
        assert report["labeled_count"] == 3
        assert report["unlabeled_count"] == 0
        assert report["label_counts"] == {"distress": 1, "not_distress": 1, "uncertain": 1}
        assert "few hundred" in report["original_scope_note"]

    def test_partial_labeling_reports_unlabeled_ids(self, tmp_path):
        corpus_path = _make_corpus_csv(tmp_path)
        corpus_df = load_corpus(corpus_path)
        output_path = tmp_path / "labels.csv"
        labels_df = load_existing_labels(output_path)
        fake_input = FakeInput(["d", "", "q"])  # label the first one, then quit before the rest
        labels_df = interactive_session(corpus_df, labels_df, input_fn=fake_input, output_path=output_path)

        report = build_report(corpus_df, labels_df)
        assert report["labeled_count"] == 1
        assert report["unlabeled_count"] == 2
        assert len(report["unlabeled_passage_ids"]) == 2
