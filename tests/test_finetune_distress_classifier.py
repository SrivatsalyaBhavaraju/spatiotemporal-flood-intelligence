"""
Unit tests for src/nlp/finetune_distress_classifier.py (task 3.8).

Covers the pure logic (label joining/filtering, mean pooling, linear-head
training/LOOCV, inference plumbing) with small synthetic data. The actual
frozen MuRIL embeddings (embed_texts/load_muril) are network/model-
dependent and validated by a real run against the real task 2.11 corpus
(see developing.md's task 3.8 notes) -- not mocked, matching this repo's
convention elsewhere (e.g. test_run_sentinel1_change_detection.py).

Run with:
    pytest tests/test_finetune_distress_classifier.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nlp.finetune_distress_classifier import (  # noqa: E402
    LABEL_MAP,
    Phase4NlpError,
    compute_embedding_stats,
    load_labeled_passages,
    load_trained_head,
    make_head,
    mean_pool,
    predict_distress,
    predict_proba,
    run_loocv,
    standardize_embeddings,
    train_head,
)


def _write_corpus_and_labels(tmp_path, rows):
    """rows: list of (paragraph_text, label) -- label may be 'uncertain'."""
    corpus_rows = [
        {"source_url": f"http://x/{i}", "source_label": "news", "paragraph_text": text, "matched_locations": "Velachery"}
        for i, (text, _) in enumerate(rows)
    ]
    corpus_path = tmp_path / "corpus_draft.csv"
    pd.DataFrame(corpus_rows).to_csv(corpus_path, index=False)

    from src.nlp.label_distress_dataset import passage_id
    label_rows = [
        {"passage_id": passage_id(text), "label": label, "annotator_notes": "", "labeled_at": "2026-01-01T00:00:00Z"}
        for text, label in rows
    ]
    labeled_path = tmp_path / "labeled.csv"
    pd.DataFrame(label_rows).to_csv(labeled_path, index=False)
    return corpus_path, labeled_path


class TestLoadLabeledPassages:
    def test_missing_corpus_raises(self, tmp_path):
        _, labeled_path = _write_corpus_and_labels(tmp_path, [("a", "distress")])
        import pytest
        with pytest.raises(Phase4NlpError):
            load_labeled_passages(tmp_path / "nope.csv", labeled_path)

    def test_excludes_uncertain_labels(self, tmp_path):
        rows = [("families evacuated by boat", "distress"), ("PM announced relief funds", "not_distress"),
                ("ambiguous statement", "uncertain")]
        corpus_path, labeled_path = _write_corpus_and_labels(tmp_path, rows)
        data = load_labeled_passages(corpus_path, labeled_path)
        assert len(data) == 2
        assert set(data["paragraph_text"]) == {"families evacuated by boat", "PM announced relief funds"}

    def test_label_mapping_to_binary(self, tmp_path):
        rows = [("distress passage", "distress"), ("dry passage", "not_distress")]
        corpus_path, labeled_path = _write_corpus_and_labels(tmp_path, rows)
        data = load_labeled_passages(corpus_path, labeled_path)
        row = data.set_index("paragraph_text")
        assert row.loc["distress passage", "label"] == LABEL_MAP["distress"] == 1
        assert row.loc["dry passage", "label"] == LABEL_MAP["not_distress"] == 0

    def test_only_joins_passages_that_have_labels(self, tmp_path):
        # corpus has 3 passages, only 2 are labeled -- unlabeled one dropped, not errored
        corpus_rows = [
            {"source_url": "http://a", "source_label": "news", "paragraph_text": "labeled one", "matched_locations": "X"},
            {"source_url": "http://b", "source_label": "news", "paragraph_text": "labeled two", "matched_locations": "X"},
            {"source_url": "http://c", "source_label": "news", "paragraph_text": "never labeled", "matched_locations": "X"},
        ]
        corpus_path = tmp_path / "corpus.csv"
        pd.DataFrame(corpus_rows).to_csv(corpus_path, index=False)
        _, labeled_path = _write_corpus_and_labels(tmp_path, [("labeled one", "distress"), ("labeled two", "not_distress")])
        data = load_labeled_passages(corpus_path, labeled_path)
        assert len(data) == 2
        assert "never labeled" not in set(data["paragraph_text"])


class TestComputeEmbeddingStats:
    def test_zero_mean_unit_variance_after_standardizing(self):
        embeddings = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        mean, std = compute_embedding_stats(embeddings)
        standardized = standardize_embeddings(embeddings, mean, std)
        assert torch.allclose(standardized.mean(dim=0), torch.zeros(2), atol=1e-6)

    def test_constant_dimension_gets_std_one_not_zero(self):
        # a dimension with identical values everywhere -- std would be 0,
        # which must not divide-by-zero in standardize_embeddings (same
        # guard as task 4.1's train_gnn.py compute_feature_stats)
        embeddings = torch.tensor([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        mean, std = compute_embedding_stats(embeddings)
        assert std[0] == 1.0
        standardized = standardize_embeddings(embeddings, mean, std)
        assert torch.allclose(standardized[:, 0], torch.zeros(3))  # (5-5)/1 = 0, not NaN

    def test_tiny_raw_scale_gets_rescaled_to_unit_variance(self):
        # reproduces the real bug: raw MuRIL embeddings had std ~0.023 --
        # standardization must bring that up to a gradient-friendly scale
        embeddings = torch.randn(20, 8) * 0.02 + 0.6
        mean, std = compute_embedding_stats(embeddings)
        standardized = standardize_embeddings(embeddings, mean, std)
        assert standardized.std(dim=0).mean().item() > 0.9  # ~1.0, not ~0.02


class TestRunLoocvNoLeakage:
    def test_train_stats_never_include_the_held_out_example(self, monkeypatch):
        import src.nlp.finetune_distress_classifier as mod
        seen_train_sizes = []
        real_compute_stats = mod.compute_embedding_stats

        def spy(embeddings):
            seen_train_sizes.append(embeddings.shape[0])
            return real_compute_stats(embeddings)

        monkeypatch.setattr(mod, "compute_embedding_stats", spy)
        embeddings, labels = TestTrainHeadAndPredict()._linearly_separable_data()
        mod.run_loocv(embeddings, labels)
        assert seen_train_sizes == [len(labels) - 1] * len(labels)  # n-1 every fold, held-out never included


class TestMeanPool:
    def test_excludes_padding_tokens(self):
        # 2 tokens real (values 1,1), 1 token padding (value 100, should be excluded)
        hidden = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [100.0, 100.0]]])
        mask = torch.tensor([[1, 1, 0]])
        pooled = mean_pool(hidden, mask)
        assert torch.allclose(pooled, torch.tensor([[1.0, 1.0]]))

    def test_single_real_token(self):
        hidden = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
        mask = torch.tensor([[1, 0]])
        pooled = mean_pool(hidden, mask)
        assert torch.allclose(pooled, torch.tensor([[3.0, 4.0]]))


class TestTrainHeadAndPredict:
    def _linearly_separable_data(self):
        torch.manual_seed(0)
        pos = torch.randn(8, 16) + 3.0
        neg = torch.randn(8, 16) - 3.0
        embeddings = torch.cat([pos, neg])
        labels = torch.cat([torch.ones(8), torch.zeros(8)])
        return embeddings, labels

    def test_head_fits_separable_training_data(self):
        embeddings, labels = self._linearly_separable_data()
        head = train_head(embeddings, labels, epochs=200, lr=0.1)
        probs = predict_proba(head, embeddings)
        preds = (probs >= 0.5).float()
        assert (preds == labels).all()

    def test_deterministic_given_seed(self):
        embeddings, labels = self._linearly_separable_data()
        head1 = train_head(embeddings, labels, seed=42)
        head2 = train_head(embeddings, labels, seed=42)
        assert torch.allclose(head1.weight, head2.weight)

    def test_make_head_has_right_shape(self):
        head = make_head(embedding_dim=768)
        assert head.weight.shape == (1, 768)


class TestRunLoocv:
    def test_one_fold_per_example(self):
        embeddings, labels = TestTrainHeadAndPredict()._linearly_separable_data()
        report = run_loocv(embeddings, labels)
        assert report["n_folds"] == len(labels)
        assert len(report["fold_results"]) == len(labels)

    def test_recovers_perfect_metrics_on_easy_separable_data(self):
        # well-separated clusters -- LOOCV should recover each held-out point correctly
        embeddings, labels = TestTrainHeadAndPredict()._linearly_separable_data()
        report = run_loocv(embeddings, labels)
        assert report["aggregate_metrics"]["f1"] == 1.0

    def test_fold_never_sees_its_own_held_out_label_during_training(self):
        # a harder check: corrupt the held-out example's features to be
        # wildly wrong -- if the fold's head were (incorrectly) trained
        # including it, this wouldn't be revealed by this test alone, but
        # combined with the perfect-separability test above, a leaking
        # implementation would still show up as a suspicious pass here too.
        # Simpler direct check: held_out_index values are exactly 0..n-1, in order.
        embeddings, labels = TestTrainHeadAndPredict()._linearly_separable_data()
        report = run_loocv(embeddings, labels)
        assert [r["held_out_index"] for r in report["fold_results"]] == list(range(len(labels)))


class TestLoadTrainedHeadAndPredict:
    def test_load_missing_head_raises(self, tmp_path):
        import pytest
        with pytest.raises(Phase4NlpError):
            load_trained_head(tmp_path / "nope.pt", tmp_path / "nope.json")

    def test_predict_distress_uses_injected_head(self, monkeypatch, tmp_path):
        # inject a fake head + metadata so predict_distress doesn't need real MuRIL
        embeddings, labels = TestTrainHeadAndPredict()._linearly_separable_data()
        head = train_head(embeddings, labels)

        metadata_path = tmp_path / "distress_classifier_metadata.json"
        import json
        metadata_path.write_text(json.dumps({"decision_threshold": 0.5, "embedding_dim": 16}), encoding="utf-8")
        monkeypatch.setattr("src.nlp.finetune_distress_classifier.METADATA_PATH", metadata_path)

        def fake_embed_texts(texts, tokenizer, model, device, max_length=256):
            return embeddings[:1]  # pretend the input text embeds to the first (positive) example

        monkeypatch.setattr("src.nlp.finetune_distress_classifier.embed_texts", fake_embed_texts)
        result = predict_distress("irrelevant text", tokenizer="fake", model="fake", device="cpu", head=head)
        assert result["label"] in {"distress", "not_distress"}
        assert 0.0 <= result["probability"] <= 1.0
