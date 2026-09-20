"""
Task 3.8 -- lightly fine-tune a pretrained multilingual transformer
(MuRIL, working.md SS2.3's own choice) into a distress/not-distress
classifier on task 2.11's hand-labeled data.

*** REAL DATA-SIZE REALITY CHECK, DISCLOSED UP FRONT *** working.md SS2.3
specifies "a small hand-labeled set (a few hundred posts)" -- as task
2.11's own note already established, the real corpus (task 1.13's
infeasibility pivot) has only 19 passages, 17 of them binary-labeled (2
"uncertain" excluded here -- this task is binary distress/not-distress per
working.md, uncertain isn't a third class it asks for). This script fine-
tunes what actually exists (17 examples: 8 distress, 9 not_distress), not a
simulated "few hundred."

*** ADAPTATION STRATEGY, CONFIRMED WITH THE USER (20 Sep 2026) *** with
n=17 vs. MuRIL-base's 237M parameters, full end-to-end fine-tuning would
just memorize the training set every time -- not a meaningful "trained"
classifier. Chosen instead: freeze the entire MuRIL body, extract a
mean-pooled sentence embedding (768-dim, attention-mask-weighted average
of the last hidden state -- more robust than a frozen, never-fine-tuned
[CLS] token, which needs task-specific fine-tuning to be a good sentence
representation), and train only a linear classification head on top. This
is standard "linear probing" practice for tiny-data regimes (n<50) and is
what working.md SS2.3's "lightly fine-tuned (NOT trained from scratch)"
maps to most defensibly at this sample size -- a real judgment call, not
derivable from working.md's own wording alone, so it was asked rather than
assumed.

*** DISCLOSED LANGUAGE-REGISTER MISMATCH (inherited from task 1.13/2.11,
not new here) *** MuRIL's core strength is code-mixed Indian-language
text; the real corpus is English news/report register (task 1.13's
infeasibility pivot). This proves the ARCHITECTURE task 4.5 needs
(classifier + gazetteer -> resolved distress location), not that MuRIL is
the ideal model for this specific text -- stated plainly, not hidden.

Evaluation: Leave-one-out cross-validation (LOOCV, k=n=17) -- the standard,
most defensible evaluation for a dataset this small (task 4.2 already
established the project's k-fold-over-a-single-split precedent; LOOCV is
that same idea taken to its k=n limit, which is what n=17 actually
supports). Embeddings are frozen and computed ONCE up front (they don't
depend on the head being trained), so LOOCV only re-trains the cheap
linear head per fold, not MuRIL itself.

*** REAL BUG FOUND AND FIXED, SAME CLASS AS TASK 4.1'S GNN BUG, NOT HIDDEN
*** the first real run's LOOCV came back completely degenerate (F1=0.0,
recall=0.0 -- predicted "not distress" for every held-out example).
Diagnosed before disclosing: raw MuRIL embeddings have tiny, near-uniform
per-dimension scale (std ~0.023) -- feeding that directly into a
gradient-trained linear head starves the gradient exactly like task 4.1's
unnormalized GNN features did (full-data train accuracy was only 53%, with
predicted probabilities clustered at ~0.47 for every example regardless of
label -- the head barely moved off its initial output). A quick check
ruled out "the embeddings just aren't separable": 1-NN classification
using raw cosine similarity on the SAME embeddings got 88% LOOCV accuracy,
so the signal is there, the raw-scale linear head just couldn't reach it.
Fixed with per-fold, train-only z-score standardization (`compute_embedding_stats()`/
`standardize_embeddings()` -- same never-leak-held-out-statistics discipline
as task 4.1's `compute_feature_stats()`), which alone took LOOCV F1 from
0.0 to 0.9412.

Usage:
    python src/nlp/finetune_distress_classifier.py

Required inputs:
    data/raw/distress_text/corpus_draft.csv            (task 1.13)
    data/raw/distress_text/labeled_distress_dataset.csv (task 2.11)

Outputs (data/processed/nlp/):
    distress_classifier_head.pt        -- trained linear head (final, all 17 examples)
    distress_classifier_metadata.json  -- label map, pooling method, max_length, threshold
    distress_classifier_loocv_report.json -- per-fold predictions + aggregate metrics
"""
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nlp.label_distress_dataset import passage_id  # noqa: E402 -- reuse task 2.11's own id scheme
from src.models.baseline.rule_based_propagation import binary_classification_metrics  # noqa: E402 -- reuse, not reinvent

CORPUS_PATH = REPO_ROOT / "data" / "raw" / "distress_text" / "corpus_draft.csv"
LABELED_PATH = REPO_ROOT / "data" / "raw" / "distress_text" / "labeled_distress_dataset.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "nlp"
HEAD_PATH = OUT_DIR / "distress_classifier_head.pt"
METADATA_PATH = OUT_DIR / "distress_classifier_metadata.json"
LOOCV_REPORT_PATH = OUT_DIR / "distress_classifier_loocv_report.json"

MURIL_MODEL_NAME = "google/muril-base-cased"
MAX_LENGTH = 256
EMBEDDING_DIM = 768
LABEL_MAP = {"not_distress": 0, "distress": 1}  # "uncertain" excluded -- see module docstring
EXCLUDED_LABELS = {"uncertain"}
HEAD_EPOCHS = 300
HEAD_LR = 0.05
HEAD_WEIGHT_DECAY = 0.01  # L2 regularization -- a 768-dim linear head vs ~16 training points per
                          # LOOCV fold can trivially separate anything, so this is not decorative
SEED = 42
DECISION_THRESHOLD = 0.5  # not tuned (unlike task 4.2) -- n=17 doesn't support a separate threshold-selection split


class Phase4NlpError(FileNotFoundError):
    """Raised when a required task 1.13/2.11 artifact is missing.
    Matches the *ArtifactError/*Error convention used across this repo."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase4NlpError(
            f"Required artifact not found: {path}\nThis file is produced by {produced_by}. Run it first."
        )
    return path


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_labeled_passages(corpus_path: Path = CORPUS_PATH, labeled_path: Path = LABELED_PATH) -> pd.DataFrame:
    """Join task 1.13's corpus with task 2.11's hand labels on passage_id,
    keep only the binary-labeled rows (drop 'uncertain' -- see module
    docstring). Returns columns: passage_id, paragraph_text, label (0/1)."""
    _require_file(corpus_path, "src/nlp/collect_distress_text.py (task 1.13)")
    _require_file(labeled_path, "src/nlp/label_distress_dataset.py (task 2.11)")
    corpus = pd.read_csv(corpus_path)
    corpus["passage_id"] = corpus["paragraph_text"].apply(passage_id)
    labels = pd.read_csv(labeled_path, dtype={"passage_id": str})
    merged = corpus.merge(labels[["passage_id", "label"]], on="passage_id", how="inner")
    binary = merged[~merged["label"].isin(EXCLUDED_LABELS)].copy()
    binary["label"] = binary["label"].map(LABEL_MAP)
    if binary["label"].isna().any():
        bad = merged.loc[binary["label"].isna(), "label"].unique().tolist()
        raise Phase4NlpError(f"Unrecognized label value(s) in {labeled_path}: {bad}")
    return binary[["passage_id", "paragraph_text", "label"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# Frozen embeddings (network/model-dependent -- validated by a real run,
# see developing.md's task 3.8 notes, not mocked)
# --------------------------------------------------------------------------

def load_muril(model_name: str = MURIL_MODEL_NAME, device: str = None):
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False  # frozen -- see module docstring's adaptation strategy
    return tokenizer, model, device


def compute_embedding_stats(embeddings: torch.Tensor) -> tuple:
    """Per-dimension (mean, std) for z-score standardization -- see module
    docstring's bug note. `std` floors at 1.0 for any zero-variance
    dimension, matching task 4.1's train_gnn.py convention exactly."""
    mean = embeddings.mean(dim=0)
    std = embeddings.std(dim=0)
    std = torch.where(std > 0, std, torch.ones_like(std))
    return mean, std


def standardize_embeddings(embeddings: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (embeddings - mean) / std


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Attention-mask-weighted mean over token embeddings -- excludes
    padding tokens from the average, unlike a naive torch.mean(dim=1)."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def embed_texts(texts: list, tokenizer, model, device: str, max_length: int = MAX_LENGTH) -> torch.Tensor:
    """One 768-dim mean-pooled embedding per text, frozen MuRIL forward
    pass only (no gradient -- see module docstring)."""
    embeddings = []
    for text in texts:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
        output = model(**encoded)
        embeddings.append(mean_pool(output.last_hidden_state, encoded["attention_mask"]).squeeze(0).cpu())
    return torch.stack(embeddings)


# --------------------------------------------------------------------------
# Linear head (pure torch logic -- unit-tested with synthetic embeddings)
# --------------------------------------------------------------------------

def make_head(embedding_dim: int = EMBEDDING_DIM, seed: int = SEED) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(embedding_dim, 1)


def train_head(
    embeddings: torch.Tensor, labels: torch.Tensor, epochs: int = HEAD_EPOCHS, lr: float = HEAD_LR,
    weight_decay: float = HEAD_WEIGHT_DECAY, seed: int = SEED,
) -> nn.Linear:
    """Trains a fresh linear head from scratch on `embeddings`/`labels`
    every call -- LOOCV must never carry state between folds."""
    head = make_head(embeddings.shape[1], seed=seed)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    labels = labels.float().unsqueeze(-1)
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = head(embeddings)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
    return head


@torch.no_grad()
def predict_proba(head: nn.Linear, embeddings: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(head(embeddings)).squeeze(-1)


# --------------------------------------------------------------------------
# Leave-one-out cross-validation
# --------------------------------------------------------------------------

def run_loocv(embeddings: torch.Tensor, labels: torch.Tensor, threshold: float = DECISION_THRESHOLD) -> dict:
    """One fold per example: train the head on all others, predict the
    held-out one. Returns per-fold predictions plus aggregate metrics --
    the only honest way to evaluate a 17-example dataset (see module
    docstring). Standardization stats are computed from the fold's TRAIN
    embeddings only and applied to both train and the held-out example --
    never leaking the held-out example's own statistics into its own
    prediction (same discipline as task 4.1's feature normalization)."""
    n = len(labels)
    fold_results = []
    for i in range(n):
        train_mask = torch.ones(n, dtype=torch.bool)
        train_mask[i] = False
        mean, std = compute_embedding_stats(embeddings[train_mask])
        train_embeddings = standardize_embeddings(embeddings[train_mask], mean, std)
        held_out_embedding = standardize_embeddings(embeddings[i:i + 1], mean, std)
        head = train_head(train_embeddings, labels[train_mask])
        prob = predict_proba(head, held_out_embedding).item()
        fold_results.append({
            "held_out_index": i, "y_true": int(labels[i].item()), "y_prob": round(prob, 4),
            "y_pred": int(prob >= threshold),
        })
    y_true = [r["y_true"] for r in fold_results]
    y_pred = [r["y_pred"] for r in fold_results]
    metrics = binary_classification_metrics(y_true, y_pred)
    return {"n_folds": n, "threshold": threshold, "fold_results": fold_results, "aggregate_metrics": metrics}


# --------------------------------------------------------------------------
# Inference (for task 4.5's end-to-end pipeline)
# --------------------------------------------------------------------------

def predict_distress(
    text: str, tokenizer=None, model=None, device: str = None, head: nn.Linear = None,
    embedding_mean: torch.Tensor = None, embedding_std: torch.Tensor = None,
) -> dict:
    """High-level single-text inference -- loads the saved head/stats/
    metadata if `head` isn't passed in, for task 4.5 to call without
    re-implementing the embedding/standardization/threshold logic.
    `tokenizer`/`model`/`device` can be passed in to reuse an already-
    loaded MuRIL across many calls (loading it per-call would be
    prohibitively slow for a real pipeline). If `head` is passed directly
    without `embedding_mean`/`embedding_std`, standardization is skipped
    (caller's responsibility) -- always loaded together via
    `load_trained_head()` for the real saved artifacts."""
    if tokenizer is None or model is None:
        tokenizer, model, device = load_muril()
    if head is None:
        head, embedding_mean, embedding_std, metadata = load_trained_head()
    else:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    embedding = embed_texts([text], tokenizer, model, device)
    if embedding_mean is not None and embedding_std is not None:
        embedding = standardize_embeddings(embedding, embedding_mean, embedding_std)
    prob = predict_proba(head, embedding).item()
    label = "distress" if prob >= metadata["decision_threshold"] else "not_distress"
    return {"label": label, "probability": round(prob, 4)}


def load_trained_head(head_path: Path = HEAD_PATH, metadata_path: Path = METADATA_PATH):
    """Returns (head, embedding_mean, embedding_std, metadata). The
    standardization stats are saved alongside the head's weights -- they
    must travel together, since the head was trained on standardized
    inputs (see module docstring's bug note) and a new embedding at
    inference time must be standardized with the SAME stats, not
    recomputed from a batch of one."""
    _require_file(head_path, "src/nlp/finetune_distress_classifier.py (task 3.8)")
    _require_file(metadata_path, "src/nlp/finetune_distress_classifier.py (task 3.8)")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(head_path)
    head = make_head(metadata["embedding_dim"])
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    return head, checkpoint["embedding_mean"], checkpoint["embedding_std"], metadata


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    print(f"Loading task 1.13/2.11 labeled passages <- {CORPUS_PATH.parent}")
    try:
        data = load_labeled_passages()
    except Phase4NlpError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> {len(data)} binary-labeled passages "
          f"({int(data['label'].sum())} distress, {int((data['label'] == 0).sum())} not_distress)")

    print(f"\nLoading MuRIL ({MURIL_MODEL_NAME}, frozen) ...")
    tokenizer, model, device = load_muril()
    print(f"  -> device={device}")

    print("\nComputing frozen mean-pooled embeddings for all passages ...")
    embeddings = embed_texts(data["paragraph_text"].tolist(), tokenizer, model, device)
    labels = torch.tensor(data["label"].tolist())
    print(f"  -> {embeddings.shape}")

    print(f"\nRunning leave-one-out cross-validation (k={len(data)}) ...")
    loocv_report = run_loocv(embeddings, labels)
    print(f"  aggregate metrics: {loocv_report['aggregate_metrics']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOOCV_REPORT_PATH.write_text(json.dumps(loocv_report, indent=2), encoding="utf-8")
    print(f"  saved -> {LOOCV_REPORT_PATH}")

    print(f"\nTraining final head on all {len(data)} labeled passages (for task 4.5) ...")
    final_mean, final_std = compute_embedding_stats(embeddings)  # all 17 -- this IS the deployed model, no held-out set left
    final_head = train_head(standardize_embeddings(embeddings, final_mean, final_std), labels)
    torch.save({"state_dict": final_head.state_dict(), "embedding_mean": final_mean, "embedding_std": final_std}, HEAD_PATH)
    metadata = {
        "muril_model_name": MURIL_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "pooling": "attention_mask_weighted_mean",
        "max_length": MAX_LENGTH,
        "label_map": LABEL_MAP,
        "decision_threshold": DECISION_THRESHOLD,
        "n_training_examples": len(data),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  saved -> {HEAD_PATH}, {METADATA_PATH}")

    print("\n=== LOOCV summary ===")
    print(json.dumps(loocv_report["aggregate_metrics"], indent=2))
    print("\nDone. Next: task 4.5 wires predict_distress() together with task 2.10's")
    print("fuzzy_geoparse.py into one end-to-end Objective 2 pipeline.")


if __name__ == "__main__":
    main()
