"""
Task 4.1 -- train GraphSAGE + temporal layer (A3TGCN) on the real fused
ground truth (task 3.4), evaluated with task 3.7's ward-based split. The
"core experiment" working.md SS1.6 sets up: this model vs. task 3.6's
rule-based baseline, on the exact same transitions/metrics.

Temporal framing -- confirmed with the user before implementing, not
assumed (no spec exists in working.md for this): the 3 usable transitions
(task 2.7) each have a DIFFERENT amount of real history available (1, 2,
3 prior phases respectively), and A3TGCN needs a fixed `periods` at
construction time. Rather than pad the shorter transitions with fake
history or train 3 separate single-example models (severe overfitting --
each transition is one graph snapshot), this trains ONE A3TGCN(periods=1)
with SHARED weights, treating each transition as an independent
(X_t -> Y_t+1) example pooled together for training -- matching task
3.6's baseline and task 2.8's transition_pairs() design exactly, so the
comparison is apples-to-apples.

Honest tradeoff, disclosed not hidden: with periods=1, A3TGCN's attention-
over-history mechanism is degenerate (attending over a single period is a
no-op). This run mainly tests whether the underlying spatial graph
convolution (GraphSAGE-style neighbor aggregation, A3TGCN's real
contribution here) can learn flood propagation from elevation/slope/
distance_to_drain/rainfall -- not genuine multi-step temporal memory, which
this dataset (one event, 4 phases) is too short to support meaningfully.

Class imbalance: pooled training labels are heavily skewed (~87% positive
at peak/receding per task 3.4, 0% at pre_event/rising) -- BCEWithLogitsLoss
uses pos_weight computed from the actual pooled TRAINING split's class
balance, not an assumed 1:1.

Semi-supervised transductive training: message passing sees every
segment's features via the full frozen edge_index (task 3.5) regardless
of split; the loss is computed ONLY on task 3.7's train-split segments'
labels. val/test segments' features are visible to the graph convolution
(as neighbors) but their labels never contribute to a gradient.

Usage:
    python src/models/gnn/train_gnn.py

Required inputs:
    data/processed/ground_truth/fused_flood_labels.csv   (task 3.4)
    data/processed/model_input/segment_splits.csv          (task 3.7)
    (task 2.7/2.8's model input + loader, loaded via load_dataset())

Outputs (data/processed/ground_truth/):
    gnn_training_curve.csv            -- epoch, loss
    gnn_evaluation_report.json         -- per-transition x per-split metrics,
                                          directly comparable to task 3.6's
                                          baseline_evaluation_report.json
    gnn_model.pt                        -- trained weights (gitignored, *.pt)
"""
import json
import sys
import types
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# torch_geometric_temporal import shim -- see src/models/gnn/README.md and
# toy_gnn_prototype.py (task 1.12) for the full explanation; reused
# verbatim, not reinvented.
if "torch_sparse" not in sys.modules:
    import torch_geometric.typing as pyg_typing
    _stub = types.ModuleType("torch_sparse")
    _stub.SparseTensor = pyg_typing.SparseTensor
    sys.modules["torch_sparse"] = _stub

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch_geometric_temporal.nn.recurrent import A3TGCN  # noqa: E402

from src.models.baseline.rule_based_propagation import binary_classification_metrics  # noqa: E402
from src.models.gnn.build_graph_snapshot_dataset import Phase2DatasetError, load_dataset  # noqa: E402

FUSED_LABELS_PATH = REPO_ROOT / "data" / "processed" / "ground_truth" / "fused_flood_labels.csv"
SEGMENT_SPLITS_PATH = REPO_ROOT / "data" / "processed" / "model_input" / "segment_splits.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "ground_truth"

EPOCHS = 300
LEARNING_RATE = 0.01
SEED = 42


class Phase4TrainingError(FileNotFoundError):
    """Raised when a required task 3.4/3.7 artifact is missing. Matches
    the *ArtifactError convention used across src/graph, src/ground_truth,
    src/nlp, src/models/gnn."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase4TrainingError(
            f"Required artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first."
        )
    return path


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_labels_for_attach(path: Path = FUSED_LABELS_PATH) -> pd.DataFrame:
    _require_file(path, "src/ground_truth/fuse_flood_labels.py (task 3.4)")
    return pd.read_csv(path)[["segment_id", "phase_id", "flood_label"]]


def load_split_masks(node_order: list, path: Path = SEGMENT_SPLITS_PATH) -> dict:
    """{"train"/"val"/"test": torch.BoolTensor aligned to node_order}."""
    _require_file(path, "src/models/gnn/build_training_harness.py (task 3.7)")
    splits = pd.read_csv(path).set_index("segment_id")["split"]
    splits = splits.reindex(node_order)
    return {name: torch.tensor((splits == name).to_numpy()) for name in ["train", "val", "test"]}


# --------------------------------------------------------------------------
# Training (pure-ish logic, given a dataset/model -- validated by a real run)
# --------------------------------------------------------------------------

def compute_pos_weight(y_values: torch.Tensor) -> float:
    """BCEWithLogitsLoss's pos_weight = n_negative / n_positive, computed
    from the actual pooled training labels -- not assumed 1:1, given the
    real ~87%/0% class skew across transitions (task 3.4)."""
    n_pos = float(y_values.sum())
    n_neg = float(len(y_values) - n_pos)
    return n_neg / n_pos if n_pos > 0 else 1.0


def compute_feature_stats(ds, train_mask: torch.Tensor) -> tuple:
    """(mean, std) per feature, from the TRAIN split ONLY across all
    phases -- avoids leaking val/test statistics into the normalization.
    Real feature scales span very different ranges (elevation ~0-20m,
    length_m ~0-1150m, distance_to_drain_m ~0-2000m, rainfall_t ~0-410mm),
    unnormalized -- feeding those raw into a GRU-gated model (A3TGCN's
    base) starved gradient flow so badly the model's output had EXACTLY
    zero variance across every node in a first real run (verified: same
    constant probability, std=0.0, for all 17,195 segments) -- it learned
    to ignore the inputs entirely rather than differentiate on them.
    """
    train_values = ds.X[:, train_mask, :].reshape(-1, ds.X.shape[-1])
    mean = train_values.mean(dim=0)
    std = train_values.std(dim=0)
    std = torch.where(std > 0, std, torch.ones_like(std))  # guard a hypothetically-constant feature
    return mean, std


def normalize_features(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def train_model(ds, train_mask: torch.Tensor, epochs: int = EPOCHS, lr: float = LEARNING_RATE, device=None) -> tuple:
    """Trains ONE A3TGCN(periods=1) with shared weights across all 3
    pooled transitions (see module docstring). Returns (model, training_curve).

    pos_weight is computed PER TRANSITION, not pooled across all 3: an
    earlier version pooled every transition's training labels into one
    global pos_weight, but pre_event->rising is 100% negative (task 3.4 --
    both are always dry) while rising->peak/peak->receding are ~93%
    positive -- pooling those blends into a misleading, diluted weight
    (~0.6, favoring negative predictions) that isn't correctly calibrated
    for either sub-task. That version's model collapsed to a trivial
    always-negative predictor (0% recall on the two transitions that
    matter). Weighting each transition's own loss term by its own
    train-split class balance fixes this.

    Returns (model, training_curve, (feature_mean, feature_std)) -- the
    normalization stats must travel with the model, since evaluation has
    to apply the exact same train-derived normalization (see
    compute_feature_stats()).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    feature_mean, feature_std = compute_feature_stats(ds, train_mask)
    feature_mean, feature_std = feature_mean.to(device), feature_std.to(device)

    pairs = list(ds.transition_pairs())
    pos_weights = [torch.tensor(compute_pos_weight(y[train_mask]), device=device) for _, y in pairs]

    model = A3TGCN(in_channels=ds.num_features, out_channels=1, periods=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_mask = train_mask.to(device)

    curve = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        for (x_t, y_t1), pos_weight in zip(pairs, pos_weights):
            X = normalize_features(x_t.x.to(device), feature_mean, feature_std).unsqueeze(-1)
            edge_index = x_t.edge_index.to(device)
            y_true = y_t1.to(device)

            logits = model(X, edge_index)
            loss = F.binary_cross_entropy_with_logits(
                logits[train_mask], y_true[train_mask], pos_weight=pos_weight
            )
            loss.backward()
            total_loss += loss.detach().item()
        optimizer.step()
        curve.append({"epoch": epoch, "loss": total_loss / len(pairs)})
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch}: loss={total_loss / len(pairs):.4f}")

    return model, curve, (feature_mean, feature_std)


# --------------------------------------------------------------------------
# Evaluation (reuses task 3.6's metrics, not reimplemented)
# --------------------------------------------------------------------------

def predict_probabilities(model, ds, feature_stats: tuple, phase_id_to_name: dict, device=None) -> dict:
    """{transition_label: (y_true[N,1], y_prob[N,1])} for every usable
    transition -- factored out of evaluate_model() so task 4.2's
    threshold tuning can sweep thresholds against the same raw
    probabilities without re-running the model."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_mean, feature_std = (t.to(device) for t in feature_stats)
    model.eval()

    results = {}
    with torch.no_grad():
        for transition in ds.schema["y_t1_contract"]["usable_transitions"]:
            x_t_phase_id, y_t1_phase_id = transition["x_t_phase_id"], transition["y_t1_phase_id"]
            phase_pos = {p["phase_id"]: t for t, p in enumerate(ds.phases)}
            x_t = ds[phase_pos[x_t_phase_id]]
            y_true = ds[phase_pos[y_t1_phase_id]].y

            X = normalize_features(x_t.x.to(device), feature_mean, feature_std).unsqueeze(-1)
            edge_index = x_t.edge_index.to(device)
            logits = model(X, edge_index)
            y_prob = torch.sigmoid(logits).cpu()

            label = f"{phase_id_to_name[x_t_phase_id]}->{phase_id_to_name[y_t1_phase_id]}"
            results[label] = (y_true, y_prob)
    return results


def evaluate_model(model, ds, split_masks: dict, phase_id_to_name: dict, feature_stats: tuple, device=None, threshold: float = 0.5) -> dict:
    """`feature_stats`: the SAME (mean, std) train_model() returned --
    evaluation must normalize with the train-derived stats, never
    statistics computed from val/test (that would leak split information
    into the "fixed" preprocessing). `threshold`: task 4.2 may tune this
    away from the naive 0.5 default."""
    probabilities = predict_probabilities(model, ds, feature_stats, phase_id_to_name, device=device)

    per_transition = {}
    for label, (y_true, y_prob) in probabilities.items():
        y_pred = (y_prob > threshold).float()
        per_split = {}
        for split_name, mask in split_masks.items():
            if mask.sum() == 0:
                continue
            per_split[split_name] = binary_classification_metrics(
                y_true[mask].squeeze(-1).tolist(), y_pred[mask].squeeze(-1).tolist()
            )
        per_transition[label] = per_split

    return per_transition


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"\nLoading task 2.7/2.8 model input + loader ...")
    try:
        ds = load_dataset()
        labels = load_labels_for_attach()
        ds.attach_labels(labels)
        split_masks = load_split_masks(ds.node_order)
    except (Phase4TrainingError, Phase2DatasetError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> {ds.num_nodes} nodes, {ds.num_features} features, "
          f"train/val/test = {int(split_masks['train'].sum())}/{int(split_masks['val'].sum())}/{int(split_masks['test'].sum())}")

    print(f"\nTraining A3TGCN(periods=1), {EPOCHS} epochs, shared weights across all 3 pooled transitions ...")
    model, curve, feature_stats = train_model(ds, split_masks["train"], device=device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curve_path = OUT_DIR / "gnn_training_curve.csv"
    pd.DataFrame(curve).to_csv(curve_path, index=False)
    print(f"\nSaved training curve -> {curve_path}")

    model_path = OUT_DIR / "gnn_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights -> {model_path}")

    stats_path = OUT_DIR / "gnn_feature_normalization_stats.json"
    feature_mean, feature_std = feature_stats
    stats_path.write_text(json.dumps({
        "mean": feature_mean.cpu().tolist(), "std": feature_std.cpu().tolist(),
    }, indent=2), encoding="utf-8")
    print(f"Saved feature normalization stats -> {stats_path} (must be reused for any future inference)")

    print("\nEvaluating per transition x per split (task 3.6's metrics, for direct comparison) ...")
    phase_id_to_name = {p["phase_id"]: p["phase_name"] for p in ds.phases}
    report = evaluate_model(model, ds, split_masks, phase_id_to_name, feature_stats, device=device)
    for label, per_split in report.items():
        print(f"  {label}: " + ", ".join(f"{s}_f1={m['f1']}" for s, m in per_split.items()))

    report_path = OUT_DIR / "gnn_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved evaluation report -> {report_path}")

    print("\n=== Full evaluation ===")
    print(json.dumps(report, indent=2, default=str))

    print("\nDone. Next: compare this report against task 3.6's baseline_evaluation_report.json")
    print("per transition -- working.md SS1.6's core experiment (task 5.3 formalizes this).")


if __name__ == "__main__":
    main()
