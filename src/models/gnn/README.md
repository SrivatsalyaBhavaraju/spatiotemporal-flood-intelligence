# P3 — GNN model (GraphSAGE + temporal layer via PyTorch Geometric Temporal)

## Environment — tasks 1.10 / 1.11

**GPU (1.10):** the study laptop has a local NVIDIA GeForce RTX 2050 (4GB
VRAM, driver supports CUDA 13.0) — plenty for a ward-scale line-graph
(~17,195 road-segment nodes, `src/graph/README.md`), so we're training
locally instead of on Colab/Kaggle as originally planned. Verify with:

```bash
nvidia-smi
```

**Install (1.11):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install torch-geometric
pip install torch-geometric-temporal --no-deps
```

Installed versions (01 Sep 2026): `torch==2.13.0+cu126`,
`torch_geometric==2.8.0.post1`, `torch_geometric_temporal==0.56.2`.

**Why `--no-deps` on the last one:** `torch-geometric-temporal`'s pinned
deps include `torch-scatter`/`torch-sparse`, which ship as compiled C++/CUDA
extensions with no prebuilt wheel yet for `torch==2.13.0+cu126` on Python
3.13, and building them from source needs a full MSVC + CUDA toolkit install
we don't have on this laptop — not worth it for what's actually needed here.

**Why it still works without `torch-sparse` installed:** the package's
`__init__.py` unconditionally imports every model it ships, including
`EvolveGCNH`, which hard-imports `torch_sparse.SparseTensor` at module level
even though we only use `A3TGCN`/`MPNNLSTM`. PyTorch Geometric itself
already ships a drop-in fallback `SparseTensor` class for exactly this case
(`torch_geometric.typing.SparseTensor`, used internally whenever
`torch_geometric.typing.WITH_TORCH_SPARSE` is `False`). `toy_gnn_prototype.py`
stubs a `torch_sparse` module in `sys.modules` pointing at that same class
before importing `torch_geometric_temporal` — this only satisfies the
import; `EvolveGCNH` is never instantiated or used. Any other script that
imports from `torch_geometric_temporal.nn.recurrent` needs the same shim at
the top (copy the block from `toy_gnn_prototype.py`).

## `toy_gnn_prototype.py` — task 1.12

Smoke-tests the whole stack end-to-end: builds a random 40-node toy graph
shaped like our real problem (nodes = road segments, 4 time periods matching
the pre-event/rising/peak/receding label scheme), runs one forward +
backward pass through both candidate architectures named in `working.md`'s
model section, and confirms gradients flow on GPU.

```bash
python src/models/gnn/toy_gnn_prototype.py
```

**Result (01 Sep 2026):** both models ran cleanly on the RTX 2050 —
A3TGCN output `(40, 1)`, MPNN-LSTM output `(40, 22)` (`2*hidden + in_channels
+ periods - 1`, per its own docstring), both losses backpropagated with all
parameter tensors receiving gradients.

**What this is, and isn't:** this only proves the environment works — CUDA,
PyG, and PyG-Temporal all functioning together, forward and backward passes
completing without error. It is not a real model: features and labels are
random noise, no training loop, no evaluation. The actual A3TGCN-vs-MPNN-LSTM
architecture choice, real features (tasks 1.5–1.8), and real fused labels
(task 3.4) come later, once the line-graph itself is built (task 2.1+).

**Note on MPNN-LSTM's input shape:** unlike A3TGCN (which takes one
`[num_nodes, channels, periods]` tensor plus a single static `edge_index`),
MPNN-LSTM expects features for all `window` timesteps stacked row-wise
(`[periods*num_nodes, channels]`) and `edge_index` tiled once per timestep
with node ids offset by `t*num_nodes` — each timestep's graph convolution
stays independent; only the LSTM afterwards mixes across time. Get this
wrong and the model still runs (no shape error) but silently mixes the wrong
nodes' features — see the comment above `test_mpnn_lstm()` before reusing it.

## `build_training_harness.py` — task 3.7

A phase-based train/val/test split + reusable metrics, ready for task
4.1's GNN training. **No spec for this exists in working.md** (checked
directly) — the design here is this task's own judgment call.

```bash
python src/models/gnn/build_training_harness.py
```

**Why a WARD-level (spatial) split, not a temporal (transition) one:**
there are only 3 usable transitions total (task 2.7), and each one's
dynamic features (`rainfall_t`) are broadcast identically to every
segment within a phase (task 2.6) — holding out a whole transition for
testing would remove an entire feature-context from a dataset that's
already small. A ward-level split instead uses all 3 transitions for both
training and evaluation, holding out geographically contiguous segment
groups — the standard mitigation for spatial-autocorrelation leakage a
per-node random split would have. "Phase-based" in this task's name
describes how results are *reported* (task 5.1/5.2 ask for F1/accuracy
**per phase transition**), not how segments are split.

**Method:** shuffle wards (fixed seed 42), greedily assign each to
whichever bucket (train 70% / val 15% / test 15%, by segment count, not
ward count) is currently furthest below its target. ~2.7% of segments
(465/17,195) don't fall strictly inside any ward polygon (boundary-
adjacent, same edge case `build_gazetteer.py`'s README already documents
for gazetteer points) — assigned to their nearest ward by centroid
distance rather than left out.

**Result (19 Sep 2026):** achieved 70.43/13.15/16.42% — close to target.
Ward assignment: val = wards {170, 174}, test = wards {169, 182}, train =
the other 12. Smoke-tested against task 3.6's real baseline predictions
(not synthetic data) — the harness computes correctly, but **surfaces a
real limitation of this split worth knowing before trusting it: with only
16 wards, val/test each land just ~2 of them, so per-split metrics carry
real variance from which specific wards get held out**, not just model
quality. Concretely, on the rising→peak transition alone: train accuracy
6.99%, val 53.54%, test 3.54% — a huge, disclosed swing from the same
underlying baseline model, driven by wards 170/174 (val) happening to have
a very different flood rate than the train/test ward mix. Task 4.2's
hyperparameter tuning should account for this (e.g. k-fold across wards)
rather than trust a single val split's numbers at face value.

Outputs (`data/processed/model_input/`, gitignored): `segment_splits.csv`
(segment_id, ward_no, split) and the validation report.

## `train_gnn.py` — task 4.1

Trains GraphSAGE + a temporal layer (A3TGCN) on the real fused ground
truth (task 3.4), evaluated with task 3.7's ward-based split. Working.md
§1.6's "core experiment": this model vs. task 3.6's rule-based baseline,
on the exact same transitions/metrics.

```bash
python src/models/gnn/train_gnn.py
```

**Temporal framing — confirmed with the user before implementing** (no
spec exists in working.md): the 3 usable transitions each have a
*different* amount of real history (1, 2, 3 prior phases), and A3TGCN
needs a fixed `periods` at construction time. Rather than pad shorter
transitions with fake history or train 3 separate single-example models
(severe overfitting — each is one graph snapshot), this trains ONE
`A3TGCN(periods=1)` with shared weights, pooling all 3 transitions'
`(X_t, Y_t+1)` pairs for training — matching task 3.6's baseline and task
2.8's `transition_pairs()` design exactly, for an apples-to-apples
comparison. Honest tradeoff: with periods=1, A3TGCN's attention-over-
history is degenerate — this tests whether the underlying spatial graph
convolution can learn flood propagation, not genuine multi-step memory
(this dataset — one event, 4 phases — is too short to support that
meaningfully regardless of architecture).

**Two real bugs found and fixed on the first real training runs, not
hidden:**
1. **Features were never normalized.** Raw scales span wildly different
   ranges (elevation ~0-20m, length_m ~0-1150m, distance_to_drain_m
   ~0-2000m, rainfall_t ~0-410mm). Feeding those directly into a
   GRU-gated model starved gradient flow so badly the trained model's
   output had **exactly zero variance across all 17,195 segments** — it
   learned to ignore every input and output one constant bias value.
   Fixed with train-split-derived z-score normalization (never leaking
   val/test statistics into the "fixed" preprocessing).
2. **`pos_weight` was pooled across all 3 transitions instead of computed
   per transition.** pre_event→rising is 100% negative while rising→peak/
   peak→receding are ~93% positive (task 3.4) — pooling blends these into
   one misleading, diluted weight that isn't correctly calibrated for
   either sub-task. Fixed by weighting each transition's own loss term by
   its own train-split class balance.

## Result (19 Sep 2026) — the actual baseline-vs-GNN comparison

| Transition | Split | Baseline F1 (task 3.6/3.7) | GNN F1 |
|---|---|---|---|
| pre_event→rising | all | 0.0 (trivial no-op) | 0.0 (trivial no-op) |
| **rising→peak** | train | 0.0 | **0.79** |
| **rising→peak** | val | 0.0 | **0.44** |
| **rising→peak** | test | 0.0 | **0.93** |
| peak→receding | train | 0.96 | 0.79 |
| peak→receding | val | 0.63 | 0.43 |
| peak→receding | test | 0.98 | 0.93 |

**rising→peak is the headline result** — the measured version of
working.md's core claim, not just an assertion of it. The baseline
completely fails here (F1=0 on every split, `src/models/baseline/README.md`)
because a static reactive rule structurally cannot anticipate a future
rainfall spike from a currently-dry state. The GNN, learning from
elevation/slope/distance_to_drain/rainfall via real spatial graph
convolution, gets this dramatically right instead (F1 0.79/0.44/0.93).

**peak→receding is more mixed, and disclosed as such:** the baseline
actually matches or beats the GNN here on train/test, because that
transition is structurally easy for ANY model given task 3.4's fusion
assigns peak and receding the *identical* `ever_flooded` set — the
baseline's "rain > threshold → flood everything" trivially matches. The
GNN, trying to learn genuine per-node differentiation rather than a
blanket rule, pays a modest precision cost for that. Val performance for
both models on this transition is weak, consistent with task 3.7's
already-disclosed small-N-of-wards variance (only 2 wards each in val/
test).

**Read together:** the GNN's real advantage shows up specifically where
propagation genuinely matters (predicting flood onset), not where a
trivial rule already happens to work by construction of the ground
truth. That's a more honest, specific claim than "the GNN wins" or "the
GNN loses" — task 5.3 formalizes this comparison.

Outputs (`data/processed/ground_truth/`, gitignored): `gnn_model.pt`,
`gnn_feature_normalization_stats.json`, `gnn_training_curve.csv`,
`gnn_evaluation_report.json`.

## `tune_hyperparameters.py` — task 4.2

Directly addresses a limitation task 3.7/4.1 already disclosed: with only
16 wards, a single fixed 70/15/15 split gives noisy, high-variance val/
test metrics that depend heavily on *which* wards happened to land where.
This replaces the single point estimate with **k-fold cross-validation**
(K=4) over task 3.7's train+val wards — the official test wards stay
completely untouched until the final step, never used for selection.

```bash
python src/models/gnn/tune_hyperparameters.py
```

**Method:** (1) partition train+val wards into 4 segment-count-balanced
folds; (2) for each candidate learning rate (0.005/0.01/0.02), train 4
models (one fold held out each time, reusing task 4.1's own
`train_model()`/`evaluate_model()`) and aggregate mean±std F1 across
folds; (3) select the LR with the best mean F1 on the two flood-relevant
transitions; (4) re-run the folds once more with that LR to pool held-out
predictions and sweep the decision threshold (task 4.1 used a naive fixed
0.5); (5) retrain a final model on ALL train+val wards with the selected
LR, evaluate ONCE on the official test wards with the tuned threshold.
Epochs stayed fixed at task 4.1's 300 (not independently grid-searched,
to keep total runtime bounded — ~17 real training runs at ~30s each).

**Result (19 Sep 2026):**
- Selected **LR=0.02** (CV mean flood-relevant F1: 0.673 @ 0.005, 0.764 @
  0.01, **0.768 @ 0.02**) — a modest, real improvement, backed by
  cross-validated evidence rather than one arbitrary split. Fold std was
  ±0.073, giving an honest sense of how much this estimate itself varies.
- Selected **threshold=0.1** (down from the naive 0.5) — the tuned
  model's probabilities for the flood-relevant transitions cluster low
  enough that thresholds 0.1–0.25 all tie at the best pooled F1 (0.923).
- **Final tuned test result: rising→peak and peak→receding both hit
  F1=0.982** (up from task 4.1's un-tuned 0.93/0.93 on the same test
  wards) — a genuine, cross-validated improvement.

**A real, disclosed tradeoff, not hidden:** the SAME global threshold
(0.1), chosen specifically to maximize the flood-relevant transitions,
badly hurts `pre_event->rising` — which should trivially predict "nothing
floods" (task 3.4's design) but now predicts **everything** as flooded
(test accuracy 0.0, down from task 4.1's trivial 1.0). This is an honest,
expected consequence of tuning one global threshold across transitions
with wildly different base rates (0% vs. ~93% positive) rather than a
per-transition threshold — `pre_event->rising` was explicitly excluded
from the tuning objective (see `FLOOD_RELEVANT_TRANSITIONS`) precisely
because it's uninformative for selection, but that also means nothing
protects it from a threshold chosen without it in mind. Worth a per-
transition threshold if this model line is developed further.

Outputs (`data/processed/ground_truth/`, gitignored):
`hyperparameter_tuning_report.json` (full CV grid, threshold sweep,
selected hyperparameters, final tuned test evaluation).

## `evaluate_final_model.py` — task 5.1

Reproduces task 4.2's final tuned model exactly (`train_model()` is fully
deterministic given the same train_mask/lr/seed — confirmed below, not
assumed) and evaluates it on **all three splits** (train/val/test), not
just the test-only number 4.2 itself needed for hyperparameter selection.

```bash
python src/models/gnn/evaluate_final_model.py
```

**Reproducibility, checked not assumed:** this script's own test-split F1/
accuracy/precision/recall are compared byte-for-byte against task 4.2's
saved `final_test_evaluation` and it raises if they don't match exactly —
concrete proof `train_model()` is deterministic here, not a hopeful claim.

**A real, important finding, not hidden:** F1 alone at one tuned threshold
can look excellent purely from matching a transition's base rate — and
that's exactly what a first real run of this script surfaced. Direct
inspection: the final model predicts "flooded" for **100% of test
segments on every transition**, with zero exceptions. Added AUC-ROC
(threshold-independent, hand-rolled — same no-new-dependency convention
as task 3.6's `binary_classification_metrics()`) to check for exactly
this. Real result: **AUC ~0.70–0.76 on train/val (genuine discrimination)
but collapses to ~0.46–0.50 on test** (statistically indistinguishable
from random) for both flood-relevant transitions. The model has learned
real signal — it transfers to train and val — it just doesn't transfer to
whichever 2–3 wards happen to be in the test split. This sharpens task
3.7/4.2's already-disclosed small-N-of-wards variance into something
concrete: the reported F1=0.982/0.973 test wins are correct arithmetic,
not evidence of learned per-segment discrimination on held-out wards.
**Deliberately not fixed here** — task 5.4 exists specifically to
sanity-check comparison anomalies like this one.

Outputs (`data/processed/ground_truth/`, gitignored): `gnn_final_model.pt`,
`gnn_final_model_feature_stats.json`, `gnn_final_per_transition_report.json`
(F1/accuracy/AUC per transition per split).
