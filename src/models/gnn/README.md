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
