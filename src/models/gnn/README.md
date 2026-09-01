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
