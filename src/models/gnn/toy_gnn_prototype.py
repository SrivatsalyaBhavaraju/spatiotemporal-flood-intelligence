"""
Task 1.12 -- toy/synthetic smoke test for the graph-ML stack (PyTorch + CUDA
+ PyTorch Geometric + PyTorch Geometric Temporal) on a graph shaped like our
real problem: nodes = road segments, edges = segment adjacency, node
features vary across the 4 flood phases (pre-event / rising / peak /
receding -- the label scheme task 3.4 will fuse from Sentinel-1 + rainfall +
news). This ONLY proves the environment runs end-to-end on GPU (forward pass
+ backward pass + real gradients) -- it is not a real model, uses random
data, and is not trained on our actual features/labels. That's tasks 3.x.

Why two architectures: A3TGCN (attention-augmented temporal GCN) and
MPNN-LSTM are the two candidates named in working.md's model section --
this prototype exercises both so a later architecture choice is informed by
having actually run each, not just read the papers.

--- torch_sparse import shim ---
torch_geometric_temporal (last released 2022) unconditionally imports
torch_sparse at package-import time for its EvolveGCN model, even though
neither A3TGCN nor MPNNLSTM below need it. No prebuilt torch_sparse wheel
exists yet for our torch 2.13+cu126 / Python 3.13 combo, and building one
from source needs a full MSVC + CUDA toolkit we don't have installed on this
laptop. Since EvolveGCN is never used here, we stub the torch_sparse module
with PyTorch Geometric's own fallback SparseTensor class -- the exact same
class PyG itself substitutes internally when torch_sparse is absent (see
torch_geometric.typing.WITH_TORCH_SPARSE). This is a documented workaround,
not silently-faked functionality; see src/models/gnn/README.md.
"""
import sys
import types

import torch
import torch.nn.functional as F
from torch_geometric.utils import erdos_renyi_graph

if "torch_sparse" not in sys.modules:
    import torch_geometric.typing as pyg_typing

    _stub = types.ModuleType("torch_sparse")
    _stub.SparseTensor = pyg_typing.SparseTensor
    sys.modules["torch_sparse"] = _stub

from torch_geometric_temporal.nn.recurrent import A3TGCN, MPNNLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Toy scale -- the real line-graph (task 2.1) will have ~17,195 road-segment
# nodes (src/graph/README.md). 40 nodes is enough to exercise every shape in
# the forward/backward pass without wasting time on a smoke test.
NUM_NODES = 40
IN_CHANNELS = 3  # toy stand-in for elevation, slope, rainfall-so-far (real features: tasks 1.5-1.8)
PERIODS = 4       # matches the real 4-phase label scheme (pre-event/rising/peak/receding)

torch.manual_seed(0)


def make_toy_graph():
    return erdos_renyi_graph(NUM_NODES, edge_prob=0.08, directed=False).to(DEVICE)


def test_a3tgcn():
    print("=== A3TGCN ===")
    edge_index = make_toy_graph()
    X = torch.randn(NUM_NODES, IN_CHANNELS, PERIODS, device=DEVICE)
    y = torch.randn(NUM_NODES, 1, device=DEVICE)  # toy target: e.g. per-segment flood depth/probability

    model = A3TGCN(in_channels=IN_CHANNELS, out_channels=1, periods=PERIODS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    out = model(X, edge_index)
    print(f"  output shape: {tuple(out.shape)}  (expected ({NUM_NODES}, 1))")

    loss = F.mse_loss(out, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  loss: {loss.item():.4f}  |  {n_grads} parameter tensors received gradients (backward pass OK)")


def test_mpnn_lstm():
    print("=== MPNN-LSTM ===")
    edge_index = make_toy_graph()
    hidden = 8
    dropout = 0.0

    # MPNNLSTM expects a window-batched, block-diagonal graph: features for
    # each of `window` timesteps stacked row-wise ([PERIODS*NUM_NODES, C]),
    # with edge_index tiled once per timestep and node ids offset by
    # t*NUM_NODES -- each timestep's graph convolution stays independent;
    # only the LSTM afterwards mixes information across time.
    snapshots = [torch.randn(NUM_NODES, IN_CHANNELS, device=DEVICE) for _ in range(PERIODS)]
    X = torch.cat(snapshots, dim=0)
    edge_index_batched = torch.cat([edge_index + t * NUM_NODES for t in range(PERIODS)], dim=1)
    edge_weight = torch.ones(edge_index_batched.size(1), device=DEVICE)

    expected_dim = 2 * hidden + IN_CHANNELS + PERIODS - 1  # per MPNNLSTM's own docstring
    y = torch.randn(NUM_NODES, expected_dim, device=DEVICE)

    model = MPNNLSTM(
        in_channels=IN_CHANNELS, hidden_size=hidden, num_nodes=NUM_NODES,
        window=PERIODS, dropout=dropout,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    out = model(X, edge_index_batched, edge_weight)
    print(f"  output shape: {tuple(out.shape)}  (expected ({NUM_NODES}, {expected_dim}))")

    loss = F.mse_loss(out, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  loss: {loss.item():.4f}  |  {n_grads} parameter tensors received gradients (backward pass OK)")


def main():
    dev_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU only"
    print(f"device: {DEVICE}  ({dev_name})")
    print(f"torch {torch.__version__}\n")
    test_a3tgcn()
    print()
    test_mpnn_lstm()
    print("\nBoth models ran a forward + backward pass on synthetic data without error.")
    print("This only smoke-tests the environment -- neither model is trained on real features/labels yet (task 3.x).")


if __name__ == "__main__":
    main()
