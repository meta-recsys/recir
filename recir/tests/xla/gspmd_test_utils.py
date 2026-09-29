# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Shared test utilities for GSPMD and shard_map XLA compilation tests.

Provides common constants, model definitions, DDP baseline helpers,
and assertion functions used across test_gspmd.py,
test_gspmd_single_process.py, and test_gspmd_multi_device.py.
"""

import torch
import torch.nn as nn
from recir.lowering.xla_lower import _is_xla_compatible, _XLAModule

# IMPORTANT: must run BEFORE any `import jax` so the cuda plugin path patch
# is in effect when jax.devices() / jax.* first triggers plugin discovery.
from recir.tests.xla import _jax_cuda_plugin_fix  # noqa: F401
from torch.nn.parallel import DistributedDataParallel as DDP


DIM = 16
LOCAL_BS = 1  # per-device batch size
NUM_ITERS = 4


class SimpleMLP(nn.Module):
    def __init__(self, dim: int = DIM):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class DestroyPGMixin:
    """Mixin that overrides destroy_pg for DTensorTestBase subclasses.

    Skips dist.barrier() which calls _get_accelerator() and fails after
    rename_privateuse1_backend("jax") is called by torchax
    (PrivateUse1HooksInterface is not registered for JAX).
    Uses a CUDA-explicit barrier via NCCL process group instead.
    """

    def destroy_pg(self):
        import torch.distributed as dist

        if dist.is_initialized():
            if torch.cuda.is_available():
                t = torch.zeros(1, device=f"cuda:{self.rank}")
                dist.all_reduce(t)
                torch.cuda.synchronize()
            dist.destroy_process_group()


def _create_model_and_warmup(rank):
    """Create a SimpleMLP on the given CUDA rank and run one warmup step."""
    torch.manual_seed(42)
    model = SimpleMLP().cuda(rank)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    warmup = torch.randn(LOCAL_BS, DIM, device=f"cuda:{rank}")
    model(warmup).sum().backward()
    opt.step()
    opt.zero_grad()
    return model, opt


def _generate_inputs(rank, num_iters=NUM_ITERS, local_bs=LOCAL_BS):
    """Pre-generate per-rank inputs: each rank gets different data."""
    torch.manual_seed(123 + rank)
    return [torch.randn(local_bs, DIM, device=f"cuda:{rank}") for _ in range(num_iters)]


def _run_ddp_baseline(rank, inputs, loss_reduction="sum"):
    """Run DDP baseline and return params snapshot after each step."""
    model, _ = _create_model_and_warmup(rank)
    ddp_model = DDP(model, device_ids=[rank])
    ddp_opt = torch.optim.SGD(ddp_model.parameters(), lr=0.01, foreach=False)
    reduce_fn = torch.sum if loss_reduction == "sum" else torch.mean
    params_per_step = []
    for inp in inputs:
        reduce_fn(ddp_model(inp)).backward()
        ddp_opt.step()
        ddp_opt.zero_grad()
        params_per_step.append(
            {n: p.clone() for n, p in ddp_model.module.named_parameters()}
        )
    return params_per_step


def _run_single_rank_baseline(
    global_input, num_ranks, num_iters=NUM_ITERS, loss_reduction="sum"
):
    """Run single-rank baseline on CPU and return params snapshot after each step.

    Creates a fresh model (seed 42) and trains with the full global batch on a
    single rank.  For sum loss the gradient is divided by ``num_ranks`` so that
    it matches the all-reduce-averaged gradient produced by replicated data
    parallelism.  For mean loss, ``mean()`` already normalises by the correct
    global element count.
    """
    torch.manual_seed(42)
    model = SimpleMLP().cpu()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    params_per_step = []
    for _ in range(num_iters):
        if loss_reduction == "sum":
            (model(global_input).sum() / num_ranks).backward()
        else:
            model(global_input).mean().backward()
        opt.step()
        opt.zero_grad()
        params_per_step.append({n: p.clone() for n, p in model.named_parameters()})
    return params_per_step


def assert_jax_multi_device(test_case, total_gpus, devices_per_process, num_processes):
    """Verify JAX is correctly initialized with multiple devices per process."""
    import jax  # @manual

    test_case.assertEqual(
        jax.device_count(),
        total_gpus,
        f"Expected {total_gpus} total JAX devices, got {jax.device_count()}",
    )
    test_case.assertEqual(
        len(jax.local_devices()),
        devices_per_process,
        f"Expected {devices_per_process} local JAX devices per process, "
        f"got {len(jax.local_devices())}",
    )
    test_case.assertEqual(
        jax.process_count(),
        num_processes,
        f"Expected {num_processes} JAX processes, got {jax.process_count()}",
    )


def _iter_subgraphs(compiled_gm):
    """Yield ``(name, sub_gm)`` for the inner gm(s) of ``compiled_gm``.

    Post-NC12 (XLABackend default ``uses_iter_graph=False``), an XLA-backed
    ``compiled_obj.gm`` is a plain ``fx.GraphModule`` rather than an
    ``IterGraphModule`` wrapper -- the ``setup_gm`` / ``main_gm`` /
    ``cleanup_gm`` attributes only exist on the wrapper.  Detect both
    shapes so a single helper works across the IterGraph and bypass
    paths.
    """
    has_iter = all(
        hasattr(compiled_gm, attr) for attr in ("setup_gm", "main_gm", "cleanup_gm")
    )
    if has_iter:
        for gm_name in ("setup_gm", "main_gm", "cleanup_gm"):
            yield gm_name, getattr(compiled_gm, gm_name)
    else:
        yield "gm", compiled_gm


def assert_no_graph_breaks(test_case, compiled_obj):
    """Verify exactly 1 _XLAModule per subgraph and that collectives are present.

    Iterates the inner sub-GraphModules (setup/main/cleanup when the gm is
    wrapped in IterGraphModule; just the gm itself when it is bypassed --
    e.g. XLAGSPMDBackend / XLAShardMapBackend default to
    ``uses_iter_graph=False`` after NC12).
    """
    for gm_name, gm in _iter_subgraphs(compiled_obj.gm):
        xla_modules = [m for m in gm.modules() if isinstance(m, _XLAModule)]
        incompatible = []
        for node in gm.graph.nodes:
            if node.op in ("placeholder", "output", "call_module"):
                continue
            ok, reason = _is_xla_compatible(node)
            if not ok:
                incompatible.append(f"  {node.target} — {reason}")
        test_case.assertEqual(
            len(xla_modules),
            1,
            f"{gm_name}: expected exactly 1 XLA subgraph (no graph "
            f"breaks), got {len(xla_modules)}.\n"
            f"Incompatible nodes:\n" + "\n".join(incompatible or ["  (none)"]),
        )
        test_case.assertTrue(
            xla_modules[0]._has_collective_ops(),
            f"{gm_name}: XLA subgraph missing _c10d_functional ops "
            f"(graph break separated collectives into a filtered "
            f"segment).\nIncompatible nodes causing the break:\n"
            + "\n".join(incompatible or ["  (none found)"]),
        )


def assert_hlo_has_collective(test_case, compiled_obj, collective_name, backend_name):
    """Verify the compiled HLO contains the expected collective operation."""
    found = False
    for mod in compiled_obj.gm.modules():
        if isinstance(mod, _XLAModule) and mod.xla_hlo_text:
            if collective_name in mod.xla_hlo_text:
                found = True
                break
    test_case.assertTrue(
        found,
        f"{backend_name}: expected {collective_name} in compiled XLA HLO",
    )


# ---------------------------------------------------------------------------
# DLRM test model and helpers
# ---------------------------------------------------------------------------

DLRM_EMBED_DIM = 8
DLRM_NUM_EMBEDDINGS = 32
DLRM_DENSE_DIM = 8
DLRM_NUM_SPARSE = 4
DLRM_BOTTOM_DIMS = [16, DLRM_EMBED_DIM]  # project dense features to embed_dim
DLRM_TOP_DIMS = [32, 1]  # final classifier


class CrossNet(nn.Module):
    """Simplified DCNv2 cross-network layer: x_{l+1} = x_0 * (W @ x_l + b) + x_l."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x0, x):
        return x0 * self.linear(x) + x


class SimpleDLRM(nn.Module):
    """Minimal DLRM model for RecIR compile-pipeline validation.

    Architecture:
        dense features  → bottom MLP → ┐
        sparse features → embedding  → ┤ concat → cross-net → top MLP → logit
    """

    def __init__(
        self,
        dense_dim: int = DLRM_DENSE_DIM,
        num_sparse: int = DLRM_NUM_SPARSE,
        num_embeddings: int = DLRM_NUM_EMBEDDINGS,
        embed_dim: int = DLRM_EMBED_DIM,
        bottom_dims=None,
        top_dims=None,
    ):
        super().__init__()
        if bottom_dims is None:
            bottom_dims = DLRM_BOTTOM_DIMS
        if top_dims is None:
            top_dims = DLRM_TOP_DIMS
        self.embedding = nn.Embedding(num_embeddings, embed_dim)
        # Bottom MLP: dense features → embed_dim-sized representation
        bottom_layers = []
        prev = dense_dim
        for dim in bottom_dims:
            bottom_layers.append(nn.Linear(prev, dim))
            bottom_layers.append(nn.ReLU())
            prev = dim
        self.bottom_mlp = nn.Sequential(*bottom_layers)
        # Cross-network: operates on concatenated bottom_mlp output + all embeddings
        interaction_dim = embed_dim + num_sparse * embed_dim  # bottom + sparse
        self.cross_net = CrossNet(interaction_dim)
        # Top MLP: interaction output → logits
        top_layers = []
        prev = interaction_dim
        for i, dim in enumerate(top_dims):
            top_layers.append(nn.Linear(prev, dim))
            if i < len(top_dims) - 1:
                top_layers.append(nn.ReLU())
            prev = dim
        self.top_mlp = nn.Sequential(*top_layers)

    def forward(self, dense_features, sparse_indices):
        """
        Args:
            dense_features: (B, dense_dim) float tensor
            sparse_indices: (B, num_sparse) int64 tensor
        Returns:
            logits: (B, 1) float tensor
        """
        bottom_out = self.bottom_mlp(dense_features)  # (B, embed_dim)
        all_embeds = self.embedding(sparse_indices)  # (B, num_sparse, embed_dim)
        sparse_out = all_embeds.flatten(1)  # (B, num_sparse * embed_dim)
        interaction_in = torch.cat(
            [bottom_out, sparse_out], dim=1
        )  # (B, interaction_dim)
        interaction_out = self.cross_net(interaction_in, interaction_in)
        return self.top_mlp(interaction_out)  # (B, 1)


def _generate_dlrm_inputs(
    global_bs,
    dense_dim=DLRM_DENSE_DIM,
    num_sparse=DLRM_NUM_SPARSE,
    num_embeddings=DLRM_NUM_EMBEDDINGS,
):
    dense = torch.randn(global_bs, dense_dim)
    sparse_indices = torch.randint(0, num_embeddings, (global_bs, num_sparse))
    labels = torch.rand(global_bs, 1)
    return dense, sparse_indices, labels


def _run_dlrm_baseline(
    dense,
    sparse_indices,
    labels,
    num_ranks,
    num_iters=NUM_ITERS,
    loss_reduction="sum",
):
    """Run single-rank DLRM baseline on CPU, return per-step param snapshots.

    Same normalization logic as _run_single_rank_baseline: for sum loss the
    gradient is divided by ``num_ranks`` to match all-reduce averaging.
    """
    torch.manual_seed(42)
    model = SimpleDLRM().cpu()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    loss_fn = nn.BCEWithLogitsLoss(
        reduction="sum" if loss_reduction == "sum" else "mean"
    )
    params_per_step = []
    for _ in range(num_iters):
        logits = model(dense, sparse_indices)
        loss = loss_fn(logits, labels)
        if loss_reduction == "sum":
            (loss / num_ranks).backward()
        else:
            loss.backward()
        opt.step()
        opt.zero_grad()
        params_per_step.append({n: p.clone() for n, p in model.named_parameters()})
    return params_per_step
