# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-ignore-all-errors

"""Shared toy model components for orchestrator-style override tests."""

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from functorch import make_fx
from recir.compiler.tracing import _rematerialize_optimizer
from recir.ops.eager_dispatch import install_graph_break_dispatch, mark_output_with_bwd
from recir.utils.compile import _enable_compile
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn.utils import stateless


_TOY_MARKER_PREFIX = "recir.tests.toy"


class ToyPreprocess(nn.Module):
    def __init__(self, num_embeddings: int = 10, embedding_dim: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)


class _FusedEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, indices, lr):
        ctx.save_for_backward(weight, indices)
        ctx.lr = lr
        return weight[indices]

    @staticmethod
    def backward(ctx, grad_output):
        weight, indices = ctx.saved_tensors
        with torch.no_grad():
            for i, idx in enumerate(indices):
                weight[idx] -= ctx.lr * grad_output[i]
        return None, None, None


class ToyFusedPreprocess(nn.Module):
    """Embedding with fused optimizer — simulates TBE behavior."""

    def __init__(
        self, num_embeddings: int = 10, embedding_dim: int = 4, lr: float = 0.1
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.lr = lr

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return _FusedEmbeddingFunction.apply(self.weight, indices, self.lr)


class ToyDummyPreprocess(nn.Module):
    """Tracing-time stand-in for ToyPreprocess/ToyFusedPreprocess.

    Plants a forward/backward marker pair via ``mark_output_with_bwd`` so the
    post-trace orchestrator can find both halves and install a graph-break
    dispatch back to the original preprocess.
    """

    def __init__(self, batch_size: int = 3, embedding_dim: int = 4, marker_id: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.marker_id = marker_id
        # Requires_grad anchor so the autograd Function backward fires during
        # tracing — produces a findable backward marker in the bwd subgraph.
        self._grad_anchor = nn.Parameter(torch.zeros(1))

    @property
    def _marker_prefix(self) -> str:
        return f"{_TOY_MARKER_PREFIX}::{self.marker_id}"

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        batch_size = indices.shape[0]
        dep = indices.float().sum() * 0.0 + self._grad_anchor * 0.0
        return mark_output_with_bwd(
            torch.zeros(batch_size, self.embedding_dim) + dep,
            self._marker_prefix,
            0,
        )


class ToyDense(nn.Module):
    def __init__(self, embedding_dim: int = 4, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).sum()


class ToyWrapper(nn.Module):
    def __init__(self, preprocess, dense):
        super().__init__()
        self.preprocess = preprocess
        self.dense = dense

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.dense(self.preprocess(indices))


class ToyDispatcher:
    """Orchestrator-style replacement for the legacy ``ToyOverride``.

    ``install_eager`` swaps the original (Fused)Preprocess submodule for a
    ``ToyDummyPreprocess`` so tracing sees a stable, fake-tensor-friendly
    shape.  ``install_graph`` finds the marker pair planted by the dummy and
    installs a graph-break dispatch back to the captured original module.
    """

    def __init__(self, dummy: ToyDummyPreprocess, num_ranks: int = 1) -> None:
        self._dummy = dummy
        self._orig_preprocess: nn.Module | None = None
        self._num_ranks = num_ranks

    def install_eager(self, model: nn.Module) -> None:
        for fqn, mod in list(model.named_modules()):
            if isinstance(mod, (ToyPreprocess, ToyFusedPreprocess)):
                self._orig_preprocess = mod
                parent_fqn, leaf = fqn.rsplit(".", 1) if "." in fqn else ("", fqn)
                parent = model.get_submodule(parent_fqn) if parent_fqn else model
                setattr(parent, leaf, self._dummy)

    def install_graph(self, gm, flat_state):
        assert self._orig_preprocess is not None, (
            "install_eager() must run before install_graph()"
        )
        install_graph_break_dispatch(
            gm,
            self._dummy._marker_prefix,
            self._orig_preprocess,  # type: ignore[arg-type]
            num_state_tensors=len(flat_state),
            num_ranks=self._num_ranks,
        )
        return gm, flat_state


def trace_train_step(wrapper, opt, x):
    """Trace wrapper + opt.step into an FX graph with proper compile context."""
    fake_mode = FakeTensorMode()
    params = dict(wrapper.named_parameters(remove_duplicate=False))
    buffers = dict(wrapper.named_buffers(remove_duplicate=False))
    named_states: dict = {}
    for n, p in params.items():
        if p in opt.state:
            named_states[n] = opt.state[p]

    def trace_fn(params, buffers, named_states, x):
        with (
            stateless._reparametrize_module(wrapper, {**params, **buffers}),
            _rematerialize_optimizer(opt, named_states, params),
        ):
            loss = wrapper(x)
            loss.backward()
            opt.step()
            return loss, params, buffers, named_states

    fake_x = fake_mode.from_tensor(x)
    with fake_mode, _enable_compile(), torch.autograd.detect_anomaly(check_nan=False):
        gm = make_fx(trace_fn, tracing_mode="fake", _allow_non_fake_inputs=False)(
            params, buffers, named_states, fake_x
        )
    flat_state = pytree.tree_leaves([params, buffers, named_states])
    return gm, flat_state


def run_graph(gm, wrapper, opt, x):
    """Execute the traced graph with real inputs."""
    real_params = dict(wrapper.named_parameters(remove_duplicate=False))
    real_buffers = dict(wrapper.named_buffers(remove_duplicate=False))
    real_states: dict = {}
    for n, p in real_params.items():
        if p in opt.state:
            real_states[n] = opt.state[p]
    with torch.no_grad():
        return gm(real_params, real_buffers, real_states, x)
