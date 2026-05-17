# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-ignore-all-errors

"""Shared toy model components for override tests."""

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from functorch import make_fx
from recir.api import _enable_compile, _rematerialize_optimizer, Override
from recir.override_ops import anchor_output, default_override_patch_graph
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn.utils import stateless


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
    def __init__(
        self, batch_size: int = 3, embedding_dim: int = 4, override_id: int = 0
    ):
        super().__init__()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.override_id = override_id
        # Requires_grad anchor so OverrideAnchorFn.backward fires during tracing.
        self._grad_anchor = nn.Parameter(torch.zeros(1))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        batch_size = indices.shape[0]
        dep = indices.float().sum() * 0.0 + self._grad_anchor * 0.0
        return anchor_output(
            torch.zeros(batch_size, self.embedding_dim) + dep,
            self.override_id,
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


class ToyOverride(Override):
    def __init__(self, dummy: ToyDummyPreprocess, num_ranks: int = 1):
        self._dummy = dummy
        self._orig_preprocess = None
        self._num_ranks = num_ranks

    def replacement(self, fqn, orig_submodule):
        if isinstance(orig_submodule, (ToyPreprocess, ToyFusedPreprocess)):
            self._orig_preprocess = orig_submodule
            return self._dummy
        return orig_submodule

    def patch_graph(self, gm, flat_state):
        return default_override_patch_graph(
            gm,
            flat_state,
            self._orig_preprocess,
            num_ranks=self._num_ranks,
        )


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
