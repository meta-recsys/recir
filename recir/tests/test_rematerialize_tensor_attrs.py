#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Rewiring of tensor references captured on objects attached to a module.

``_reparametrize_module`` swaps a module's own parameters and buffers for the
lifted proxies during a trace. It cannot reach a tensor that some OTHER object
grabbed a reference to -- an LR scheduler, a warmup controller, an EMA tracker.
Such an object goes on mutating the pre-trace tensor while the graph reads a
placeholder nothing writes, so the update never enters the graph and the state
freezes at its seed value for the entire run.

This previously froze a caller's learning rate at its initial value.

The caller names the objects to rewire; nothing is discovered. Declaring one
whose tensors are not module state raises rather than quietly doing nothing,
because "quietly doing nothing" is the failure above.
"""

import unittest

import torch
import torch.nn as nn
from recir.compiler.tracing import _rematerialize_tensor_attrs


class _Holder:
    """The shape of the problem: a plain object owning module state."""

    def __init__(self, counter: torch.Tensor) -> None:
        self.counter = counter
        self.unrelated = torch.zeros((), dtype=torch.int64)
        self.not_a_tensor = "left alone"

    def step(self) -> None:
        self.counter.add_(1)


class _Inner(nn.Module):
    # Annotated at class level: register_buffer goes through nn.Module's
    # __getattr__, which types as `Module | Tensor` without this.
    inner_buf: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("inner_buf", torch.zeros((), dtype=torch.int64))


class _M(nn.Module):
    counter: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(2))
        self.register_buffer("counter", torch.zeros((), dtype=torch.int64))
        self.inner = _Inner()
        self.holder = _Holder(self.counter)


class RematerializeTensorAttrsTest(unittest.TestCase):
    def _lifted(self, m: _M) -> tuple[dict[str, torch.Tensor], dict[int, str]]:
        state = {
            **dict(m.named_parameters(remove_duplicate=False)),
            **dict(m.named_buffers(remove_duplicate=False)),
        }
        real_id_to_fqn = {id(t): n for n, t in state.items()}
        # Stand-ins for the lifted proxies: distinct objects, same FQNs.
        proxies: dict[str, torch.Tensor] = {
            n: torch.zeros_like(t) for n, t in state.items()
        }
        return proxies, real_id_to_fqn

    def test_a_captured_buffer_is_swapped_for_the_proxy(self) -> None:
        m = _M()
        proxies, id_map = self._lifted(m)
        with _rematerialize_tensor_attrs([m.holder], proxies, id_map):
            self.assertIs(m.holder.counter, proxies["counter"])

    def test_writes_inside_the_context_land_on_the_proxy(self) -> None:
        """The whole point. Outside, ``step()`` must move the real buffer;
        inside, it must move the proxy and leave the real one alone."""
        m = _M()
        proxies, id_map = self._lifted(m)
        m.holder.step()
        self.assertEqual(int(m.counter), 1)
        with _rematerialize_tensor_attrs([m.holder], proxies, id_map):
            m.holder.step()
        self.assertEqual(int(proxies["counter"]), 1)
        self.assertEqual(int(m.counter), 1, "the real buffer was mutated")

    def test_the_original_is_restored_on_exit(self) -> None:
        m = _M()
        original = m.holder.counter
        proxies, id_map = self._lifted(m)
        with _rematerialize_tensor_attrs([m.holder], proxies, id_map):
            pass
        self.assertIs(m.holder.counter, original)

    def test_it_restores_even_when_the_body_raises(self) -> None:
        """A trace that dies mid-way must not leave a dead proxy on the object;
        the next step would then write a tensor from a finished trace."""
        m = _M()
        original = m.holder.counter
        proxies, id_map = self._lifted(m)
        with self.assertRaises(RuntimeError):
            with _rematerialize_tensor_attrs([m.holder], proxies, id_map):
                raise RuntimeError("trace blew up")
        self.assertIs(m.holder.counter, original)

    def test_a_holder_nested_under_a_submodule_is_rewired(self) -> None:
        """Reachability is the caller's problem, not a search's: an object one
        level down is rewired the same as one hanging off the root."""
        m = _M()
        nested = _Holder(m.inner.inner_buf)
        proxies, id_map = self._lifted(m)
        with _rematerialize_tensor_attrs([nested], proxies, id_map):
            self.assertIs(nested.counter, proxies["inner.inner_buf"])

    def test_unrelated_tensors_are_untouched(self) -> None:
        """The rule is identity against the module's real state. A tensor the
        object merely owns is not module state and must not be swapped."""
        m = _M()
        unrelated = m.holder.unrelated
        proxies, id_map = self._lifted(m)
        with _rematerialize_tensor_attrs([m.holder], proxies, id_map):
            self.assertIs(m.holder.unrelated, unrelated)

    def test_non_tensor_attributes_are_untouched(self) -> None:
        m = _M()
        proxies, id_map = self._lifted(m)
        with _rematerialize_tensor_attrs([m.holder], proxies, id_map):
            self.assertEqual(m.holder.not_a_tensor, "left alone")

    def test_no_holders_is_a_noop(self) -> None:
        m = _M()
        proxies, id_map = self._lifted(m)
        with _rematerialize_tensor_attrs([], proxies, id_map):
            self.assertIs(m.holder.counter, m.counter)

    def test_a_holder_that_rewires_nothing_raises(self) -> None:
        """The silent case, made loud: a controller writing a tensor that was
        never registered on the module would train against a frozen value."""
        m = _M()
        proxies, id_map = self._lifted(m)
        stray = _Holder(torch.zeros((), dtype=torch.int64))
        with self.assertRaisesRegex(AssertionError, "nothing was rewired"):
            with _rematerialize_tensor_attrs([stray], proxies, id_map):
                pass
