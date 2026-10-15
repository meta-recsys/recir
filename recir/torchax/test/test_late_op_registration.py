# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Ops registered AFTER an Environment was built are still found.

``Environment.load_ops()`` snapshots the registry into ``self._ops`` at
construction. A lazily imported module registering its op later — which is the
normal way to make importing it opt-in — was therefore invisible to any
Environment that already existed.

The failure was not a clean error, which is what made it expensive. Dispatch
raises ``OperatorNotFound``, ``__torch_function__`` catches it and falls through
to calling the op directly, ``autograd_impl`` redispatches BELOW the function
mode, ``__torch_dispatch__`` misses again and calls the op once more outside the
environment, and the assertion that finally fires says only "torchax Tensors can
only do math within the torchax environment". Nothing in that chain names the
op, and it reproduced only when something else happened to construct the
Environment early — turning on AMP, in the case that found it.
"""

from unittest import TestCase

import torch
from recir.torchax import tensor
from recir.torchax.ops import ops_registry

_LIB: torch.library.Library = torch.library.Library("recir_late_reg_test", "DEF")
_LIB.define("late_op(Tensor x) -> Tensor")


@torch.library.register_fake("recir_late_reg_test::late_op")
def _late_op_meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def _late_op_jax(x: object) -> object:
    return x


class LateOpRegistrationTest(TestCase):
    def setUp(self) -> None:
        # An Environment built BEFORE the op is registered, which is the whole
        # point: load_ops() has already run and copied the registry.
        self.env: tensor.Environment = tensor.Environment()
        self.op: torch._ops.OpOverloadPacket = torch.ops.recir_late_reg_test.late_op

    def tearDown(self) -> None:
        for key in (self.op, self.op.default):
            if key in ops_registry.all_aten_ops:
                del ops_registry.all_aten_ops[key]

    def test_an_op_registered_after_construction_is_found(self) -> None:
        ops_registry.register_torch_dispatch_op(self.op, _late_op_jax)
        # Would have raised OperatorNotFound before the live-registry fallback.
        # Asserting on the impl, not just non-None: resolving to SOME operator
        # while running someone else's lowering is the kind of wrong this whole
        # exercise has been about.
        self.assertIs(self.env._get_op_or_decomp(self.op).func, _late_op_jax)

    def test_the_overload_resolves_too(self) -> None:
        """The traced graph calls the overload, not the packet."""
        ops_registry.register_torch_dispatch_op(self.op, _late_op_jax)
        self.assertIs(self.env._get_op_or_decomp(self.op.default).func, _late_op_jax)

    def test_it_is_cached_into_the_environment(self) -> None:
        """Found once, then served from _ops -- the fallback is not a per-call
        scan of the global registry."""
        ops_registry.register_torch_dispatch_op(self.op, _late_op_jax)
        self.env._get_op_or_decomp(self.op)
        self.assertIn(self.op, self.env._ops)

    def test_a_genuinely_unknown_op_still_raises(self) -> None:
        """The fallback must not turn a real miss into something else: an op
        nobody registered has no lowering and has to say so."""
        with self.assertRaises(tensor.OperatorNotFound):
            self.env._get_op_or_decomp(self.op)
