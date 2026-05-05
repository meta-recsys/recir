# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""End-to-end producer + consumer parity tests for the bundle pipeline.

The hard-required test in this module
(:meth:`BundleEndToEndTest.test_end_to_end_constants_survive_iter_graph_module_wrapping`)
exercises the full path:

    build a get_attr-bearing GM via T5 ``export_bundle`` →
    load via T6 ``import_bundle`` → wrap via T6 ``wrap_and_lower`` →
    invoke the resulting ``IterGraphModule`` and compare to eager.

Per the test contract: on EACH of the three :class:`IterGraphModule`
sub-GMs (``setup_gm``, ``main_gm``, ``cleanup_gm``) the test asserts
that for every constant FQN the producer wrote:

  1. :func:`torch.fx.graph_module._get_attr` resolves to a tensor
     (NOT ``None``);
  2. the FQN is NOT in :meth:`torch.nn.Module.named_buffers`
     (proving the device-migration reflip restored
     :data:`~torch.export.unflatten._AttrKind.CONSTANT`).

Numerical parity vs eager is asserted on the forward output to confirm
the constants the wrapper restored on the sub-GMs are actually consumed
correctly during execution.
"""

from __future__ import annotations

import tempfile
import unittest

import torch
import torch.nn as nn
from recir.bundle_export import export_bundle
from recir.bundle_runtime import import_bundle, wrap_and_lower
from recir.distributed.parallel_mode import DataParallel
from recir.iter_graph_module import IterGraphModule
from torch.fx.graph_module import _get_attr
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class _ConstAdderModule(nn.Module):
    """Module whose forward adds an inline tensor constant.

    The ``torch.tensor([1.0, 2.0])`` literal lifts to a
    ``_tensor_constantN`` get_attr after tracing, exercising the
    constants pipeline end-to-end.  The :class:`nn.Linear` contributes
    a parameter so the bundle has a non-trivial model_state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc: nn.Linear = nn.Linear(2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + torch.tensor([1.0, 2.0])


class BundleEndToEndTest(DTensorTestBase):
    """Producer + consumer end-to-end parity, with the sub-GM constant invariant.

    Inherits from :class:`DTensorTestBase` so the
    :class:`recir.distributed.parallel_mode.DataParallel.partition`
    invocation can construct its :class:`DeviceMesh` against a real
    (gloo, single-rank) process group.
    """

    @property
    def world_size(self) -> int:
        # Single-rank suffices: this test guards the structural reflip
        # invariant + numerical parity, neither of which depends on the
        # cross-rank collective semantics.
        return 1

    @property
    def device_type(self) -> str:
        # CPU/gloo so the test runs on machines without an accelerator;
        # the lazy-Inductor lowering happens on first invocation, which
        # this test does exercise -- so we keep the inductor backend
        # safe for CPU by setting device_type to "cpu" on the
        # ``DataParallel`` mesh.
        return "cpu"

    @with_comms
    def test_end_to_end_constants_survive_iter_graph_module_wrapping(
        self,
    ) -> None:
        """Full pipeline + per-sub-GM constant invariant + eager parity.

        Per the architectural contract:

          * :class:`IterGraphModule.__init__` rebuilds each of
            ``setup_gm``, ``main_gm``, ``cleanup_gm`` via
            ``fx.GraphModule(src, graph)``, whose ``_copy_attr`` walk
            silently re-registers tensor get_attr attrs as buffers.
          * :func:`recir.bundle_runtime.runtime.wrap_and_lower` invokes
            the device-migration reflip after the IterGraphModule is
            built; this test asserts the reflip held on EACH of the
            three sub-GMs.

        The numerical-parity check confirms the constants the wrapper
        restored are actually consumed correctly at execution time --
        if the reflip silently dropped a value, the forward output
        would diverge from eager.
        """
        torch.manual_seed(0)
        mod = _ConstAdderModule()
        sample_input = torch.randn(2, 2)
        eager_out = mod(sample_input)

        # 1. Export via T5 ``export_bundle`` to a temp dir.
        with tempfile.TemporaryDirectory() as d:
            export_bundle(mod, (sample_input,), output_dir=d, target_device="cpu")

            # 2. Load via T6 ``import_bundle``.
            loaded = import_bundle(d)

            # Sanity: the fixture actually emits at least one tensor
            # constant -- otherwise the test degenerates and stops guarding
            # the reflip path.
            expected_constant_fqns = list(loaded.constants.keys())
            self.assertGreater(
                len(expected_constant_fqns),
                0,
                "fixture is degenerate: no constants emitted by export_bundle",
            )

            # 3. Wrap + lower into an IterGraphModule.
            parallel_mode = DataParallel(
                "replicate",
                num_ranks=1,
                num_processes=1,
                device_type=self.device_type,
            )
            wrapped = wrap_and_lower(loaded, parallel_mode=parallel_mode)
            self.assertIsInstance(wrapped, IterGraphModule)

            # 4. Per-sub-GM constant invariant -- THE hard requirement.
            for sub_gm_name in ("setup_gm", "main_gm", "cleanup_gm"):
                sub_gm = getattr(wrapped, sub_gm_name)
                buffer_fqns = set(dict(sub_gm.named_buffers()).keys())
                for fqn in expected_constant_fqns:
                    with self.subTest(sub_gm=sub_gm_name, fqn=fqn):
                        # (a) The attribute resolves to a tensor (NOT None).
                        value = _get_attr(sub_gm, fqn)
                        self.assertIsNotNone(
                            value,
                            f"{fqn!r} resolves to None on {sub_gm_name} -- "
                            f"the device-migration reflip lost the value",
                        )
                        self.assertIsInstance(
                            value,
                            torch.Tensor,
                            f"{fqn!r} on {sub_gm_name} is not a Tensor "
                            f"(got {type(value).__name__})",
                        )

                        # (b) The FQN is NOT in named_buffers -- proves
                        #     ``_AttrKind.CONSTANT`` is the live
                        #     registration kind.
                        self.assertNotIn(
                            fqn,
                            buffer_fqns,
                            f"{fqn!r} leaked into named_buffers on "
                            f"{sub_gm_name} -- the reflip failed to "
                            f"undo ``_copy_attr``'s buffer registration",
                        )

            # 5. Numerical parity: invoke the wrapped module + compare
            #    to eager.  The constants are NOT threaded through the
            #    call vector (per BindingTable.call_vector contract);
            #    only params + buffers + named_states + user_inputs
            #    appear positionally.
            flat_leaves: list[torch.Tensor] = []
            flat_leaves.extend(loaded.params.values())
            flat_leaves.extend(loaded.buffers.values())
            for state in loaded.named_states.values():
                flat_leaves.extend(state.values())
            flat_leaves.append(sample_input)

            with torch.no_grad():
                wrapped_out = wrapped(*flat_leaves)

            # ``_local_compile``'s ``stateless_func`` returns
            # ``(ret, params, buffers, named_states)``; the
            # caller-flattened codegen preserves that out_spec, so the
            # wrapped module's return is the same 4-tuple.  Element 0
            # is the forward tensor we compare against eager.
            self.assertIsInstance(wrapped_out, tuple)
            forward_out = wrapped_out[0]
            self.assertIsInstance(forward_out, torch.Tensor)
            torch.testing.assert_close(
                forward_out,
                eager_out,
                rtol=1e-5,
                atol=1e-5,
                msg=(
                    "wrap_and_lower output must match eager forward; if "
                    "this fails the constant attrs were dropped on a "
                    "sub-GM during the IterGraphModule rebuild and the "
                    "forward consumed stale or zero values"
                ),
            )


if __name__ == "__main__":
    unittest.main()
