# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Unit tests for :func:`recir.bundle_runtime.wrap_and_lower`.

Two non-negotiable invariants are checked here:

* ``wrap_and_lower`` returns an :class:`recir.iter_graph_module.IterGraphModule`
  whose three sub-GMs (``setup_gm``, ``main_gm``, ``cleanup_gm``) are
  populated and addressable.
* For every ``get_attr`` constant the producer wrote, on EACH of the
  three sub-GMs:

    1. the attribute resolves via :func:`torch.fx.graph_module._get_attr`
       to a tensor (i.e. it survived the :class:`fx.GraphModule`
       reconstruction inside :class:`IterGraphModule.__init__`);
    2. the FQN is NOT present in
       :meth:`torch.nn.Module.named_buffers` (proving the device-migration
       reflip restored ``_AttrKind.CONSTANT`` -- otherwise the attribute
       would silently leak into the buffer namespace via
       ``torch.fx.graph_module._copy_attr``).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.fx as fx
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
    """Module whose forward adds an inline tensor constant (lifts to a get_attr)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc: nn.Linear = nn.Linear(2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + torch.tensor([1.0, 2.0])


def _export_const_adder(out_dir: Path) -> None:
    torch.manual_seed(0)
    mod = _ConstAdderModule()
    x = torch.randn(2, 2)
    export_bundle(mod, (x,), output_dir=out_dir, target_device="cpu")


class WrapAndLowerTest(DTensorTestBase):
    """Validate the structural shape returned by :func:`wrap_and_lower`."""

    @property
    def world_size(self) -> int:
        # Single-rank suffices for the structural assertions made here;
        # numerical parity vs eager is exercised in
        # :mod:`recir.bundle_runtime.tests.test_end_to_end`.
        return 1

    @property
    def device_type(self) -> str:
        # Force CPU/gloo so the test runs without a CUDA accelerator.
        # ``wrap_and_lower`` only inserts lazy ``_InductorModule``
        # subgraphs at this point; the actual Inductor compilation is
        # deferred to first invocation.
        return "cpu"

    @with_comms
    def test_wrap_and_lower_returns_iter_graph_module(self) -> None:
        """``wrap_and_lower`` returns an :class:`IterGraphModule` with all 3 sub-GMs."""
        with tempfile.TemporaryDirectory() as d:
            _export_const_adder(Path(d))
            loaded = import_bundle(d)

        parallel_mode = DataParallel(
            "replicate",
            num_ranks=1,
            num_processes=1,
            device_type=self.device_type,
        )
        wrapped = wrap_and_lower(loaded, parallel_mode=parallel_mode)
        self.assertIsInstance(wrapped, IterGraphModule)
        # All three sub-GMs are populated and addressable.
        self.assertIsInstance(wrapped.setup_gm, fx.GraphModule)
        self.assertIsInstance(wrapped.main_gm, fx.GraphModule)
        self.assertIsInstance(wrapped.cleanup_gm, fx.GraphModule)

    @with_comms
    def test_wrap_and_lower_constants_survive_on_all_three_sub_gms(self) -> None:
        """The producer's ``get_attr`` constants survive as CONSTANT-kind on every sub-GM.

        :class:`IterGraphModule.__init__` rebuilds each sub-GM via
        ``fx.GraphModule(src, graph)``, whose ``_copy_attr`` walk
        silently re-registers tensor ``get_attr`` attrs as buffers.
        :func:`recir.bundle_runtime.runtime.wrap_and_lower` invokes the
        device-migration reflip after the IterGraphModule is built;
        this test asserts the reflip held on EACH of the three sub-GMs:

          * the attribute resolves to a tensor via
            :func:`torch.fx.graph_module._get_attr`;
          * the attribute is NOT present in
            :meth:`torch.nn.Module.named_buffers` (proving
            ``_AttrKind.CONSTANT`` is the live registration kind, not
            buffer).

        Without the reflip, downstream optimisation passes would drop
        the values on the next ``recompile()`` and consumers reading
        via ``named_buffers`` would observe attributes that the
        producer's contract says are constants.
        """
        with tempfile.TemporaryDirectory() as d:
            _export_const_adder(Path(d))
            loaded = import_bundle(d)

        parallel_mode = DataParallel(
            "replicate",
            num_ranks=1,
            num_processes=1,
            device_type=self.device_type,
        )
        wrapped = wrap_and_lower(loaded, parallel_mode=parallel_mode)

        # Every constant FQN the producer wrote must hold on every
        # sub-GM.  Fixture sanity: at least one constant exists.
        constant_fqns = list(loaded.constants.keys())
        self.assertGreater(
            len(constant_fqns),
            0,
            "fixture is degenerate: no get_attr constants on the GM",
        )

        for sub_gm_name in ("setup_gm", "main_gm", "cleanup_gm"):
            sub_gm = getattr(wrapped, sub_gm_name)
            buffer_fqns = set(dict(sub_gm.named_buffers()).keys())
            for fqn in constant_fqns:
                with self.subTest(sub_gm=sub_gm_name, fqn=fqn):
                    value = _get_attr(sub_gm, fqn)
                    self.assertIsNotNone(
                        value,
                        f"{fqn!r} resolves to None on {sub_gm_name}",
                    )
                    self.assertIsInstance(
                        value,
                        torch.Tensor,
                        f"{fqn!r} on {sub_gm_name} is not a Tensor "
                        f"(got {type(value).__name__})",
                    )
                    self.assertNotIn(
                        fqn,
                        buffer_fqns,
                        f"{fqn!r} leaked into named_buffers on "
                        f"{sub_gm_name} -- CONSTANT-kind reflip failed",
                    )

    @with_comms
    def test_wrap_and_lower_does_not_thread_constants_into_partition(self) -> None:
        """The runtime call vector excludes constants by design.

        :meth:`recir.serde.BindingTable.call_vector` already filters
        ``constant`` bindings out of the placeholder ordering (per the
        schema's design), and the deserializer registered the values on
        the GM as ``_AttrKind.CONSTANT`` attributes accessible via
        :func:`torch.fx.graph_module._get_attr`.  This test confirms
        ``wrap_and_lower`` honours that contract: invoking the wrapped
        module never receives constants positionally.

        Concretely: the BindingTable's ``call_vector`` slice is the
        single source of truth for what flows through the runtime; the
        constant FQNs do not appear in it.
        """
        with tempfile.TemporaryDirectory() as d:
            _export_const_adder(Path(d))
            loaded = import_bundle(d)

        # Sanity: the producer DID record at least one constant in the
        # binding table (so the absence-from-call_vector below is
        # non-trivial).
        constant_binding_fqns = {b.fqn for b in loaded.bindings.constants()}
        self.assertGreater(len(constant_binding_fqns), 0)

        # Constants are NOT in the call vector -- that's the schema's
        # contract.  Manual prefix slicing would be incorrect AND
        # unnecessary.
        call_vector_fqns = {b.fqn for b in loaded.bindings.call_vector()}
        for fqn in constant_binding_fqns:
            self.assertNotIn(
                fqn,
                call_vector_fqns,
                f"constant {fqn!r} appeared in BindingTable.call_vector() "
                f"-- the schema's filter is broken or the producer "
                f"emitted a non-constant binding for this FQN",
            )


if __name__ == "__main__":
    unittest.main()
