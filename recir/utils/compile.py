# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Low-level helpers shared across the recir compile/trace paths.

Kept dependency-free (torch only) so both ``recir.api`` and the lower
``recir.region_transforms`` layer can import it without an import cycle.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import torch


@contextmanager
def _enable_compile() -> Generator[None, None, None]:
    """Make ``is_compiling()`` report True for the duration of the context.

    recir needs this while tracing for two reasons:

    - ``torch._utils.is_compiling()`` changes optimizer behavior; it must be
      True to bake ``opt.step`` into the graph (and to skip capturable-mode
      device asserts / ``.item()`` calls in ``_get_value()``). See
      https://github.com/pytorch/pytorch/blob/a524123c91ab399c9dd6882c1189596dd77e7734/torch/optim/optimizer.py#L41
    - Many ops gate data-dependent asserts / ``.item()`` behind
      ``if not torch.compiler.is_compiling():`` (which reads
      ``_is_compiling_flag``) so they are skipped during capture; otherwise
      they run on unbacked SymInts and raise ``GuardOnDataDependentSymNode``.

    Used by the outer trace (``api._local_compile``) and the region-transform
    discovery + inner ``make_fx`` (``api._discover_for_region_transforms`` and
    ``region_transforms.core._apply_region_transforms``).
    """

    def _f_true() -> bool:
        return True

    orig_is_compiling_code = torch._utils.is_compiling.__code__  # pyre-ignore[16]
    torch._utils.is_compiling.__code__ = _f_true.__code__  # pyre-ignore[16]
    orig_compiling_flag = torch.compiler._is_compiling_flag
    torch.compiler._is_compiling_flag = True
    try:
        yield
    finally:
        torch._utils.is_compiling.__code__ = orig_is_compiling_code  # pyre-ignore[16]
        torch.compiler._is_compiling_flag = orig_compiling_flag
