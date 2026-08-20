# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Scoped symbolic meta overrides used during tracing."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager, ExitStack
from typing import Callable, Generator

from recir.ops.fake.scaled_mm import override_scaled_mm_size_oblivious_meta


_OpMetaOverride = Callable[[], AbstractContextManager[None]]

_OP_META_OVERRIDES: tuple[_OpMetaOverride, ...] = (
    override_scaled_mm_size_oblivious_meta,
)


@contextmanager
def override_op_meta() -> Generator[None, None, None]:
    """Apply registered op meta overrides for the current tracing scope."""
    with ExitStack() as stack:
        for override in _OP_META_OVERRIDES:
            stack.enter_context(override())
        yield
