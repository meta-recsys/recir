# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""The fault guard every snapshot step runs under."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

logger: logging.Logger = logging.getLogger(__name__)


@contextmanager
def contained(what: str) -> Iterator[None]:
    """Run one snapshot step inside this context; a fault is logged with its traceback and never raised."""
    try:
        yield
    except Exception as e:
        logger.warning("recir snapshot: %s: %r", what, e, exc_info=True)
