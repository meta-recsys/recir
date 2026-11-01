# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""The fault guard every snapshot step runs under."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class Fault:
    """What a contained snapshot step raised, once it is over; None if nothing."""

    error: Exception | None = None

    @property
    def error_type(self) -> str | None:
        """The type name of the error; None if nothing was raised."""
        return type(self.error).__name__ if self.error is not None else None

    def note(self, block: dict[str, Any], key: str) -> None:
        """If the step faulted, set ``block[f"{key}_error"]`` to the error's type name."""
        if self.error is not None:
            block[f"{key}_error"] = self.error_type


@contextmanager
def contained(what: str) -> Iterator[Fault]:
    """
    Run one snapshot step inside this context, which gives the step's
    ``Fault``; a fault is logged with its traceback and never raised.
    """
    fault = Fault()
    try:
        yield fault
    except Exception as e:
        fault.error = e
        logger.warning("recir snapshot: %s: %r", what, e, exc_info=True)
