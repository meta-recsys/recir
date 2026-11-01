# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""How a stage or compile ended, and what became of each artifact it writes."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any


class Outcome(enum.Enum):
    """How a stage's or compile's code ended: it ran to its end, or it raised."""

    COMPLETED = "completed"
    RAISED = "raised"

    @classmethod
    def of(cls, error: BaseException | None) -> Outcome:
        """The outcome of code that raised ``error``; ``error`` is None if it raised nothing."""
        return cls.COMPLETED if error is None else cls.RAISED


class ArtifactState(enum.Enum):
    """What became of one artifact: written, or lost to a fault while formatting or writing it."""

    EMITTED = "emitted"
    EMIT_FAILED = "emit_failed"


@dataclass(frozen=True)
class ArtifactStatus:
    """
    One artifact's status in its stage's manifest: its state, and for a fault
    the exception's type name.
    """

    state: ArtifactState
    error_type: str | None = None

    def to_json(self) -> dict[str, Any]:
        """The status as it appears in the manifest."""
        out: dict[str, Any] = {"status": self.state.value}
        if self.error_type is not None:
            out["error"] = self.error_type
        return out
