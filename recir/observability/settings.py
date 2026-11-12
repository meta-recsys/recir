# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""The ``RECIR_SNAPSHOT_*`` settings."""

from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass
from typing import ClassVar

logger: logging.Logger = logging.getLogger(__name__)


class Tier(enum.Enum):
    """
    How much a snapshot writes, set by ``RECIR_SNAPSHOT_TIER``.

    ``OFF`` writes nothing, ``INDEX`` writes each stage's graph text, and
    ``FULL`` also writes each build's IR, on backends whose builds capture it.
    """

    OFF = "off"
    INDEX = "index"
    FULL = "full"


@dataclass(frozen=True)
class Settings:
    """
    The snapshot settings: the ``RECIR_SNAPSHOT_*`` knobs, read once per process.

    This class is responsible for reading the knobs from the environment and
    validating them. A malformed knob turns snapshots off, with a warning.
    Deciding when to snapshot belongs to ``compiles.snapshots_on``.
    """

    # Read on first use, so a knob set after import still takes effect.
    _loaded: ClassVar[Settings | None] = None

    tier: Tier = Tier.OFF

    @classmethod
    def load(cls) -> Settings:
        """The process's settings, read from the environment on first call."""
        if cls._loaded is None:
            cls._loaded = cls.from_env()
        return cls._loaded

    @classmethod
    def from_env(cls) -> Settings:
        """
        Settings read from the environment now. A malformed knob gives the
        defaults, with a warning.
        """
        raw = os.environ.get("RECIR_SNAPSHOT_TIER", "")
        try:
            return cls(tier=Tier(raw.strip().lower() or Tier.OFF.value))
        except ValueError:
            logger.warning(
                "recir snapshot: malformed setting, snapshots off: "
                "RECIR_SNAPSHOT_TIER=%r",
                raw,
            )
            return cls()

    @property
    def enabled(self) -> bool:
        """Whether snapshots are on at all."""
        return self.tier is not Tier.OFF

    @property
    def full(self) -> bool:
        """Whether snapshots also write each build's IR, on backends whose builds capture it."""
        return self.tier is Tier.FULL
