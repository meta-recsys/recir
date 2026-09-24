# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from recir.pipeline import (
    PipelineAction,
    PipelineConfig,
    PipelinePlan,
    PipelineSchedule,
    PipelineStageSpec,
    PipelineStreamSpec,
)


class PipelineRegion(StrEnum):
    PREPROCESS = "preprocess"
    ROUTE_START = "route_start"
    ROUTE_CONTINUE = "route_continue"
    EMBEDDING_LOOKUP = "embedding_lookup"


@dataclass(frozen=True)
class _BoundedStalenessSchedule(PipelineSchedule):
    state_read_stage: str
    state_write_stage: str
    async_start_batch: int

    @staticmethod
    def _action_for_stage(
        actions: Sequence[PipelineAction], stage_name: str
    ) -> PipelineAction | None:
        return next(
            (action for action in actions if action.stage_name == stage_name),
            None,
        )

    def order_actions(
        self,
        ready_actions: Sequence[PipelineAction],
    ) -> Sequence[PipelineAction]:
        ordered = list(super().order_actions(ready_actions))
        state_read = self._action_for_stage(ordered, self.state_read_stage)
        state_write = self._action_for_stage(ordered, self.state_write_stage)
        if (
            state_read is not None
            and state_write is not None
            and state_write.batch_index >= self.async_start_batch
        ):
            ordered.remove(state_read)
            ordered.insert(ordered.index(state_write), state_read)
        return tuple(ordered)

    def dependencies(
        self,
        action: PipelineAction,
        ready_actions: Sequence[PipelineAction],
    ) -> Sequence[PipelineAction]:
        state_write = self._action_for_stage(ready_actions, self.state_write_stage)
        if (
            action.stage_name == self.state_read_stage
            and state_write is not None
            and state_write.batch_index < self.async_start_batch
        ):
            return (state_write,)
        return ()


def sdd_pipeline_plan(schedule: PipelineSchedule | None = None) -> PipelinePlan:
    return PipelinePlan(
        stages=(
            PipelineStageSpec(
                name="route_start",
                regions=(PipelineRegion.PREPROCESS, PipelineRegion.ROUTE_START),
                required_regions=(PipelineRegion.ROUTE_START,),
            ),
            PipelineStageSpec(
                name="route_continue",
                regions=(PipelineRegion.ROUTE_CONTINUE,),
                required_regions=(PipelineRegion.ROUTE_CONTINUE,),
            ),
            PipelineStageSpec(
                name="compute",
                include_unassigned_nodes=True,
            ),
        ),
        schedule=schedule or PipelineSchedule(),
    )


def semi_sync_pipeline_plan(
    *,
    start_batch: int = 900,
    schedule: PipelineSchedule | None = None,
) -> PipelinePlan:
    if start_batch < 0:
        raise ValueError("start_batch must be non-negative")
    return PipelinePlan(
        stages=(
            PipelineStageSpec(
                name="route_start",
                regions=(PipelineRegion.PREPROCESS, PipelineRegion.ROUTE_START),
                required_regions=(PipelineRegion.ROUTE_START,),
            ),
            PipelineStageSpec(
                name="route_continue",
                regions=(PipelineRegion.ROUTE_CONTINUE,),
                required_regions=(PipelineRegion.ROUTE_CONTINUE,),
            ),
            PipelineStageSpec(
                name="embedding_lookup",
                regions=(PipelineRegion.EMBEDDING_LOOKUP,),
                required_regions=(PipelineRegion.EMBEDDING_LOOKUP,),
                allow_state_reads=True,
            ),
            PipelineStageSpec(
                name="compute",
                include_unassigned_nodes=True,
            ),
        ),
        schedule=schedule
        or _BoundedStalenessSchedule(
            state_read_stage="embedding_lookup",
            state_write_stage="compute",
            async_start_batch=start_batch,
        ),
    )


def embedding_steady_state_pipeline_plan(
    schedule: PipelineSchedule | None = None,
) -> PipelinePlan:
    return PipelinePlan(
        stages=(
            PipelineStageSpec(
                name="embedding",
                regions=(
                    PipelineRegion.PREPROCESS,
                    PipelineRegion.ROUTE_START,
                    PipelineRegion.ROUTE_CONTINUE,
                ),
                required_regions=(
                    PipelineRegion.ROUTE_START,
                    PipelineRegion.ROUTE_CONTINUE,
                ),
                allow_state_reads=True,
            ),
            PipelineStageSpec(
                name="compute",
                include_unassigned_nodes=True,
            ),
        ),
        schedule=schedule or PipelineSchedule(),
        side_streams=(
            PipelineStreamSpec(
                name="embedding_input",
                regions=(PipelineRegion.PREPROCESS, PipelineRegion.ROUTE_START),
            ),
        ),
    )


def embedding_steady_state_pipeline_config(
    *,
    gradient_accumulation_steps: int = 1,
    schedule: PipelineSchedule | None = None,
) -> PipelineConfig:
    return PipelineConfig.from_plan(
        embedding_steady_state_pipeline_plan(schedule),
        gradient_accumulation_steps=gradient_accumulation_steps,
        materialize_steady_state=True,
    )


def sdd_pipeline_config(
    *,
    gradient_accumulation_steps: int = 1,
    schedule: PipelineSchedule | None = None,
) -> PipelineConfig:
    return PipelineConfig.from_plan(
        sdd_pipeline_plan(schedule),
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


def semi_sync_pipeline_config(
    *,
    start_batch: int = 900,
    gradient_accumulation_steps: int = 1,
    schedule: PipelineSchedule | None = None,
) -> PipelineConfig:
    return PipelineConfig.from_plan(
        semi_sync_pipeline_plan(
            start_batch=start_batch,
            schedule=schedule,
        ),
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
