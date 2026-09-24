# Cross-Iteration Steady-State Scheduling in RecIR

## Overview

When `PipelineConfig.materialize_steady_state` is enabled, RecIR uses logical
pipeline stages to build one flattened steady-state FX graph instead of one
Inductor graph per stage.

The key distinction is:

> Stages choose which logical batch an operation processes. They are not
> Inductor compilation boundaries.

For two stages, steady-state invocation `t` runs:

```text
stage 1: batch t - 1
stage 0: batch t
```

Each source node is copied once. Values crossing a stage boundary become
explicit inputs and outputs of the steady-state graph, so the compiled callable
can be reused for every steady-state invocation.

## Configuration

A `PipelinePlan` defines:

- `PipelineStageSpec`: maps marked regions to stages. Exactly one final stage
  must include all unassigned nodes.
- `PipelineSchedule`: orders the stage actions represented in one steady-state
  invocation.
- `PipelineStreamSpec`: optionally maps regions to accelerator side streams
  and assigns stream priorities.

`embedding_steady_state_pipeline_config()` currently creates two stages:

1. `embedding`: `PREPROCESS`, `ROUTE_START`, and `ROUTE_CONTINUE`.
2. `compute`: every otherwise unassigned node.

`PREPROCESS` and `ROUTE_START` run on the `embedding_input` side stream.
The rest of the graph uses the current stream.

## Graph materialization

`SteadyStateGraphModule` receives the original graph and its node-to-stage
assignment.

For every boundary, `_compute_live_after()` finds values produced on or before
that boundary and consumed after it. These values form the carried state.
For asynchronous `all_to_all_single`, the source tensor is also kept alive
while the collective result crosses an invocation boundary.

For a two-stage graph:

```text
A -> B

A = stage 0
B = stage 1
```

the materialized graph is conceptually:

```python
def steady(batch_t, state_from_batch_t_minus_1):
    output_t_minus_1 = B(state_from_batch_t_minus_1)
    next_state = A(batch_t)
    return output_t_minus_1, (next_state,)
```

The actual node copy order comes from `PipelineSchedule.order_actions()`.
Inductor receives the resulting flat graph and may schedule or fuse independent
operations across the former stage boundary.

## Weight and loop-carried state

RecIR distinguishes per-batch carried values from non-batch inputs using
placeholder metadata:

- `is_weight` marks inputs that may be read directly by multiple stages. The
  name includes model parameters, buffers, optimizer state, and other
  non-batch tensor inputs.
- `is_state_input` identifies the subset of inputs that belong to the
  functional process state. Their updated values are returned as the trailing
  leaves of the source graph output in the same order as the state inputs.

Weights are excluded from `_compute_live_after()` because they do not travel
with one batch through the pipeline. During eager fill and drain,
`_refresh_weight_inputs()` replaces the weight values in each pending slot with
the current invocation's values. During steady execution, weight placeholders
are directly available to every copied stage. `_run_steady()` also retains the
current weights in rebuilt slots so a later eager drain can read them.

For an `is_state_input` weight, `_find_updated_weight_nodes()` maps its input
placeholder to the corresponding trailing output node. The default schedule
copies stages in reverse order, so the final stage for the retiring batch is
copied before stage 0 for the newly admitted batch. `_build_steady_graph()` can
therefore connect the final stage's updated state directly to stage 0:

```text
final stage(batch t - 1, weight_t) -> weight_t_plus_1
                                           |
                                           v
stage 0(batch t, weight_t_plus_1) -> next carried state
```

This fresh-versus-stale selection currently depends on graph copy order. If the
updated node is not yet present in the final-stage value map, stage 0 silently
falls back to the weight passed as an invocation argument. This ordering
contract should be made explicit and validated before adding schedules with
different state-read semantics.

## Runtime phases

The wrapper has three phases:

- Fill: `_advance_eager()` runs source-graph stages until every carried-state
  slot is populated. Calls return `PIPELINE_PENDING` until an output retires.
- Steady state: `_run_steady()` invokes the single flattened
  `steady_gm`, returns the oldest batch output, and installs the returned
  state for the next invocation.
- Drain: calls with `last_iter=True` use the eager stage runner without
  admitting a new batch, until no slots remain.

Only `steady_gm` is lowered as one unit. `InductorBackend.dispatch()`
recognizes `SteadyStateGraphModule` and calls
`lower_steady(partial_lower)`.

## Side-stream synchronization

Before lowering, RecIR creates each configured side stream and writes its
Dynamo external-object index to `node.meta["custom"]["stream"]`.

For every direct tensor edge crossing streams, RecIR inserts:

```text
producer -> record_event
wait_event -> consumer
```

One event is reused for all consumers of a producer, and each destination
stream waits at most once for that producer.

At runtime RecIR also:

- waits for `input_ready_event` on side streams;
- waits for prior slot-completion events before consuming carried state;
- calls `record_stream` when inputs are not already owned by the current
  stream;
- re-registers external stream objects for the current invocation; and
- joins all side streams onto the current stream before recording the
  invocation completion event.

## Current constraints

- Stage and stream placement are explicit; there is no automatic retiming or
  cost model.
- `gradient_accumulation_steps` must be 1.
- A steady-state schedule may reorder stages, but explicit
  `PipelineSchedule.dependencies()` are not supported during graph
  materialization.
- Fill and drain execute the original stage nodes eagerly on the current
  stream; only the steady state is compiled as one graph.
- Cross-stream synchronization is inferred from direct FX data edges. Other
  ordering requirements must already be represented in the graph.
