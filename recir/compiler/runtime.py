# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Runtime adapters and state controls for compiled RecIR callables."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.utils._pytree as pytree
from recir.compiler.process_state import CompiledProcessState
from recir.pipeline import is_pipeline_pending


# Note that the Python convention of __dict__ requires the key to be str.
# TODO: ensure the key is unique.
COMPILED_OBJECT_KEY = "_compiled_obj"
PREPARE_COMPILED_ATTR = "_recir_prepare_compiled"


def make_compiled_runner(compiled_obj: Any) -> Callable[..., Any]:
    """Wrap a ``_CompiledResult`` into a user-facing callable.

    Builds the standard "prepare_inputs -> gm(*flat_inps) -> commit_step_state"
    runtime adapter that every recir orchestrator (``recir.compile``,
    :func:`recir.training.compile.compile_step`, and custom dispatchers in
    tests) emits after stage 10.
    The wrapper:

    * pops ``last_train_step`` from ``kwargs`` (default ``False``);
    * routes user args through :meth:`compiled_obj.state_manager.prepare_inputs`
      to flatten and dedup against ``compiled_obj.flat_state``;
    * invokes ``compiled_obj.gm(*flat_inps)``, falling back to a plain call
      when the GM doesn't accept the ``last_iter`` kwarg (i.e. when no
      :class:`IterGraphModule` wrap was applied);
    * splits the epilogue between
      :meth:`compiled_obj.state_manager.commit_step_state` (make the step's
      state outputs the resident state) and :meth:`~materialize_output` (hand
      the caller the step's output, unless it armed ``keep_output_jax`` to pull
      it itself later);
    * stashes ``compiled_obj`` on the wrapper's ``__dict__`` under
      :data:`COMPILED_OBJECT_KEY` so :func:`sync_device_state_to_torch` can
      find it.

    Args:
        compiled_obj: a :class:`_CompiledResult` returned by
            :func:`compile_from_gm` (or the legacy
            :func:`_compile_and_transform`).

    Returns:
        A callable with the same signature as the original train step.
        Pass ``last_train_step=True`` on the final call so IterGraph-wrapped
        modules can flush their state.
    """

    # Introspect once per wrapper: determine whether the underlying GM's
    # ``forward`` accepts a ``last_iter`` parameter (i.e. it's an
    # IterGraphModule wrap). Avoids relying on TypeError substring matching,
    # which is fragile across Python/torch versions. Explicit ``bool``
    # annotation is required because the value is captured by the ``runner``
    # closure below and pyre [53] flags unannotated captured locals.
    accepts_last_iter: bool
    accepts_input_ready_event: bool
    try:
        gm_forward = compiled_obj.gm.forward
        gm_sig = inspect.signature(gm_forward)
        accepts_last_iter = "last_iter" in gm_sig.parameters
        accepts_input_ready_event = "input_ready_event" in gm_sig.parameters
    except (TypeError, ValueError):
        accepts_last_iter = False
        accepts_input_ready_event = False

    def runner(*call_args: Any, **call_kwargs: Any) -> Any:
        last_train_step = call_kwargs.pop("last_train_step", False)
        input_ready_event: torch.cuda.Event | None = call_kwargs.pop(
            "input_ready_event", None
        )
        user_inps = pytree.arg_tree_leaves(*call_args, **call_kwargs)
        flat_inps = compiled_obj.state_manager.prepare_inputs(
            compiled_obj.gm,
            compiled_obj.flat_state,
            user_inps,
            last_train_step,
        )
        with torch.no_grad():
            gm_kwargs: dict[str, Any] = {}
            if last_train_step and accepts_last_iter:
                # Final iter: pass ``last_iter`` so IterGraphModule can flush.
                gm_kwargs["last_iter"] = last_train_step
            if input_ready_event is not None and accepts_input_ready_event:
                gm_kwargs["input_ready_event"] = input_ready_event
            output_all = compiled_obj.gm(*flat_inps, **gm_kwargs)
            if is_pipeline_pending(output_all):
                return output_all
            output = output_all[0]
            updated_flat = pytree.tree_leaves(output_all[1:])
            state_manager = compiled_obj.state_manager
            state_manager.commit_step_state(
                compiled_obj.flat_state, updated_flat, last_train_step
            )
            if state_manager.defers_output(last_train_step):
                return output
            return state_manager.materialize_output(output)

    runner.__dict__[COMPILED_OBJECT_KEY] = compiled_obj
    return runner


def prepare_compiled(
    compiled_fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> bool:
    """Compile and cache a RecIR callable without executing its graph."""
    prepare = compiled_fn.__dict__.get(PREPARE_COMPILED_ATTR)
    if not callable(prepare):
        return False
    prepare(*args, **kwargs)
    return True


def sync_device_state_to_torch(
    compiled_fn: Callable[..., Any],
    skip_fqns: Sequence[str] | None = None,
) -> None:
    """Synchronize device-resident state back to PyTorch tensors.
    Call before checkpointing or inspecting model parameters.

    ``skip_fqns``: if set, state entries whose FQN contains any of these
    substrings are NOT copied back to host. Use for per-step syncs to skip large
    device-resident tensors (e.g. on-device embedding-table shards, matched by
    their module prefix) whose host copy is only needed at checkpoint time; pass
    ``None`` (default) for a full sync before checkpointing.
    """
    compiled_obj = compiled_fn.__dict__.get(COMPILED_OBJECT_KEY, None)
    if compiled_obj is None:
        return
    if compiled_obj.state_manager.device_state is None:
        return
    compiled_obj.state_manager.copy_state_to_host(
        compiled_obj.flat_state, skip_fqns=skip_fqns
    )


def sync_named_state_to_torch(
    compiled_fn: Callable[..., Any], fqn_substrings: Sequence[str]
) -> None:
    """Copy back only the state entries whose FQN matches one of the substrings."""
    compiled_obj = compiled_fn.__dict__.get(COMPILED_OBJECT_KEY, None)
    if compiled_obj is None or compiled_obj.state_manager.device_state is None:
        return
    compiled_obj.state_manager.copy_named_state_to_host(
        compiled_obj.flat_state, fqn_substrings
    )


def set_compiled_output_deferred(
    compiled_fn: Callable[..., Any], deferred: bool
) -> bool:
    """Choose whether an XLA compiled call returns device-resident outputs.

    Returns ``False`` before the callable has compiled. Callers can use that
    signal to process the first synchronous result normally, then defer later
    results once the device-state manager exists.
    """
    compiled_obj = compiled_fn.__dict__.get(COMPILED_OBJECT_KEY, None)
    if compiled_obj is None:
        return False
    compiled_obj.state_manager.keep_output_jax = deferred
    return deferred


def materialize_compiled_output(compiled_fn: Callable[..., Any], output: Any) -> Any:
    """Materialize an output retained by ``set_compiled_output_deferred``.

    Before the first compilation, or for a backend whose outputs are already
    Torch tensors, this is a no-op.
    """
    compiled_obj = compiled_fn.__dict__.get(COMPILED_OBJECT_KEY, None)
    if compiled_obj is None:
        return output
    return compiled_obj.state_manager.materialize_output(output)


def get_compiled_process_state(
    compiled_fn: Callable[..., Any],
) -> CompiledProcessState | None:
    """Return the canonical post-layout host state after compilation."""
    compiled_obj = compiled_fn.__dict__.get(COMPILED_OBJECT_KEY, None)
    if compiled_obj is None:
        return None
    process_state = getattr(compiled_obj, "process_state", None)
    if not isinstance(process_state, CompiledProcessState):
        return None
    return process_state


def invalidate_device_state(compiled_fn: Callable[..., Any]) -> None:
    """Discard cached device-resident state so the next call reads PyTorch state."""
    compiled_obj = compiled_fn.__dict__.get(COMPILED_OBJECT_KEY, None)
    if compiled_obj is None:
        return
    compiled_obj.state_manager.invalidate_device_state()
