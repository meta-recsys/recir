# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Ahead-of-time prebuild + serialize + load for recir XLA executables.

Pattern: build (XLA HLO + PJRT codegen) once on a single high-RAM CPU
host using ``jax.experimental.topologies.get_topology_desc`` to
materialize a "compile-only" mesh mirroring the target accelerator
topology, serialize each per-subgraph compiled binary via
``jax.experimental.serialize_executable.serialize``, and persist. At
training time, every accelerator pod calls ``deserialize_and_load``
instead of building, eliminating the host-RAM build spike that otherwise
hits every host in parallel.

Vocabulary note: the build step inside ``_XLAModule._build_xla_executable``
is what JAX's ``.lower()`` + ``.compile()`` produce. Calling this "AOT
prebuild" (rather than "AOT compile") keeps the verb consistent with the
rest of the recir vocabulary — see P2336892697.

Public surface stays transport-agnostic. The serialize side returns a
plain ``dict[str, tuple[bytes, in_tree, out_tree]]`` keyed by qualified
subgraph paths; callers pickle it (or ship it through any other
PyTreeDef-aware transport) themselves.

Usage:

    # On a CPU host whose default mesh is a virtual TPU topology:
    from recir.lowering.aot import (
        aot_prebuild_and_serialize, aot_load_prebuilt_into,
    )
    from recir.lowering.xla_lower import aot_topology

    topo = get_topology_desc(platform="tpu", topology="2x2x1")
    with aot_topology(topo):
        artifacts = aot_prebuild_and_serialize(compiled_obj.gm)
    pickle.dump(artifacts, open("/path/to/aot.bin", "wb"))

    # On every accelerator pod:
    artifacts = pickle.load(open("/path/to/aot.bin", "rb"))
    aot_load_prebuilt_into(compiled_obj.gm, artifacts)
    # ``compiled_obj.gm(...)`` now skips ``_build_xla_executable`` and
    # reuses the shipped XLA binary on every ``_XLAModule``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator, List, Tuple

import torch
import torch.fx as fx
from jax.experimental import serialize_executable  # @manual
from recir.lowering.xla_lower import _XLAModule
from recir.passes.iter_graph_module import IterGraphModule


logger: logging.Logger = logging.getLogger(__name__)


# Tuple of (serialized_bytes, in_pytree_def, out_pytree_def). Returned as a
# plain tuple rather than a dataclass so the public API doesn't commit to a
# specific scaffolding shape — callers can unpack and re-pack into any
# transport they want.
PrebuiltArtifact = Tuple[bytes, Any, Any]


def _walk_recir_compiled(
    gm: fx.GraphModule | IterGraphModule,
) -> Iterator[Tuple[str, str, _XLAModule]]:
    """Yield ``(gm_name, sub_path, _XLAModule)`` for every ``_XLAModule``
    inside ``gm``.

    For an :class:`IterGraphModule`, delegates to
    :meth:`IterGraphModule.named_xla_modules` so the (setup_gm / main_gm
    / cleanup_gm) walk lives on the owning type. For a plain
    :class:`fx.GraphModule`, walks via ``named_modules`` with
    ``gm_name=""``.
    """
    if isinstance(gm, IterGraphModule):
        yield from gm.named_xla_modules()
    elif isinstance(gm, fx.GraphModule):
        for sub_path, mod in gm.named_modules():
            if isinstance(mod, _XLAModule):
                yield "", sub_path, mod
    else:
        raise TypeError(
            f"Expected fx.GraphModule or IterGraphModule, got {type(gm).__name__}"
        )


def _key(gm_name: str, sub_path: str) -> str:
    """Flat string key for the artifact dict."""
    return f"{gm_name}::{sub_path}" if gm_name else sub_path


def _synthesize_example_args(mod: _XLAModule) -> List[torch.Tensor]:
    """Build dummy CPU tensors for each placeholder in ``mod.gm``.

    ``segment_and_lower`` lifts every subgraph input as a placeholder
    whose ``meta["val"]`` carries the FakeTensor produced during fx
    tracing. Reading shape + dtype from there means the AOT path does
    not need the caller to construct per-subgraph args (subgraph leaves
    are flattened tensors, not the original ``params_dict`` / batch
    tuple the user fed the parent ``gm``) and avoids running the parent
    ``gm`` end-to-end just to reach the subgraph.
    """
    placeholders = [n for n in mod.gm.graph.nodes if n.op == "placeholder"]
    out: List[torch.Tensor] = []
    for p in placeholders:
        val = p.meta.get("val", None)
        if val is None:
            raise RuntimeError(
                f"placeholder {p.name!r} of subgraph has no meta['val']; "
                "cannot synthesize an example tensor"
            )
        shape = tuple(int(s) for s in val.shape)
        out.append(torch.zeros(shape, dtype=val.dtype, device="cpu"))
    return out


def aot_prebuild_and_serialize(
    gm: fx.GraphModule | IterGraphModule,
) -> dict[str, PrebuiltArtifact]:
    """Drive eager build of every ``_XLAModule`` and serialize each.

    For each :class:`_XLAModule` reachable from ``gm``, this function:

      1. Synthesizes dummy CPU tensors from each subgraph's placeholder
         ``meta["val"]`` (see :func:`_synthesize_example_args`).
      2. Invokes ``mod(*dummy)`` once. The lazy-build path inside
         ``_build_*_executable`` materializes a ``jax.stages.Compiled``
         and stashes it on ``mod._jax_compiled``.
      3. Serializes ``mod._jax_compiled`` via
         ``jax.experimental.serialize_executable.serialize``.

    The single-warmup-call pattern is the only correct way to drive
    backend-build of ``_XLAModule``: two of the three backends
    (SHARD_MAP-with-collectives and GSPMD) only finalize the executable
    on first ``run`` invocation, so calling
    ``_build_xla_executable`` in isolation does not capture the
    artifact for those backends.

    Args:
        gm: Result of ``recir.compile(...)``. Either a bare
            ``fx.GraphModule`` or an ``IterGraphModule``.

    Returns:
        A dict mapping qualified subgraph keys to
        :data:`PrebuiltArtifact` tuples ``(serialized_bytes, in_tree,
        out_tree)``. The dict is pickleable; callers persist it however
        they like.

    Raises:
        RuntimeError: If any subgraph fails to populate ``_jax_compiled``
            after a warmup call (typically a backend bug or missing
            ``meta["val"]`` annotations on placeholders).
    """
    artifacts: dict[str, PrebuiltArtifact] = {}
    seen = 0
    for gm_name, sub_path, mod in _walk_recir_compiled(gm):
        seen += 1
        dummy_args = _synthesize_example_args(mod)
        # Single forward call drives the lazy-build path inside
        # ``_build_*_executable`` and populates ``mod._jax_compiled``.
        mod(*dummy_args)
        if mod._jax_compiled is None:
            raise RuntimeError(
                f"AOT warmup did not populate _jax_compiled for "
                f"{gm_name!r}/{sub_path!r}; backend={mod.backend_type.value}"
            )
        bin_, in_tree, out_tree = serialize_executable.serialize(mod._jax_compiled)
        artifacts[_key(gm_name, sub_path)] = (bin_, in_tree, out_tree)
        logger.info(
            "AOT: serialized %s/%s (%d bytes, backend=%s)",
            gm_name,
            sub_path,
            len(bin_),
            mod.backend_type.value,
        )
    logger.info("AOT: walked %d subgraphs, serialized %d", seen, len(artifacts))
    return artifacts


def aot_load_prebuilt_into(
    gm: fx.GraphModule | IterGraphModule,
    artifacts: dict[str, PrebuiltArtifact],
    *,
    backend: str | None = None,
    execution_devices: Iterable[Any] | None = None,
    strict: bool = True,
) -> int:
    """Install pre-built executables on every ``_XLAModule`` in ``gm``.

    After this call, the next ``_XLAModule.forward`` invocation on each
    matched subgraph reuses the loaded executable instead of triggering
    ``_build_xla_executable``.

    Args:
        gm: same as for :func:`aot_prebuild_and_serialize`.
        artifacts: dict produced by :func:`aot_prebuild_and_serialize`.
        backend: ``jax`` backend name (``"tpu"``, ``"cpu"``, ``"gpu"``)
            or an explicit PJRT client. ``None`` (default) lets
            :mod:`jax.experimental.serialize_executable` pick the
            platform of the first ``jax.devices()`` entry.
        execution_devices: explicit device list to bind the loaded
            executable to. ``None`` (default) lets the same module use
            ``backend.devices()``. For multi-host training pass the
            global device list returned by ``jax.devices()`` here, NOT
            ``jax.local_devices()``.
        strict: when ``True`` (default), raise if any ``_XLAModule``
            lacks a matching artifact — a missing artifact in production
            would silently bring back the per-host build spike that AOT
            exists to prevent. Tests can pass ``strict=False`` to
            tolerate partial bundles.

    Returns:
        Number of subgraphs whose executable was installed.
    """
    devices_seq = list(execution_devices) if execution_devices is not None else None
    # Materialize the walk once so we can do a pre-flight missing-key check
    # under ``strict=True`` BEFORE mutating any module. Without this, a
    # strict-mode failure would leave ``gm`` partially mutated (some
    # _XLAModule._jax_compiled populated, others not) which a caller that
    # catches ``RuntimeError`` would observe as a half-loaded gm.
    walk = [
        (gm_name, sub_path, mod) for gm_name, sub_path, mod in _walk_recir_compiled(gm)
    ]
    if strict:
        missing_preflight = [
            _key(gm_name, sub_path)
            for gm_name, sub_path, _mod in walk
            if artifacts.get(_key(gm_name, sub_path)) is None
        ]
        if missing_preflight:
            raise RuntimeError(
                f"aot_load_prebuilt_into(strict=True): {len(missing_preflight)} subgraphs "
                f"have no matching artifact and would fall back to lazy build, "
                f"reintroducing the per-host build spike: {missing_preflight[:5]}"
                + (
                    f" (and {len(missing_preflight) - 5} more)"
                    if len(missing_preflight) > 5
                    else ""
                )
            )
    loaded = 0
    for gm_name, sub_path, mod in walk:
        key = _key(gm_name, sub_path)
        art = artifacts.get(key)
        if art is None:
            logger.info(
                "AOT: no artifact for %s/%s; falling back to lazy build",
                gm_name,
                sub_path,
            )
            continue
        bin_, in_tree, out_tree = art
        compiled = serialize_executable.deserialize_and_load(
            bin_,
            in_tree,
            out_tree,
            backend=backend,
            execution_devices=devices_seq,
        )
        mod._jax_compiled = compiled
        logger.info(
            "AOT: loaded %s/%s (backend=%s)",
            gm_name,
            sub_path,
            mod.backend_type.value,
        )
        loaded += 1
    return loaded
