# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""AOT (ahead-of-time) compile + serialize + load for recir XLA executables.

Pattern: compile once on a single high-RAM CPU host using
``jax.experimental.topologies.get_topology_desc`` to materialize a
"compile-only" mesh that mirrors the target accelerator topology,
serialize each per-subgraph compiled binary via
``jax.experimental.serialize_executable.serialize``, and persist the
serialized payloads to disk. At training time, every accelerator pod
calls ``deserialize_and_load`` instead of compiling, eliminating the
host-RAM compile spike that otherwise hits every host in parallel.

This module deliberately stays transport-agnostic: the public surface
returns/accepts plain ``dict[str, AOTArtifact]`` payloads keyed by
fully-qualified subgraph paths, so callers can persist them to local
disk, distributed object stores, or anywhere else. The included
``save_artifacts`` / ``load_artifacts`` helpers exist for the common
single-file local-disk case and for tests.

Usage:

    # On a CPU host whose default mesh is a virtual TPU topology
    # produced by ``jax.experimental.topologies.get_topology_desc``:
    artifacts = aot_compile_and_serialize(
        compiled_obj, example_args_per_gm
    )
    save_artifacts(artifacts, "/path/to/aot.bin")

    # On every accelerator pod (no virtual topology):
    artifacts = load_artifacts("/path/to/aot.bin")
    aot_load_into(compiled_obj, artifacts)
    # ``compiled_obj.gm(...)`` now skips ``_compile_xla`` and reuses the
    # shipped XLA binary on every ``_XLAModule``.
"""

from __future__ import annotations

import dataclasses
import logging
import pickle
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.fx as fx
from jax.experimental import serialize_executable  # @manual
from recir.iter_graph_module import IterGraphModule
from recir.lowering.xla_lower import _XLAModule


logger: logging.Logger = logging.getLogger(__name__)

# Subgraph paths inside an ``IterGraphModule`` are scoped under the
# ``setup_gm`` / ``main_gm`` / ``cleanup_gm`` attributes. Centralizing
# the tuple here keeps the walk and the load paths consistent and lets
# tests / callers introspect the contract without re-deriving it.
_ITER_GM_NAMES: Tuple[str, ...] = ("setup_gm", "main_gm", "cleanup_gm")


@dataclasses.dataclass
class AOTArtifact:
    """A single serialized XLA executable plus its pytree metadata.

    ``serialized`` is the opaque PJRT byte string returned by
    ``jax.experimental.serialize_executable.serialize``; ``in_tree`` and
    ``out_tree`` are ``jax.tree_util.PyTreeDef`` instances that describe
    the input/output structure expected by the compiled program. All
    three are pickleable, so an entire ``dict[str, AOTArtifact]`` can be
    written to disk with the standard ``pickle`` module.
    """

    serialized: bytes
    in_tree: Any
    out_tree: Any


def _walk_xla_modules(gm: fx.GraphModule) -> Iterable[Tuple[str, _XLAModule]]:
    """Yield ``(qualified_name, _XLAModule)`` for every ``_XLAModule``
    reachable from ``gm`` via ``named_modules``.

    Uses ``named_modules`` (rather than the bare ``modules()`` iterator)
    so call sites can key artifacts by stable, human-readable paths
    instead of insertion order.
    """
    for name, mod in gm.named_modules():
        if isinstance(mod, _XLAModule):
            yield name, mod


def _walk_recir_compiled(
    gm: fx.GraphModule | IterGraphModule,
) -> Iterable[Tuple[str, str, _XLAModule]]:
    """Yield ``(gm_name, sub_path, _XLAModule)`` for every ``_XLAModule``
    inside ``gm``.

    For an :class:`IterGraphModule`, ``gm_name`` is one of
    ``"setup_gm"`` / ``"main_gm"`` / ``"cleanup_gm"``. For a plain
    :class:`fx.GraphModule`, ``gm_name`` is the empty string. Callers
    use the full ``(gm_name, sub_path)`` tuple as a stable artifact key.
    """
    if isinstance(gm, IterGraphModule):
        for gm_name in _ITER_GM_NAMES:
            sub_gm = getattr(gm, gm_name, None)
            if sub_gm is None:
                continue
            for sub_path, mod in _walk_xla_modules(sub_gm):
                yield gm_name, sub_path, mod
    elif isinstance(gm, fx.GraphModule):
        for sub_path, mod in _walk_xla_modules(gm):
            yield "", sub_path, mod
    else:
        raise TypeError(
            f"Expected fx.GraphModule or IterGraphModule, got {type(gm).__name__}"
        )


def _key(gm_name: str, sub_path: str) -> str:
    """Flat string key for the artifact dict.

    Using a string (rather than a tuple) keeps the dict trivially JSON /
    repr / Manifold-blob friendly without losing the gm-name namespace.
    """
    return f"{gm_name}::{sub_path}" if gm_name else sub_path


def _synthesize_example_args(mod: _XLAModule) -> List[torch.Tensor]:
    """Build dummy CPU tensors for each placeholder in ``mod.gm``.

    ``segment_and_lower`` lifts every subgraph input as a placeholder
    whose ``meta["val"]`` carries the FakeTensor produced during fx
    tracing. Reading shape + dtype from there means the AOT path does
    not need the caller to construct per-subgraph args (which would be
    awkward — subgraph leaves are flattened tensors, not the original
    ``params_dict`` / batch tuple the user fed the parent ``gm``) and
    avoids running the parent ``gm`` end-to-end just to reach the
    subgraph in the right shape.

    Returns CPU tensors; the JAX bridge (``_to_sharded_jax`` /
    ``_t2j_safe``) handles transferring them to the backend device.
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
        # FakeTensor carries shape + dtype. ``torch.zeros`` keeps the
        # warmup deterministic and avoids the cost of random init.
        shape = tuple(int(s) for s in val.shape)
        out.append(torch.zeros(shape, dtype=val.dtype, device="cpu"))
    return out


def aot_compile_and_serialize(
    gm: fx.GraphModule | IterGraphModule,
) -> dict[str, AOTArtifact]:
    """Drive eager compile of every ``_XLAModule`` and serialize each.

    For each :class:`_XLAModule` reachable from ``gm``, this function:

      1. Synthesizes dummy CPU tensors from each subgraph's placeholder
         ``meta["val"]`` (see :func:`_synthesize_example_args`).
      2. Invokes ``mod(*dummy)`` once. The lazy-compile path inside
         ``_jit_*`` materializes a ``jax.stages.Compiled`` and stashes
         it on ``mod._jax_compiled``.
      3. Serializes ``mod._jax_compiled`` via
         ``jax.experimental.serialize_executable.serialize``.

    The single-warmup-call pattern is the recommended way to drive
    compile of ``_XLAModule``: two of the three backends
    (SHARD_MAP-with-collectives and GSPMD) only finalize compilation
    on first ``run`` invocation, so calling ``_compile_xla`` in
    isolation does not capture the executable for those backends.

    Args:
        gm: Result of ``recir.compile(...)``. Either a bare
            ``fx.GraphModule`` or an ``IterGraphModule`` (the production
            path with ``enable_iter_graph=True``).

    Returns:
        A dict mapping fully-qualified subgraph keys (see :func:`_key`)
        to :class:`AOTArtifact` payloads. Callers persist this dict
        with :func:`save_artifacts` (or any other ``pickle``-compatible
        writer).

    Raises:
        RuntimeError: If any subgraph fails to populate ``_jax_compiled``
            after a warmup call (typically a backend bug or missing
            ``meta["val"]`` annotations on placeholders).
    """
    artifacts: dict[str, AOTArtifact] = {}
    seen = 0
    for gm_name, sub_path, mod in _walk_recir_compiled(gm):
        seen += 1
        dummy_args = _synthesize_example_args(mod)
        # Drive a single forward call so the lazy-compile path inside
        # the ``_jit_*`` helpers populates ``mod._jax_compiled``. We do
        # not catch exceptions here: if the warmup fails, the subgraph
        # cannot be AOT'd in the current environment and the caller
        # should know.
        mod(*dummy_args)
        if mod._jax_compiled is None:
            raise RuntimeError(
                f"AOT warmup did not populate _jax_compiled for "
                f"{gm_name!r}/{sub_path!r}; backend={mod.backend_type.value}"
            )
        bin_, in_tree, out_tree = serialize_executable.serialize(mod._jax_compiled)
        artifacts[_key(gm_name, sub_path)] = AOTArtifact(
            serialized=bin_, in_tree=in_tree, out_tree=out_tree
        )
        logger.info(
            "AOT: serialized %s/%s (%d bytes, backend=%s)",
            gm_name,
            sub_path,
            len(bin_),
            mod.backend_type.value,
        )
    logger.info("AOT: walked %d subgraphs, serialized %d", seen, len(artifacts))
    return artifacts


def save_artifacts(artifacts: dict[str, AOTArtifact], path: str) -> None:
    """Pickle ``artifacts`` to ``path``.

    The pickle protocol is fixed at the highest available ASCII protocol
    so the resulting file is portable across Python minor versions.

    NOTE: ``AOTArtifact.serialized`` is itself a pickle stream produced
    by ``jax.experimental.serialize_executable``'s custom Pickler — JAX
    has no non-pickle serialization for compiled XLA executables, so any
    AOT-shipping path is going to involve pickle. Read AOT bundles only
    from trusted sources (your own compile job's output blob).
    """
    with open(path, "wb") as f:
        pickle.dump(  # noqa: PYTHONPICKLEISBAD — see docstring for rationale
            artifacts, f, protocol=pickle.HIGHEST_PROTOCOL
        )


def load_artifacts(path: str) -> dict[str, AOTArtifact]:
    """Inverse of :func:`save_artifacts`. Same trust requirements apply."""
    with open(path, "rb") as f:
        return pickle.load(  # noqa: PYTHONPICKLEISBAD — see save_artifacts
            f
        )


def aot_load_into(
    gm: fx.GraphModule | IterGraphModule,
    artifacts: dict[str, AOTArtifact],
    *,
    backend: Optional[str] = None,
    execution_devices: Optional[Iterable[Any]] = None,
) -> int:
    """Install pre-compiled executables on every ``_XLAModule`` in ``gm``.

    After this call, the next ``_XLAModule.forward`` invocation on each
    matched subgraph reuses the loaded executable instead of triggering
    ``_compile_xla``. Subgraphs without a matching artifact key fall
    back to the regular lazy-compile path.

    Args:
        gm: same as for :func:`aot_compile_and_serialize`.
        artifacts: dict produced by :func:`aot_compile_and_serialize`.
        backend: ``jax`` backend name (``"tpu"``, ``"cpu"``, ``"gpu"``)
            or an explicit ``xla_client.Client`` instance. ``None``
            (default) defers to ``jax.experimental.serialize_executable``,
            which picks the platform of the first device returned by
            ``jax.devices()``.
        execution_devices: explicit device list to bind the loaded
            executable to. ``None`` (default) defers to the same
            module, which uses ``backend.devices()``. For multi-host
            training pass the global device list returned by
            ``jax.devices()`` here, NOT ``jax.local_devices()``.

    Returns:
        Number of subgraphs whose executable was installed.
    """
    loaded = 0
    devices_seq = list(execution_devices) if execution_devices is not None else None
    for gm_name, sub_path, mod in _walk_recir_compiled(gm):
        key = _key(gm_name, sub_path)
        art = artifacts.get(key)
        if art is None:
            logger.info(
                "AOT: no artifact for %s/%s; falling back to lazy compile",
                gm_name,
                sub_path,
            )
            continue
        compiled = serialize_executable.deserialize_and_load(
            art.serialized,
            art.in_tree,
            art.out_tree,
            backend=backend,
            execution_devices=devices_seq,
        )
        mod._jax_compiled = compiled
        mod._aot_loaded = True
        logger.info(
            "AOT: loaded %s/%s (backend=%s)",
            gm_name,
            sub_path,
            mod.backend_type.value,
        )
        loaded += 1
    return loaded
