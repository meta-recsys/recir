# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Bundle consumer: load the artifacts produced by ``recir.bundle_export``.

Public entry point: :func:`import_bundle`.

Responsibilities:

* parse ``manifest.json`` and reject any unsupported
  :attr:`BundleManifest.bundle_format_version`;
* recompute every artifact's sha256 against
  :attr:`BundleManifest.artifact_index` and abort on mismatch;
* load ``constants.safetensors`` and seed the FX deserializer with it so
  ``get_attr`` constants are registered on the GM as
  :data:`~torch.export.unflatten._AttrKind.CONSTANT` (NOT silently as
  buffers);
* parse ``train.binding.json`` so the consumer can walk the placeholder
  ordering (constants are excluded from :meth:`BindingTable.call_vector`
  by design);
* load ``model_state.safetensors`` and assemble FQN-keyed ``params`` /
  ``buffers`` dicts by walking the :class:`BindingTable`.

The producer is forward-only (no optimizer state in the bundle); the
loader returns an empty ``named_states`` dict so the kwargs-only
:meth:`recir.distributed.parallel_mode.ParallelMode.partition` signature
still receives the slot it expects. Optimizer reconstruction is the
caller's responsibility, driven by the consumer's own launch config.

The loader does NOT reconstruct an :class:`nn.Module` or
:class:`torch.optim.Optimizer`.  It returns a :class:`LoadedBundle` whose
fields plug directly into the kwargs-only
:meth:`recir.distributed.parallel_mode.ParallelMode.partition` signature
via :func:`recir.bundle_runtime.runtime.wrap_and_lower`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import torch
import torch.fx as fx
import torch.nn as nn
from recir.bundle_export.exporter import (
    BINDING_FILENAME,
    CONSTANTS_FILENAME,
    GRAPH_FILENAME,
    MANIFEST_FILENAME,
    MODEL_STATE_FILENAME,
)
from recir.bundle_export.manifest import (
    BUNDLE_FORMAT_VERSION,
    BundleManifest,
    manifest_from_json,
)
from recir.serde import BindingTable, deserialize_fx_graph, parse_bindings


@dataclass
class LoadedBundle:
    """In-memory view of a bundle on disk.

    Attributes:
        gm: The deserialized FX :class:`torch.fx.GraphModule`.  Tensor
            ``get_attr`` constants are already registered on the GM as
            :data:`~torch.export.unflatten._AttrKind.CONSTANT` attributes
            (the deserializer seeds them BEFORE
            :class:`fx.GraphModule.__init__` runs and reflips any
            silently-registered buffers afterward).
        bindings: The :class:`BindingTable` parsed from the bundle's
            binding sidecar.  The producer recorded constants alongside
            placeholder bindings, but :meth:`BindingTable.call_vector`
            already filters them out -- consumers walk that method to
            obtain the runtime-call placeholder ordering.
        params: FQN-keyed module parameters
            (from ``mod.named_parameters()`` at export time).
        buffers: FQN-keyed module buffers
            (from ``mod.named_buffers()`` at export time).
        named_states: Always empty for forward-only bundles (the producer
            is forward-only by construction). Surfaced as a field so the
            kwargs-only ``parallel_mode.partition`` signature still has a
            ``named_states`` argument to receive when the loader's output
            is fed straight through.
        constants: FQN -> tensor map for ``get_attr``-backed constants.
            **Metadata only**: the values are already attached to ``gm``
            as ``_AttrKind.CONSTANT`` attributes, so consumers do NOT
            thread them into the runtime call vector.  Surfaced here for
            inspection / cache-warming purposes.
        manifest: The parsed :class:`BundleManifest`.  The consumer
            owns optimizer construction via its own launch config; the
            manifest carries no optimizer hints.
    """

    gm: fx.GraphModule
    bindings: BindingTable
    params: Dict[str, nn.Parameter]
    buffers: Dict[str, torch.Tensor]
    named_states: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)
    constants: Dict[str, torch.Tensor] = field(default_factory=dict)
    manifest: BundleManifest | None = None


def _sha256_of_file(path: Path) -> str:
    """Recompute the sha256 of a file in 1 MiB chunks.

    Mirrors :func:`recir.bundle_export.exporter._sha256_of_file` so the
    consumer-side digest matches the producer-side digest byte-for-byte.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksums(bundle_dir: Path, manifest: BundleManifest) -> None:
    """Recompute and verify each artifact's sha256 against the manifest.

    A mismatch aborts the load BEFORE any tensor data is touched, so a
    corrupted bundle never reaches the deserializer or the
    safetensors loader.
    """
    for fname, expected_sha in manifest.artifact_index.items():
        path = bundle_dir / fname
        if not path.is_file():
            raise RuntimeError(
                f"Bundle artifact missing: {fname!r} (expected at {path})"
            )
        actual_sha = _sha256_of_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Bundle checksum mismatch for {fname!r}: "
                f"expected {expected_sha}, got {actual_sha}"
            )


def _gate_format_version(manifest: BundleManifest) -> None:
    """Reject any bundle whose ``bundle_format_version`` we do not understand.

    The check is symmetric -- ANY value other than the currently-supported
    :data:`recir.bundle_export.manifest.BUNDLE_FORMAT_VERSION` is rejected,
    not just a specific legacy version.  The error message cites both the
    rejected value and the supported value so an operator inspecting the
    log can tell at a glance whether to refresh the producer or the
    consumer.
    """
    if manifest.bundle_format_version != BUNDLE_FORMAT_VERSION:
        raise RuntimeError(
            f"Unsupported bundle_format_version "
            f"{manifest.bundle_format_version!r}; this consumer "
            f"supports {BUNDLE_FORMAT_VERSION!r} only. Use a matching "
            f"producer/consumer pair."
        )


def _load_manifest(bundle_dir: Path) -> BundleManifest:
    """Parse ``manifest.json`` from the bundle directory."""
    manifest_path = bundle_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Bundle is missing {MANIFEST_FILENAME!r} at {manifest_path}"
        )
    return manifest_from_json(manifest_path.read_text())


def _load_safetensors(path: Path) -> Dict[str, torch.Tensor]:
    """Read a safetensors file into a flat FQN -> tensor dict.

    The producer always writes the file (possibly empty) for layout
    uniformity, so an absent file is a real bundle integrity error
    rather than a benign empty case.
    """
    # Local import so callers without the safetensors dep can still
    # import this module's other helpers.
    from safetensors.torch import load_file

    if not path.is_file():
        raise RuntimeError(f"Bundle artifact missing: {path}")
    return load_file(str(path))


def _load_binding_table(bundle_dir: Path) -> BindingTable:
    """Parse ``train.binding.json`` into a :class:`BindingTable`.

    Routes through :func:`recir.serde.parse_bindings` so the binding
    sidecar shape stays in lock-step with what the FX serializer emits
    (single top-level dict with a ``"bindings"`` key holding a list of
    per-binding dicts).
    """
    binding_path = bundle_dir / BINDING_FILENAME
    if not binding_path.is_file():
        raise RuntimeError(f"Bundle is missing {BINDING_FILENAME!r} at {binding_path}")
    payload = json.loads(binding_path.read_text())
    table = parse_bindings(payload)
    if table is None:
        raise RuntimeError(
            f"{BINDING_FILENAME!r} parsed to None; expected a BindingTable payload"
        )
    return table


def _assemble_params_and_buffers(
    bindings: BindingTable,
    model_state: Dict[str, torch.Tensor],
) -> tuple[Dict[str, nn.Parameter], Dict[str, torch.Tensor]]:
    """Walk the BindingTable to split ``model_state`` into params + buffers.

    The producer wrote both into a single safetensors file (FQN-keyed,
    same namespace) but recorded the kind in the BindingTable so the
    consumer can rehydrate parameters as :class:`nn.Parameter` (with
    ``requires_grad=True``) and buffers as plain tensors.  Iterating in
    binding order preserves the dict insertion order the producer wrote.
    """
    params: Dict[str, nn.Parameter] = {}
    buffers: Dict[str, torch.Tensor] = {}
    for binding in bindings.bindings:
        if binding.kind == "param":
            if binding.fqn not in model_state:
                raise RuntimeError(
                    f"Model state safetensors missing param {binding.fqn!r} "
                    f"required by BindingTable"
                )
            # Wrap as nn.Parameter so the type matches the producer side
            # (``mod.named_parameters()`` returns Parameter instances).
            # ``requires_grad=True`` keeps autograd-aware passes
            # recognising it as a trainable weight.
            params[binding.fqn] = nn.Parameter(
                model_state[binding.fqn], requires_grad=True
            )
        elif binding.kind == "buffer":
            if binding.fqn not in model_state:
                raise RuntimeError(
                    f"Model state safetensors missing buffer {binding.fqn!r} "
                    f"required by BindingTable"
                )
            buffers[binding.fqn] = model_state[binding.fqn]
    return params, buffers


def import_bundle(bundle_dir: str | Path) -> LoadedBundle:
    """Load every artifact in a bundle directory into a :class:`LoadedBundle`.

    Steps:

      1. Parse ``manifest.json`` and reject any unsupported
         ``bundle_format_version``.
      2. Recompute each artifact's sha256 against the manifest's
         ``artifact_index``; abort on mismatch.
      3. Load ``constants.safetensors`` so the FX deserializer can seed
         ``get_attr`` constants on the GM as
         :data:`~torch.export.unflatten._AttrKind.CONSTANT` attributes
         BEFORE :class:`fx.GraphModule.__init__` runs (the canonical
         seed-before-construct dance from
         :func:`recir.serde.deserialize_fx_graph`).
      4. Deserialize ``train_step.fx.json`` with the constants dict.
      5. Parse ``train.binding.json`` into a :class:`BindingTable`.
      6. Load ``model_state.safetensors`` and walk the BindingTable to
         assemble FQN-keyed ``params`` / ``buffers`` dicts.

    Args:
        bundle_dir: Path to the bundle directory.

    Returns:
        A populated :class:`LoadedBundle`.

    Raises:
        RuntimeError: On a missing artifact, checksum mismatch,
            unsupported format version, BindingTable inconsistency, or
            safetensors key gap.
    """
    p = Path(bundle_dir)
    if not p.is_dir():
        raise RuntimeError(f"Bundle path is not a directory: {p}")

    manifest = _load_manifest(p)
    _gate_format_version(manifest)
    _verify_checksums(p, manifest)

    # Load constants BEFORE the FX deserializer so we can seed them into
    # the GM construction.  T4's deserializer expects ``constants`` as a
    # FQN -> tensor dict and does the seed-before-construct + reflip
    # dance to keep them as ``_AttrKind.CONSTANT`` (not silent buffers).
    constants_path = p / CONSTANTS_FILENAME
    constants: Dict[str, torch.Tensor] = (
        _load_safetensors(constants_path) if constants_path.is_file() else {}
    )

    graph_path = p / GRAPH_FILENAME
    if not graph_path.is_file():
        raise RuntimeError(f"Bundle is missing {GRAPH_FILENAME!r} at {graph_path}")
    gm = deserialize_fx_graph(graph_path, constants=constants if constants else None)

    bindings = _load_binding_table(p)

    model_state = _load_safetensors(p / MODEL_STATE_FILENAME)
    params, buffers = _assemble_params_and_buffers(bindings, model_state)

    return LoadedBundle(
        gm=gm,
        bindings=bindings,
        params=params,
        buffers=buffers,
        constants=constants,
        manifest=manifest,
    )
