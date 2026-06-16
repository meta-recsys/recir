# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Bundle producer: trace a raw ``nn.Module`` and serialize the forward-only graph + state.

Public entry point: :func:`export_bundle`. Takes a raw :class:`nn.Module`
and sample inputs, traces through :func:`recir.api._local_compile` +
``make_fx``, cleans the resulting FX graph, and writes the bundle
artifacts to disk.

The producer is **forward-only**: optimizer state lives on the consumer
side, which wraps the loaded graph into a full train step. The consumer
owns the optimizer class and hyperparameters via its own launch config;
the bundle carries no optimizer hints.

Bundle layout -- **5 artifacts**:

    <bundle_dir>/
      manifest.json           — bundle metadata + artifact index + target_device
      train_step.fx.json      — serialized FX GraphModule (recir.serde format)
      train.binding.json      — BindingTable (constants + params + buffers +
                                explicit user_input entries)
      model_state.safetensors      — params + buffers, FQN-keyed
      constants.safetensors        — get_attr-backed tensor constants

The bundle is **topology-free**: no parallelism config, world_size, or rank
are written.  Only the producer's declared ``target_device`` class travels
in the manifest, and only so the export-time gates have something to check
against.

The producer reads tensor constants directly from the GraphModule's
``get_attr``-backed attributes via :func:`torch.fx.graph_module._get_attr`
and the ``required_constants`` field emitted by :func:`recir.serde.serialize_fx_graph`.
It does NOT walk a runtime call vector or rely on placeholder ordering to
discover constants -- those tensors live as CONSTANT-kind attributes on
the GM, never as positional inputs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.fx as fx
import torch.nn as nn
from recir.api import _local_compile
from recir.serde import BindingTable, serialize_fx_graph
from recir.serde.schema import Binding
from safetensors.torch import save_file
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.graph_module import _get_attr

from ._gates import (
    _validate_kwargs_meta_consistency,
    _validate_no_cpu_scalar_in_cuda_bundle,
)
from .manifest import BUNDLE_FORMAT_VERSION, BundleManifest, write_manifest


logger: logging.Logger = logging.getLogger(__name__)


# Artifact filenames inside the bundle directory.  Centralised here so the
# consumer side can import them rather than hard-coding strings.
MANIFEST_FILENAME: str = "manifest.json"
GRAPH_FILENAME: str = "train_step.fx.json"
BINDING_FILENAME: str = "train.binding.json"
MODEL_STATE_FILENAME: str = "model_state.safetensors"
CONSTANTS_FILENAME: str = "constants.safetensors"


def _sha256_of_file(path: Path) -> str:
    """Compute the sha256 hex digest of a file on disk.

    Reads in 1 MiB chunks so we don't OOM on large safetensors blobs.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_capture_recir_sha() -> str:
    """Best-effort capture of the producer's source SHA.

    Tries ``hg log -r . -T '{node}'`` first (matching the in-tree source
    control), then falls back to ``git rev-parse HEAD``.  Returns
    ``"unknown"`` on any failure -- the manifest is still useful without a
    SHA, just less forensically traceable.
    """
    here = Path(__file__).resolve().parent
    for cmd in (
        ["hg", "log", "-r", ".", "-T", "{node}"],
        ["git", "rev-parse", "HEAD"],
    ):
        try:
            result = subprocess.run(
                cmd,
                cwd=str(here),
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            continue
    return "unknown"


def _collect_constants_from_gm(
    gm: fx.GraphModule, required_constants: list[str]
) -> dict[str, torch.Tensor]:
    """Read every required constant FQN off the GraphModule's attributes.

    The producer treats the GM as the single source of truth for constant
    values: ``required_constants`` is the producer-emitted list of
    ``get_attr`` FQNs (collected by :func:`recir.serde.serialize_fx_graph`),
    and each entry is fetched via :func:`torch.fx.graph_module._get_attr`
    so nested FQNs (e.g. ``inner.weight_const``) are walked correctly.

    Constants are NOT discovered from any runtime call vector or
    ``BindingTable``: they live as CONSTANT-kind attributes on the GM,
    never as positional placeholders.

    Args:
        gm: The :class:`fx.GraphModule` to read constants from.
        required_constants: List of FQNs the serializer collected while
            walking ``get_attr`` nodes.

    Returns:
        Dict mapping FQN to the tensor value pulled from the GM.

    Raises:
        AttributeError: If a required FQN is not resolvable on the GM.
    """
    constants: dict[str, torch.Tensor] = {}
    for fqn in required_constants:
        value = _get_attr(gm, fqn)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Required constant {fqn!r} on GraphModule is not a Tensor "
                f"(got {type(value).__name__}); the bundle producer cannot "
                f"serialise non-tensor constants."
            )
        constants[fqn] = value
    return constants


def _build_binding_table(
    constants: dict[str, torch.Tensor],
    params: dict[str, nn.Parameter],
    buffers: dict[str, torch.Tensor],
    user_input_count: int,
) -> BindingTable:
    """Construct the ordered :class:`BindingTable` for the bundle.

    The placeholder ordering is::

        [constants...] [params...] [buffers...] [user_inputs...]

    where:

    * ``constants`` are metadata-only (they live as GM attributes, not
      placeholders) but are recorded so the consumer can compute the
      manifest of state-bearing FQNs without walking the FX graph;
    * params and buffers appear in their dict insertion order;
    * each user input gets its own ``Binding(kind="user_input",
      fqn="<input_N>")`` so the consumer can rebuild the call signature
      without inspecting placeholder names.

    The explicit ``user_input`` bindings are the producer's contract with
    the consumer: positional args at runtime are passed in the order
    ``[input_0, input_1, ...]`` recorded here.

    Args:
        constants: FQN -> tensor map for ``get_attr``-backed constants.
        params: FQN -> parameter map (from ``mod.named_parameters()``).
        buffers: FQN -> buffer map (from ``mod.named_buffers()``).
        user_input_count: Number of user-input placeholders the runtime
            will receive.  Each gets a synthesised ``<input_N>`` FQN.

    Returns:
        A populated :class:`BindingTable` whose ``bindings`` is in the
        documented order.
    """
    bindings: list[Binding] = []
    for fqn in constants:
        bindings.append(Binding(kind="constant", fqn=fqn))
    for fqn in params:
        bindings.append(Binding(kind="param", fqn=fqn))
    for fqn in buffers:
        bindings.append(Binding(kind="buffer", fqn=fqn))
    for i in range(user_input_count):
        bindings.append(Binding(kind="user_input", fqn=f"<input_{i}>"))
    return BindingTable(bindings=bindings)


def _binding_table_to_jsonable(bt: BindingTable) -> dict[str, Any]:
    """Serialize a :class:`BindingTable` to a JSON-friendly dict.

    Mirrors the shape used by ``recir.serde``: a single top-level dict
    whose only key is ``"bindings"``, holding a list of per-binding dicts.
    Kept self-contained so consumers can parse the binding sidecar without
    going through :func:`recir.serde.deserialize_fx_graph`.
    """
    return {
        "bindings": [dataclasses.asdict(b) for b in bt.bindings],
    }


def _save_safetensors(
    tensors: dict[str, torch.Tensor],
    path: Path,
) -> None:
    """Write a flat FQN -> tensor dict to a safetensors file.

    All tensors are detached and moved to CPU before writing -- safetensors
    stores raw byte buffers and cannot serialise CUDA pointers.  An empty
    ``tensors`` dict is allowed (the file is still written, with zero
    entries) so the bundle layout is uniform across models with and
    without constants.
    """
    cpu_tensors: dict[str, torch.Tensor] = {
        fqn: t.detach().cpu().contiguous() for fqn, t in tensors.items()
    }
    save_file(cpu_tensors, str(path))


@dataclass
class CompiledBundleInputs:
    """Inputs to :meth:`BundleExporter.export`.

    A small structural container so the export entry point can be called
    with either a ``recir.api._LocalCompiledResult`` (which has the same
    shape minus ``user_input_count``) or a hand-assembled bundle in tests.

    The producer is forward-only: there is no ``optimizer`` or
    ``named_states`` field. Optimizer reconstruction is the consumer's
    responsibility, driven by advisory metadata the caller stashes in
    ``manifest_extras``.

    Attributes:
        gm: The traced :class:`fx.GraphModule`.  Tensor constants must be
            registered on it as :data:`torch.export.unflatten._AttrKind.CONSTANT`
            attributes so :func:`torch.fx.graph_module._get_attr` can fetch
            them -- this is the layout :func:`recir.api._local_compile`
            already produces after the retag dance.
        params: FQN -> :class:`nn.Parameter` map (from ``mod.named_parameters()``).
        buffers: FQN -> tensor map (from ``mod.named_buffers()``).
        user_input_count: Number of positional user inputs the GM expects
            at call time.  Each one gets a synthesised ``<input_N>``
            ``user_input`` binding in the emitted :class:`BindingTable`.
    """

    gm: fx.GraphModule
    params: dict[str, nn.Parameter]
    buffers: dict[str, torch.Tensor]
    user_input_count: int


@dataclass
class BundleExporter:
    """Object form of the bundle producer.

    Most callers should use :func:`export_bundle` instead; this class
    exists so advanced callers can override individual steps (e.g. inject
    a custom recir-SHA capture).

    Attributes:
        recir_sha: Override for the producer recir SHA in the manifest.
            ``None`` (default) means auto-detect via
            :func:`_try_capture_recir_sha`.
    """

    recir_sha: str | None = None

    def export(
        self,
        compiled: CompiledBundleInputs,
        output_dir: str | Path,
        target_device: str | torch.device,
        *,
        manifest_extras: dict[str, Any] | None = None,
    ) -> BundleManifest:
        """Run the export.  See :func:`export_bundle` for argument docs."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        target_device_str = (
            str(target_device)
            if not isinstance(target_device, torch.device)
            else (
                f"{target_device.type}:{target_device.index}"
                if target_device.index is not None
                else target_device.type
            )
        )

        # 1. Run export-time validation gates BEFORE writing any artifact
        #    so a failure leaves the output dir clean.
        _validate_kwargs_meta_consistency(compiled.gm)
        _validate_no_cpu_scalar_in_cuda_bundle(compiled.gm, target_device)

        # 2. Serialize FX graph + collect constants off the GM.
        graph_payload, binding_table, constants = self._serialize_graph_and_bindings(
            compiled
        )
        graph_path = out / GRAPH_FILENAME
        graph_path.write_text(json.dumps(graph_payload))

        # 3. Write the binding sidecar separately so a consumer can parse
        #    placeholder ordering without loading the full FX payload.
        binding_path = out / BINDING_FILENAME
        binding_path.write_text(
            json.dumps(_binding_table_to_jsonable(binding_table), indent=2)
        )

        # 4. Write tensor artifacts (model_state + constants).
        model_state_path, constants_path = self._write_state_artifacts(
            compiled, constants, out
        )

        # 5. manifest.json -- written LAST, after every sibling artifact
        #    exists on disk, so the artifact_index sha256 hashes are
        #    computed over the actual bytes the consumer will read.
        return self._write_manifest(
            out,
            target_device_str,
            graph_path,
            binding_path,
            model_state_path,
            constants_path,
            manifest_extras,
        )

    def _serialize_graph_and_bindings(
        self, compiled: CompiledBundleInputs
    ) -> tuple[dict[str, Any], BindingTable, dict[str, torch.Tensor]]:
        """Serialize the FX graph (with bindings embedded) and return constants."""
        # The serializer collects every distinct ``get_attr`` FQN into
        # ``required_constants``; the producer uses that exact list (NOT
        # the BindingTable, NOT a runtime call vector) to fetch tensor
        # values off the GM.
        graph_payload = serialize_fx_graph(compiled.gm)
        required = graph_payload.get("required_constants", [])
        assert isinstance(required, list)
        constants = _collect_constants_from_gm(
            compiled.gm, [str(fqn) for fqn in required]
        )

        # Build the BindingTable once we know the constant FQNs.  Re-serialise
        # the graph WITH bindings embedded so the FX payload and the binding
        # sidecar agree.
        binding_table = _build_binding_table(
            constants,
            compiled.params,
            compiled.buffers,
            compiled.user_input_count,
        )
        graph_payload = serialize_fx_graph(compiled.gm, bindings=binding_table)
        return graph_payload, binding_table, constants

    def _write_state_artifacts(
        self,
        compiled: CompiledBundleInputs,
        constants: dict[str, torch.Tensor],
        out: Path,
    ) -> tuple[Path, Path]:
        """Write model_state.safetensors + constants.safetensors."""
        # model_state.safetensors -- params + buffers in one file. Both are
        # FQN-keyed and have identical semantics from the runtime's
        # perspective. Reject collisions explicitly (a buffer FQN equal to
        # a param FQN would silently overwrite the param's tensor).
        collision = set(compiled.params) & set(compiled.buffers)
        if collision:
            raise ValueError(f"Param/buffer FQN collision: {sorted(collision)}")
        model_state: dict[str, torch.Tensor] = {}
        model_state.update(compiled.params)
        model_state.update(compiled.buffers)
        model_state_path = out / MODEL_STATE_FILENAME
        _save_safetensors(model_state, model_state_path)

        # constants.safetensors -- tensor values pulled off the GM. Kept
        # SEPARATE from model_state.safetensors because constants have
        # distinct semantics (read-only at runtime, never updated) and a
        # distinct key namespace (``_tensor_constantN`` vs FQNs).
        constants_path = out / CONSTANTS_FILENAME
        _save_safetensors(constants, constants_path)
        return model_state_path, constants_path

    def _write_manifest(
        self,
        out: Path,
        target_device_str: str,
        graph_path: Path,
        binding_path: Path,
        model_state_path: Path,
        constants_path: Path,
        manifest_extras: dict[str, Any] | None,
    ) -> BundleManifest:
        """Build the artifact_index from on-disk bytes and write manifest.json."""
        artifact_index: dict[str, str] = {
            GRAPH_FILENAME: _sha256_of_file(graph_path),
            BINDING_FILENAME: _sha256_of_file(binding_path),
            MODEL_STATE_FILENAME: _sha256_of_file(model_state_path),
            CONSTANTS_FILENAME: _sha256_of_file(constants_path),
        }
        manifest = BundleManifest(
            bundle_format_version=BUNDLE_FORMAT_VERSION,
            producer_recir_sha=(
                self.recir_sha
                if self.recir_sha is not None
                else _try_capture_recir_sha()
            ),
            producer_torch_version=torch.__version__,
            producer_python_version=sys.version,
            target_device=target_device_str,
            artifact_index=artifact_index,
            extras=dict(manifest_extras) if manifest_extras else {},
        )
        manifest_path = out / MANIFEST_FILENAME
        write_manifest(manifest, manifest_path)
        return manifest


def export_bundle(
    model: nn.Module,
    sample_inputs: tuple[Any, ...],
    output_dir: str | Path,
    target_device: str | torch.device,
    *,
    manifest_extras: dict[str, Any] | None = None,
) -> BundleManifest:
    """Trace ``model(*sample_inputs)`` and write a forward-only bundle to disk.

    Wraps a ``forward_only(model, *user_inputs)`` closure around the
    module, traces it through :func:`recir.api._local_compile` (which
    drives :func:`torch.fx.experimental.proxy_tensor.make_fx` under
    :class:`FakeTensorMode`), prunes dead non-tensor placeholders, drops
    ``meta['val']`` from mixed-type tuple/list nodes (e.g. SDPA flash-attn
    decompositions) so the recir FX serializer accepts the graph, then
    writes the 5 artifacts listed in this module's docstring to
    ``output_dir``. The directory is created if it does not exist;
    existing files in it are overwritten.

    No optimizer is constructed at trace time, and no optimizer hints
    are written to the manifest; the consumer owns optimizer construction
    via its own launch config.

    The bundle is **topology-free**: only the producer's declared
    ``target_device`` class travels in the manifest. Topology
    (parallelism, world_size, rank) is a runtime concern.

    Args:
        model: The :class:`nn.Module` to trace. Must be on the device its
            sample inputs live on (no device migration happens here).
        sample_inputs: Positional sample inputs to pass into ``model``.
            Tensors are converted to :class:`FakeTensor` via the internal
            :class:`FakeTensorMode`; non-tensor inputs are passed through.
            ``user_input_count`` is set to ``len(sample_inputs)``.
        output_dir: Destination directory. Created if missing.
        target_device: Device class the bundle is validated against
            (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``). String or
            :class:`torch.device`. Used by the export-time gates to reject
            graphs whose kwargs disagree with the declared class.
        manifest_extras: Optional dict of caller-supplied JSON-friendly
            metadata to embed in the manifest's ``extras`` field. Merged
            FIRST so the producer-populated key (``bundle_kind``) wins on
            collision.

    Returns:
        The :class:`BundleManifest` that was written, for caller inspection.

    Raises:
        AssertionError: If the trace produces an optimizer or named
            states (forward-only bundles must not).
        SchemaViolationError: If an export-time gate rejects the graph
            (e.g. CPU scalar_tensor in a CUDA bundle, device kwarg / meta
            drift).
        ValueError: If the traced model's params and buffers share an FQN.
    """
    fake_mode = FakeTensorMode()

    def forward_only(mod: nn.Module, *user_inputs: Any) -> torch.Tensor:
        return mod(*user_inputs)

    logger.info(
        "Tracing forward-only graph for %s with %d sample inputs",
        type(model).__name__,
        len(sample_inputs),
    )
    result = _local_compile(
        forward_only,
        model,
        *sample_inputs,
        fake_mode=fake_mode,
    )
    assert not result.opts and not result.named_states, (
        "forward-only trace must produce no optimizer state"
    )

    # ``_local_compile`` passes the wrapper through *args, so make_fx emits
    # a placeholder for it whose meta['val'] is the nn.Module (not a tensor)
    # and which ``forward_only`` never references through tensor ops. Erase
    # any such dead non-tensor placeholders so the call vector matches
    # ``user_input_count``.
    pruned = 0
    for node in list(result.gm.graph.nodes):
        if (
            node.op == "placeholder"
            and len(node.users) == 0
            and not isinstance(node.meta.get("val"), torch.Tensor)
        ):
            result.gm.graph.erase_node(node)
            pruned += 1
    if pruned:
        result.gm.recompile()

    # Drop ``meta['val']`` from any node whose val is a tuple/list with
    # mixed FakeTensor + None / int / etc. types -- the recir FX serializer
    # rejects those (it only encodes pure-FakeTensor tuples). This affects
    # multi-output decompositions like
    # ``aten._scaled_dot_product_flash_attention.default`` whose tuple
    # includes None slots for unused varlen cum-seq tensors and int slots
    # for max-seq-len. The downstream consumer only references slot 0 via
    # ``getitem(0)`` and never reads ``meta['val']`` of these nodes, so
    # dropping the val keeps the graph callable while satisfying the
    # schema.
    val_dropped = 0
    for node in result.gm.graph.nodes:
        val = node.meta.get("val")
        if isinstance(val, (tuple, list)) and not all(
            isinstance(v, torch.Tensor) for v in val
        ):
            del node.meta["val"]
            val_dropped += 1

    logger.info(
        "Trace OK: %d params, %d buffers, %d constants, pruned %d unused "
        "placeholders, dropped %d mixed-type tuple vals",
        len(result.params),
        len(result.buffers),
        len(result.constants),
        pruned,
        val_dropped,
    )

    compiled = CompiledBundleInputs(
        gm=result.gm,
        params=result.params,
        buffers=result.buffers,
        user_input_count=len(sample_inputs),
    )

    # Caller-supplied extras are merged FIRST so the producer-populated
    # key (``bundle_kind``) wins on collision.
    extras: dict[str, Any] = dict(manifest_extras) if manifest_extras else {}
    extras["bundle_kind"] = "forward_only"

    return BundleExporter().export(
        compiled,
        output_dir,
        target_device,
        manifest_extras=extras,
    )
