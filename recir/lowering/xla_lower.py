# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import contextvars
import ctypes
import functools
import gc
import logging
import os
import weakref
from typing import Any, Callable, Generator, List, Optional, Tuple

import jax  # @manual
import jax.numpy as jnp  # @manual
import ml_dtypes
import numpy as np
import recir.torchax as torchax
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from recir.backends.types import BackendType
from recir.compiler.batch_dim import (
    propagate_batch_input_meta,
    propagate_fsdp_sharded_meta,
)
from recir.distributed.sharding_spec import get_state_layout
from recir.lowering.lowering_utils import (
    has_meta_kernel,
    segment_and_lower,
    strip_profiler_nodes,
)
from recir.lowering.op_swap import OpSwapRegistry, swap_ops
from recir.torchax.gspmd_context import gspmd_mesh_context
from recir.torchax.ops.mappings import t2j
from recir.utils.meta_tensor_init import (
    default_embedding_init_bound,
    get_meta_init_spec,
    get_meta_init_stream_id,
)

try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

logger: logging.Logger = logging.getLogger(__name__)


def _host_memory_status() -> dict[str, int]:
    status: dict[str, int] = {}
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key in {"VmRSS", "VmHWM", "VmSize", "RssAnon", "RssFile"}:
                    fields = value.strip().split()
                    if fields and fields[0].isdigit():
                        status[key] = int(fields[0])
    except OSError as error:
        status["error"] = -1
        logger.warning("Failed to read /proc/self/status: %s", error)
    return status


def _trim_host_memory_after_compile() -> None:
    if os.environ.get("RECIR_XLA_TRIM_HOST_MEMORY_AFTER_COMPILE") != "1":
        return
    before = _host_memory_status()
    collected = gc.collect()
    malloc_trim_result: int | None = None
    malloc_trim_error: str | None = None
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim_result = int(malloc_trim(0))
    except (AttributeError, OSError) as error:
        malloc_trim_error = f"{type(error).__name__}: {error}"
    logger.info(
        "Trimmed host memory after XLA compile: gc_collected=%d "
        "malloc_trim_result=%s malloc_trim_error=%s before=%s after=%s",
        collected,
        malloc_trim_result,
        malloc_trim_error,
        before,
        _host_memory_status(),
    )


# Cross-host AOT compile support. When set, the device-discovery helpers
# below report the virtual topology's devices instead of the host's local
# JAX devices, so recir's mesh construction (and the per-input sharded
# JAX builder) targets the AOT topology rather than the CPU backend that
# is actually running the build job. The contextvar default of ``None``
# keeps every existing caller on the live ``jax.devices()`` /
# ``jax.local_devices()`` path — the indirection is a true no-op for
# non-AOT users.
_AOT_TOPOLOGY: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "aot_topology", default=None
)


# Cache reshard executables by target sharding. ``_to_sharded_jax`` reshards a
# prior-step ``device_state`` output that spans non-addressable devices to the
# input's target sharding via a jitted identity; a fresh ``lambda`` per call
# would force a NEW jit trace+compile for every FSDP weight every step (~180
# recompiles/step on a 60-layer Gemma -> ~37s/step of pure compile). The target
# ``NamedSharding`` is hashable and there are only a handful of distinct ones
# (fsdp-1d, fsdp-2d, batch, replicated), so cache + reuse across steps/shapes.
_RESHARD_JIT_CACHE: dict[Any, Any] = {}


def _cached_reshard(target: Any) -> Any:
    fn = _RESHARD_JIT_CACHE.get(target)
    if fn is None:
        fn = jax.jit(lambda x: x, out_shardings=target)
        _RESHARD_JIT_CACHE[target] = fn
    return fn


# Cache entry: weakref, _version, data_ptr, conversion layout, device copy.
_StateInputCacheEntry = Tuple[Any, int, int, Any, Any]
_StateInputCache = dict[Any, _StateInputCacheEntry]


def _state_input_cache_entry_matches(
    entry: _StateInputCacheEntry,
    tensor: torch.Tensor,
    cache_layout: Any,
) -> bool:
    return (
        entry[0]() is tensor
        and entry[1] == tensor._version
        and entry[2] == tensor.data_ptr()
        and entry[3] == cache_layout
        and not entry[4].is_deleted()
    )


def _state_input_cache_slot_namespace(key: Any) -> Any:
    """Return a lowered executable's identity token from a cache key."""
    return key[0] if isinstance(key, tuple) and len(key) == 2 else None


def _state_input_cache_slots_can_share(
    cache: _StateInputCache,
    destination: Any,
    source: Any,
    device_copy: Any,
) -> bool:
    """Whether sharing would avoid aliasing two args of one executable."""
    destination_namespace = _state_input_cache_slot_namespace(destination)
    source_namespace = _state_input_cache_slot_namespace(source)
    if destination_namespace is None or destination_namespace is source_namespace:
        return False
    return not any(
        _state_input_cache_slot_namespace(key) is destination_namespace
        and entry[4] is device_copy
        for key, entry in cache.items()
    )


def _find_shared_state_input(
    cache: _StateInputCache,
    destination: Any,
    tensor: torch.Tensor,
    cache_layout: Any,
) -> Optional[_StateInputCacheEntry]:
    """Find a reusable copy in another module and purge stale aliases."""
    shared_entry = None
    for source, candidate in list(cache.items()):
        cached_tensor = candidate[0]()
        if cached_tensor is None or candidate[4].is_deleted():
            cache.pop(source, None)
        elif cached_tensor is tensor:
            if candidate[1] != tensor._version or candidate[2] != tensor.data_ptr():
                cache.pop(source, None)
            elif candidate[3] == cache_layout and _state_input_cache_slots_can_share(
                cache, destination, source, candidate[4]
            ):
                shared_entry = candidate
    return shared_entry


def _reuse_unchanged_state_input(
    cache: _StateInputCache,
    index: Any,
    tensor: Any,
    is_batch_input: bool,
    is_frozen_parameter: bool,
    convert: Callable[[], Any],
    cache_layout: Any = None,
) -> Any:
    """Return the cached device copy of an unchanged frozen parameter.

    Frozen parameters are never outputs of the train step, so the device-state
    fast path keeps handing their host tensors to every call and each call
    re-uploads them (multi-GB per device for split-vocab embeddings). Cache only
    parameters identified as frozen while tracing: mutable buffers and trainable
    state may be changed through storage aliases that do not update ``_version``.
    The identity, version, and storage checks provide additional invalidation for
    ordinary PyTorch mutations. Donated inputs are deleted by the call and are
    therefore always re-converted.
    """
    if (
        not is_frozen_parameter
        or is_batch_input
        or not isinstance(tensor, torch.Tensor)
        or tensor.is_meta
        or tensor.requires_grad
        or tensor.is_inference()
        or _AOT_TOPOLOGY.get() is not None
    ):
        # A slot commonly transitions from its first-call host tensor to a JAX
        # device-state output. Do not keep the superseded device buffer alive.
        cache.pop(index, None)
        return convert()
    # Remove the entry before a miss conversion so a multi-GB stale buffer can
    # be released before its replacement is allocated.
    entry = cache.pop(index, None)
    if entry is not None and _state_input_cache_entry_matches(
        entry, tensor, cache_layout
    ):
        cache[index] = entry
        return entry[4]
    del entry

    # Multiple lowered modules can consume one tied/shared parameter. Reuse a
    # matching device copy instead of retaining one multi-GB copy per module.
    # Purge dead or stale aliases while scanning so they cannot pin old buffers.
    shared_entry = _find_shared_state_input(cache, index, tensor, cache_layout)
    if shared_entry is not None:
        cache[index] = shared_entry
        return shared_entry[4]

    device_copy = convert()
    if isinstance(device_copy, jax.Array):
        cache[index] = (
            weakref.ref(tensor),
            tensor._version,
            tensor.data_ptr(),
            cache_layout,
            device_copy,
        )
    return device_copy


@contextlib.contextmanager
def aot_topology(topology_desc: Any) -> Generator[None, None, None]:
    """Override device discovery to a virtual topology for AOT prebuild.

    Use on a CPU host when AOT-prebuilding for an accelerator pod.
    Inside the context, recir's mesh construction sees
    ``topology_desc.devices`` instead of the local CPU devices, so the
    produced binary targets the virtual TPU topology and can be
    serialized via :func:`recir.lowering.aot.aot_prebuild_and_serialize`
    for execution on the actual pod (the Pax / MaxText
    "build-once-on-CPU, load-everywhere-on-TPU" pattern).

    Args:
        topology_desc: A ``jax.experimental.topologies.TopologyDescription``
            (or any object exposing a ``.devices`` attribute that returns
            a sequence of JAX device handles).
    """
    token = _AOT_TOPOLOGY.set(topology_desc)
    try:
        yield
    finally:
        _AOT_TOPOLOGY.reset(token)


def _get_global_devices() -> List[Any]:
    """Global JAX device list, honoring an active :func:`aot_topology`."""
    topo = _AOT_TOPOLOGY.get()
    return list(topo.devices) if topo is not None else jax.devices()


def _get_local_devices() -> List[Any]:
    """Local JAX device list, honoring an active :func:`aot_topology`.

    AOT mode: returns the per-process slice of ``topo.devices`` (matching
    the runtime semantics of ``jax.local_devices()`` on each pod).
    Without this, ``num_local == num_global`` collapses the host-shard
    math in ``_to_sharded_jax`` to a no-op and the build sees per-host
    (not per-mesh) batch shapes — the resulting executable then fails
    sharding-divisibility checks at load time (e.g. axis 0 sized 8
    cannot be partitioned 256 ways on a 32-host v7-128 pod).

    Outside AOT mode it falls back to ``jax.local_devices()``, matching
    the original behavior.
    """
    topo = _AOT_TOPOLOGY.get()
    if topo is None:
        return jax.local_devices()
    devs = list(topo.devices)
    p0 = [d for d in devs if getattr(d, "process_index", 0) == 0]
    return p0 if p0 else devs


_GSPMD_DATA_AXIS = "gspmd_data"
_GSPMD_FSDP_AXIS = "torch_dist"


def _gspmd_mesh(
    devices: List[Any],
    fsdp_axis_size: int,
) -> tuple[Mesh, Any, bool]:
    if fsdp_axis_size < 1 or len(devices) % fsdp_axis_size != 0:
        raise ValueError(
            "GSPMD device count must be divisible by the FSDP axis; "
            f"devices={len(devices)}, fsdp_axis_size={fsdp_axis_size}"
        )
    if fsdp_axis_size == len(devices):
        return Mesh(devices, axis_names=(_GSPMD_FSDP_AXIS,)), _GSPMD_FSDP_AXIS, False
    local_device_count = len(_get_local_devices())
    if fsdp_axis_size % local_device_count != 0:
        raise ValueError(
            "The hybrid GSPMD FSDP axis must contain a whole number of hosts; "
            f"fsdp_axis_size={fsdp_axis_size}, local_devices={local_device_count}"
        )
    data_axis_size = len(devices) // fsdp_axis_size
    device_grid = np.asarray(devices, dtype=object).reshape(
        data_axis_size,
        fsdp_axis_size,
    )
    mesh = Mesh(device_grid, axis_names=(_GSPMD_DATA_AXIS, _GSPMD_FSDP_AXIS))
    return mesh, (_GSPMD_DATA_AXIS, _GSPMD_FSDP_AXIS), True


# torch -> jax.numpy dtype map for on-device (meta) tensor materialization,
# where there is no host tensor to infer the dtype from via numpy.


_TORCH_TO_JNP: dict[torch.dtype, Any] = {
    torch.float32: jnp.float32,
    torch.float64: jnp.float64,
    torch.float16: jnp.float16,
    torch.bfloat16: jnp.bfloat16,
    torch.int64: jnp.int64,
    torch.int32: jnp.int32,
    torch.int16: jnp.int16,
    torch.int8: jnp.int8,
    torch.uint8: jnp.uint8,
    torch.bool: jnp.bool_,
}


def _materialize_meta_fsdp_tensor(tensor: Any, mesh: Any) -> Any:
    """On-device materialization for a *meta* FSDP tensor.

    Meta FSDP tensors have no host storage, so the normal torch->numpy path
    cannot run. Table weights are generated directly in HBM with random uniform
    initialization. Optimizer state tensors are explicitly marked for zero init
    during state sharding and must not be randomized.
    """
    local_device_count = len(_get_local_devices())
    fsdp_axis_size = int(mesh.shape[_GSPMD_FSDP_AXIS])
    fsdp_host_shard_count = fsdp_axis_size // local_device_count
    trailing = tuple(int(s) for s in tensor.shape[1:])
    global_rows = int(tensor.shape[0]) * fsdp_host_shard_count
    global_shape = (global_rows, *trailing)
    jdtype = _TORCH_TO_JNP.get(tensor.dtype, jnp.float32)
    wspec = P(_GSPMD_FSDP_AXIS, *([None] * (tensor.ndim - 1)))
    sharding = NamedSharding(mesh, wspec)
    if _AOT_TOPOLOGY.get() is not None:
        return jax.ShapeDtypeStruct(global_shape, jdtype, sharding=sharding)
    rows_per_device = global_shape[0] // fsdp_axis_size
    init_kind, bound = get_meta_init_spec(
        tensor, default_embedding_init_bound(global_shape[0])
    )
    stream_id = get_meta_init_stream_id(tensor)

    # Bind statics as default args (not closure captures) to satisfy
    # pyre [53]; shard_map invokes _gen_shard with only the sharded arg.
    def _gen_shard(
        _idx: Any,
        _rpd: int = rows_per_device,
        _tr: Any = trailing,
        _kind: str = init_kind,
        _b: float = bound,
        _dt: Any = jdtype,
        _stream_id: int = stream_id,
    ) -> Any:
        if _kind == "zeros":
            return jnp.zeros((_rpd, *_tr), dtype=_dt)
        gi = jax.lax.axis_index(_GSPMD_FSDP_AXIS)
        key = jax.random.fold_in(jax.random.PRNGKey(0), gi)
        key = jax.random.fold_in(key, _stream_id)
        shard = jax.random.uniform(key, (_rpd, *_tr), dtype=_dt, minval=-_b, maxval=_b)
        return shard.astype(_dt)

    return shard_map(  # pyre-fixme[28]
        _gen_shard,
        mesh=mesh,
        in_specs=P(_GSPMD_FSDP_AXIS),
        out_specs=wspec,
        # pyrefly: ignore [unexpected-keyword]
        check_vma=False,
    )(jnp.arange(fsdp_axis_size, dtype=jnp.int32))


class _XLAModule(torch.nn.Module):
    """Wraps an fx.GraphModule subgraph for lazy XLA compilation.

    On first forward call, the subgraph is compiled to an XLA computation
    via the selected backend. Subsequent calls reuse the cached executable.

    Supported XLA backends:
      - XLA_DYNAMO_BRIDGE: uses torch_xla.core.dynamo_bridge.extract_compiled_graph
        to trace through the lazy-tensor path and produce a cached XLA HLO.
      - XLA_SHARD_MAP: uses the JAX-based torchax environment to jit-compile
        the graph through JAX/XLA, with shard_map for collective ops.
      - XLA_GSPMD: uses GSPMD auto-parallelism via jax.jit with
        input/output sharding annotations.
    """

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        backend_type: BackendType = BackendType.XLA_SHARD_MAP,
        capture_hlo_text: bool = False,
    ) -> None:
        super().__init__()
        self.gm = gm
        self.backend_type = backend_type
        dump_graphs_mode = os.environ.get("RECIR_DUMP_GRAPHS", "")
        self.capture_hlo_text = capture_hlo_text or dump_graphs_mode not in (
            "",
            "0",
            "false",
            "False",
        )
        self.compiled: Optional[Callable[..., List[torch.Tensor]]] = None
        self.xla_hlo_text: Optional[str] = None
        self.keep_jax_output: bool = False
        self.torch_output_indices: Optional[set[int]] = None
        self.final_output_indices: set[int] = set()
        self.safe_donate_indices: Optional[set[int]] = None
        # AOT (ahead-of-time) build state. Populated by ``_build_xla_executable``
        # whenever a backend produces a ``jax.stages.Compiled``, so an external
        # caller (see ``recir.lowering.aot``) can serialize and ship the
        # executable instead of rebuilding on every host. Also set directly by
        # ``aot_load_prebuilt_into`` on the load side; the run closures
        # returned by ``_build_*_executable`` prefer this attribute over a
        # fresh JAX cache lookup.
        self._jax_compiled: Optional[Any] = None
        # Guard so RECIR_DUMP_GRAPHS dumps each subgraph's FX + XLA graph once.
        self._dumped_graphs: bool = False
        self._logged_memory_analysis: bool = False
        self._state_input_cache_namespace: object = object()
        self._state_input_cache: _StateInputCache = {}

    def _bind_state_input_cache(self, cache: _StateInputCache) -> None:
        """Use the device-state manager's cache for this lowered module."""
        if self._state_input_cache is cache:
            return
        self._state_input_cache.clear()
        self._state_input_cache = cache

    def _log_compiled_memory_analysis(self) -> None:
        if self._logged_memory_analysis or self._jax_compiled is None:
            return
        memory_analysis = getattr(self._jax_compiled, "memory_analysis", None)
        if not callable(memory_analysis):
            return
        try:
            stats = memory_analysis()
        except Exception:  # noqa: BLE001 -- diagnostics must never break training
            logger.warning("XLA compiled memory analysis failed", exc_info=True)
            return
        self._logged_memory_analysis = True
        gib = float(1024**3)
        logger.info(
            "XLA compiled memory: arguments=%.3f GiB, outputs=%.3f GiB, "
            "aliases=%.3f GiB, temporaries=%.3f GiB, code=%.3f GiB",
            float(getattr(stats, "argument_size_in_bytes", 0) or 0) / gib,
            float(getattr(stats, "output_size_in_bytes", 0) or 0) / gib,
            float(getattr(stats, "alias_size_in_bytes", 0) or 0) / gib,
            float(getattr(stats, "temp_size_in_bytes", 0) or 0) / gib,
            float(getattr(stats, "generated_code_size_in_bytes", 0) or 0) / gib,
        )

    def _maybe_dump_graphs(self, tag: str) -> None:  # noqa: C901
        """Diagnostic: when RECIR_DUMP_GRAPHS is set, log this subgraph's FX
        graph and its compiled XLA HLO so you can verify lowering (e.g. that
        aten.embedding_dense_backward survives and produces a small [vocab,D]
        all-reduce rather than a gathered [N_global,D] one) directly from the
        run log -- no trace download needed.

        RECIR_DUMP_GRAPHS=1     -> embedding/index/scatter/collective lines only.
        RECIR_DUMP_GRAPHS=full  -> the entire FX graph + full HLO.
        """
        mode = os.environ.get("RECIR_DUMP_GRAPHS", "")
        if mode in ("", "0", "false", "False") or self._dumped_graphs:
            return
        self._dumped_graphs = True
        full = mode.lower() == "full"

        # 1. FX graph (the aten ops JaxInterpreter lowers).
        targets = [
            str(n.target) for n in self.gm.graph.nodes if n.op == "call_function"
        ]
        keys = (
            "embedding",
            "index_select",
            "index_add",
            "index_put",
            "index",
            "scatter",
            "all_reduce",
            "all_to_all",
        )
        emb_ops = sorted({t for t in targets if any(k in t for k in keys)})
        logger.info(
            "[RECIR_DUMP_GRAPHS] subgraph=%s backend=%s n_call_function=%d "
            "embedding/index/scatter ops=%s",
            tag,
            self.backend_type.value,
            len(targets),
            emb_ops,
        )

        # 1b. For each index/embedding/scatter node, report operand shapes and
        # walk operand-0 back to its source placeholder/get_attr (parameter) so
        # we can identify WHICH table (by [vocab, D] shape + param name) each
        # index_add/index_select belongs to. stack_trace is stripped by the
        # recir transforms, but this data-flow info is not.
        def _shape(n: Any) -> str:
            v = getattr(n, "meta", {}).get("val") if hasattr(n, "meta") else None
            return str(tuple(v.shape)) if v is not None and hasattr(v, "shape") else "?"

        _passthrough: Tuple[str, ...] = (
            "view",
            "reshape",
            "_to_copy",
            "to.",
            "clone",
            "detach",
            "expand",
            "t.",
            "transpose",
            "unsqueeze",
            "squeeze",
            "contiguous",
            "_unsafe_view",
            "constant_pad_nd",
            "slice",
            "select.",
        )

        def _trace_source(n: Any, depth: int = 12) -> str:
            cur = n
            for _ in range(depth):
                if not hasattr(cur, "op"):
                    return f"const({cur})"
                if cur.op in ("placeholder", "get_attr"):
                    return f"{cur.op}:{cur.target} {_shape(cur)}"
                if cur.op == "call_function" and cur.args:
                    t = str(cur.target)
                    a0 = cur.args[0]
                    if any(p in t for p in _passthrough) and hasattr(a0, "op"):
                        cur = a0
                        continue
                    return f"{t} {_shape(cur)}"
                return f"{getattr(cur, 'op', '?')}:{getattr(cur, 'target', cur)}"
            return "<too deep>"

        node_keys = ("index_select", "index_add", "index_put", "index", "embedding")
        for n in self.gm.graph.nodes:
            if n.op != "call_function":
                continue
            t = str(n.target)
            if not any(k in t for k in node_keys):
                continue
            try:
                arg_shapes = [
                    _shape(a) if hasattr(a, "op") else f"lit:{a}"[:80]
                    for a in n.args[:2]
                ]
                weight_src = _trace_source(n.args[0]) if n.args else "?"
                logger.info(
                    "[RECIR_DUMP_GRAPHS] subgraph=%s node=%s target=%s out=%s "
                    "args=%s weight<-%s",
                    tag,
                    n.name,
                    t,
                    _shape(n),
                    arg_shapes,
                    weight_src,
                )
            except Exception as e:  # never let diagnostics break the run
                logger.warning(
                    "[RECIR_DUMP_GRAPHS] node=%s trace failed: %s", n.name, e
                )
        if full:
            logger.info(
                "[RECIR_DUMP_GRAPHS] subgraph=%s FX graph:\n%s",
                tag,
                self.gm.graph,
            )

        # 2. XLA HLO (collectives + scatters, with shapes + op_name metadata).
        hlo = self.xla_hlo_text
        if hlo:
            if full:
                logger.info("[RECIR_DUMP_GRAPHS] subgraph=%s HLO:\n%s", tag, hlo)
            else:
                lines = [
                    ln.strip()
                    for ln in hlo.splitlines()
                    if "=" in ln
                    and any(
                        k in ln.lower() for k in ("all-reduce", "all-gather", "scatter")
                    )
                ]
                logger.info(
                    "[RECIR_DUMP_GRAPHS] subgraph=%s HLO collectives/scatters "
                    "(%d lines):\n%s",
                    tag,
                    len(lines),
                    "\n".join(ln[:240] for ln in lines),
                )
        else:
            logger.info("[RECIR_DUMP_GRAPHS] subgraph=%s HLO not captured yet", tag)

    def _postprocess_jax_output(
        self, jax_out: Any, input_device: torch.device
    ) -> List[Any]:
        if not self.keep_jax_output:
            return _jax_to_torch(jax_out, input_device)

        items = list(jax_out) if isinstance(jax_out, (list, tuple)) else [jax_out]
        torch_indices = self.torch_output_indices
        if not torch_indices:
            return items
        return [
            _jax_to_torch(item, input_device) if i in torch_indices else item
            for i, item in enumerate(items)
        ]

    def _compile_dynamo_bridge(
        self, *args: torch.Tensor
    ) -> Callable[..., List[torch.Tensor]]:
        """Compile via torch_xla.core.dynamo_bridge.

        extract_compiled_graph:
          1. Copies the GraphModule to the lazy/XLA device.
          2. Runs a forward pass to build lazy IR.
          3. Extracts and caches the HLO computation keyed by graph hash.
          4. Returns a callable that re-invokes the cached computation.
        """
        try:
            import torch_xla.core.dynamo_bridge as bridge  # @manual  # pyre-fixme[21]
        except ImportError as e:
            raise ImportError(
                "torch_xla is required for the DYNAMO_BRIDGE backend. "
                "Install via: https://github.com/pytorch/xla#pytorchxla"
            ) from e

        return bridge.extract_compiled_graph(self.gm, args)

    def _has_collective_ops(self) -> bool:
        """Check if the subgraph contains _c10d_functional collective ops."""
        for node in self.gm.graph.nodes:
            if node.op == "call_function" and isinstance(
                node.target, torch._ops.OpOverload
            ):
                if node.target.name().startswith("_c10d_functional::"):
                    return True
        return False

    def _compile_xla(self, *args: torch.Tensor) -> Callable[..., List[torch.Tensor]]:
        """Compile the subgraph via torchax/JAX for XLA_SHARD_MAP or XLA_GSPMD.

        Both backends share the same JAX-traceable function
        (``torchax.interop.call_torch``) since segment_and_lower ensures
        all subgraph inputs are tensors.  The difference is in how
        ``jax.jit`` is invoked:

        - **XLA_SHARD_MAP with collectives**: wrapped in ``shard_map``
          with replicated P() specs so ``jax.lax.psum`` etc. can
          reference the ``"torch_dist"`` axis.
        - **XLA_SHARD_MAP without collectives**: plain ``jax.jit`` on a
          single device.
        - **XLA_GSPMD**: ``jax.jit`` with explicit ``in_shardings`` /
          ``out_shardings`` read from per-node ``node.meta`` attributes.

        Each ``_build_*_executable`` helper returns a ``run`` closure that, on
        first invocation, materializes a ``jax.stages.Compiled`` and stashes
        it on ``self._jax_compiled``. AOT callers
        (:func:`recir.lowering.aot.aot_prebuild_and_serialize`) trigger this
        by issuing a single warmup call to the parent ``_XLAModule`` so the
        cached executable can be serialized.
        """
        torchax.default_env()
        gm: torch.fx.GraphModule = self.gm
        input_device: torch.device = args[0].device if args else torch.device("cpu")
        local_devs = _get_local_devices()
        jax_device: Any = local_devs[0] if local_devs else _get_global_devices()[0]

        def jax_fn(*jax_args: Any) -> Any:
            return torchax.interop.call_torch(gm, *jax_args)

        if self.backend_type == BackendType.XLA_GSPMD:
            lowered, run = self._jit_gspmd(jax_fn, args, input_device)
        elif self._has_collective_ops():
            lowered, run = self._jit_shard_map(jax_fn, args, input_device)
        else:
            lowered, run = self._jit_plain(jax_fn, args, input_device, jax_device)

        # For shard_map, capture HLO from lowering (includes shard_map ops).
        # For GSPMD, HLO is captured lazily after compilation (run.hlo_text)
        # because GSPMD inserts collectives during XLA compilation, not lowering.
        if self.capture_hlo_text and self.backend_type != BackendType.XLA_GSPMD:
            self.xla_hlo_text = lowered.as_text()
        logger.info(
            "XLA HLO compiled (backend=%s)",
            self.backend_type.value,
        )
        return run

    @staticmethod
    def _torch_to_numpy_for_jax(tensor: torch.Tensor) -> Any:
        """Convert a torch tensor to a numpy array with the matching JAX dtype.

        Returns a plain ``numpy.ndarray`` (with ``ml_dtypes`` dtypes where
        numpy lacks native support, e.g. ``bfloat16``). The caller passes
        the array directly to ``jax.make_array_from_process_local_data`` —
        no JAX entry happens here, so this path never emits a tracer-level
        ``jit(convert_element_type)``.

        Why this matters: the previous implementation routed bf16 through
        ``bf16 torch -> fp32 torch -> numpy fp32 -> jnp.asarray ->
        astype(jnp.bfloat16)``. That ``astype(jnp.bfloat16)`` is a JAX
        primitive bind, which under multi-host TPU with
        ``JAX_SHARE_BINARY_BETWEEN_HOSTS=true`` triggers a per-weight
        ``jit(stage)+jit(convert_element_type)`` pair whose cross-host
        rendezvous deadlocks at ``_convert_element_type_bind_with_trace``
        once the bf16 weight count crosses ~50. Reinterpreting the bits
        through ``view(int16) -> view(ml_dtypes.bfloat16)`` is pure CPU,
        zero JAX entry, and bit-identical.
        """
        t = tensor.detach().cpu().contiguous()
        if t.dtype == torch.bfloat16:
            # ``torch.Tensor.view(dtype)`` requires contiguous memory and
            # reinterprets the underlying bits without copy. The numpy
            # int16 view shares the same bits, and ``ndarray.view`` to
            # the ``ml_dtypes.bfloat16`` dtype is also a zero-copy
            # reinterpretation.
            # pyre-ignore[16]: ml_dtypes.bfloat16 is dynamically defined in the C extension.
            return t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
        return t.numpy()

    @staticmethod
    def _to_sharded_jax(  # noqa: C901
        tensor: Any,
        mesh: Any,
        is_batch_input: bool = False,
        input_batch_dim: int = 0,
        is_fsdp_tensor: bool = False,
        batch_axis_resource: Any = "torch_dist",
        fsdp_axis_name: str = "torch_dist",
        fsdp_host_shard_count: int = 1,
    ) -> Any:
        """Convert a PyTorch tensor to a properly sharded JAX array.

        Routes torch tensors through a pure-numpy intermediate (via
        :meth:`_torch_to_numpy_for_jax`) — JAX is only entered at the
        final :func:`jax.make_array_from_process_local_data`. This avoids
        the per-weight ``jit(stage)+jit(convert_element_type)``
        micro-compile pair that the previous bf16 round-trip
        (``fp32 torch -> numpy fp32 -> astype(jnp.bfloat16)``) emitted
        and that deadlocked multi-host TPU runs above ~50 bf16 weights.
        """
        if tensor is None:
            return None
        if not isinstance(tensor, torch.Tensor):
            assert isinstance(tensor, jax.Array), (
                f"Expected torch.Tensor or jax.Array, got {type(tensor)}"
            )
            # A jax.Array reaches here as a cached device_state entry (the
            # XLADeviceStateManager fast path): _convert_flat_state_to_jax
            # device_puts the model state onto a single local device on the
            # first step, so it carries a SingleDeviceSharding. On a multi-host
            # mesh that disagrees with the GSPMD executable's declared
            # in_sharding and raises "Computation was compiled for input
            # shardings that disagree with the shardings of arguments". Re-shard
            # it to the layout this input expects -- the SAME layout a torch
            # tensor would get -- so the fix is generic across replicated /
            # batch / FSDP rather than special-cased to one of them.
            #
            # AOT prebuild uses a compile-only client that cannot device_put;
            # the run closure short-circuits before exec, so pass through.
            if _AOT_TOPOLOGY.get() is not None:
                return tensor
            # Target sharding for this input (mirrors the torch branches below).
            ndim = tensor.ndim
            if is_batch_input and 0 <= input_batch_dim < ndim:
                _spec: List[Any] = [None] * max(ndim, input_batch_dim + 1)
                _spec[input_batch_dim] = batch_axis_resource
                target = NamedSharding(mesh, P(*_spec))
            elif is_fsdp_tensor:
                target = NamedSharding(
                    mesh,
                    P(fsdp_axis_name, *([None] * (ndim - 1))),
                )
            else:
                target = NamedSharding(mesh, P())
            # Fast path: already laid out correctly (e.g. device_state captured
            # from the executable's mesh-sharded outputs on later steps) -> reuse
            # as-is, avoiding a host round-trip on every step.
            if tensor.sharding == target:
                return tensor
            # A device_state entry from a PRIOR step's output that already spans
            # the mesh across processes (not fully addressable) can't be
            # host-fetched -- ``jax.device_get`` raises "Fetching value for
            # jax.Array that spans non-addressable devices". This is the
            # shard_map multi-process feedback case: the updated FSDP weight
            # comes back Shard(0)-over-N but with a sharding object that != the
            # target NamedSharding, so the fast path above misses. Reshard
            # on-device via a jitted identity (multi-process safe, no host
            # round-trip) instead of falling through to the host-buffer assembly.
            #
            # Replicated batch intermediates may still have a host-local shape.
            # Resharding changes only their layout, so rebuild them as global batches.
            if not tensor.is_fully_addressable:
                replicated = NamedSharding(mesh, P())
                if is_batch_input and tensor.sharding == replicated:
                    shards = tensor.addressable_shards
                    nparray = np.asarray(jax.device_get(shards[0].data))
                else:
                    return _cached_reshard(target)(tensor)
            else:
                # Fully-addressable (single-device / single-process) array: bring to
                # host and fall through to the unified construction below. NB a plain
                # jax.device_put(tensor, target) fails here ("CopyArrays only
                # supports destination device list of the same size ...") -- a
                # single-device array can't be copied straight onto the global mesh;
                # the local-buffer assembly below is the multi-process-safe path.
                nparray = np.asarray(jax.device_get(tensor))
        else:
            # Meta FSDP tensors have no host data; materialize each shard
            # directly on-device (see the helper).
            if is_fsdp_tensor and tensor.is_meta and tensor.ndim >= 1:
                return _materialize_meta_fsdp_tensor(tensor, mesh)

            nparray = _XLAModule._torch_to_numpy_for_jax(tensor)

        num_local = len(_get_local_devices())
        # Honor :func:`aot_topology` here too: when set, the topology's
        # devices are reported globally, but ``_get_local_devices`` returns
        # the per-process slice so ``num_global // num_local`` still
        # encodes the host count.
        num_global = len(_get_global_devices())
        # AOT mode: the compile-only client returned by
        # ``get_topology_desc`` raises ``MakeArraysFromHostBufferShards
        # not available with compile-only client`` from
        # ``jax.make_array_from_process_local_data``. Lowering only needs
        # abstract input shapes/dtypes/shardings, so return
        # ``jax.ShapeDtypeStruct`` instead. The corresponding ``run``
        # closures (see ``_build_*_executable``) short-circuit before
        # invoking the executable so these abstract values are never
        # executed.
        aot_active = _AOT_TOPOLOGY.get() is not None
        # Local→global multiplier. Use the topology-aware ratio
        # (num_global // num_local) so prebuild and runtime agree on the
        # global batch shape:
        #   * Real TPU runtime: jax.local_devices() / jax.devices() gives
        #     num_local=8, num_global=256 → multiplier=32 (=
        #     jax.process_count()).
        #   * aot_topology prebuild on a CPU host targeting a 4x4x8
        #     virtual topology: _get_local_devices filters topo.devices
        #     by process_index=0 (=8), _get_global_devices=256 →
        #     multiplier=32. Matches runtime. jax.process_count() on the
        #     CPU host is 1, so using THAT would compile for a different
        #     global shape than the runtime executes against and the
        #     AOT-loaded executable would TypeError on the first call.
        # `_make_trace_input_transform` in recir.api._compile_gspmd must use
        # the same multiplier (see GSPMDParallel.num_processes for the
        # matching property).
        host_count = num_global // num_local
        if is_batch_input and 0 <= input_batch_dim < nparray.ndim:
            global_shape = list(nparray.shape)
            global_shape[input_batch_dim] *= host_count
            ndim = len(global_shape)
            spec_parts: List[Any] = [None] * max(ndim, input_batch_dim + 1)
            spec_parts[input_batch_dim] = batch_axis_resource
            sharding = NamedSharding(mesh, P(*spec_parts))
            if aot_active:
                return jax.ShapeDtypeStruct(
                    tuple(global_shape), nparray.dtype, sharding=sharding
                )
            return jax.make_array_from_process_local_data(
                sharding, nparray, tuple(global_shape)
            )
        if is_fsdp_tensor and nparray.ndim >= 1:
            # FSDP: input is a local shard (after shard_params). Same
            # rationale as the is_batch_input branch above -- the
            # local→global multiplier here must match the gm's baked
            # global shape (via _make_trace_input_transform's host_count).
            global_shape = list(nparray.shape)
            global_shape[0] *= fsdp_host_shard_count
            ndim = len(global_shape)
            spec = P(fsdp_axis_name, *([None] * (ndim - 1)))
            sharding = NamedSharding(mesh, spec)
            if aot_active:
                return jax.ShapeDtypeStruct(
                    tuple(global_shape), nparray.dtype, sharding=sharding
                )
            return jax.make_array_from_process_local_data(
                sharding, nparray, tuple(global_shape)
            )
        sharding = NamedSharding(mesh, P())
        if aot_active:
            return jax.ShapeDtypeStruct(
                tuple(nparray.shape), nparray.dtype, sharding=sharding
            )
        return jax.make_array_from_process_local_data(
            sharding, nparray, tuple(nparray.shape)
        )

    def _jit_shard_map(
        self,
        jax_fn: Callable[..., Any],
        args: Tuple[torch.Tensor, ...],
        input_device: torch.device,
    ) -> Any:
        """jax.jit wrapped in shard_map for collectives."""
        # NOTE: unlike ``_jit_gspmd`` (which rebuilds ``runtime_mesh``
        # inside its ``run`` closure to defend against the AOT-prebuild
        # virtual-mesh pitfall), here ``mesh`` is captured at build time
        # and reused inside ``run``. This is intentional: ``shard_map``
        # binds the mesh into the JIT'd computation itself (see the
        # ``shard_map(jax_fn, mesh=mesh, ...)`` call below), and the
        # resulting ``lowered`` / ``jitted`` artifacts would not see a
        # rebuilt mesh anyway — swapping ``mesh`` inside ``run`` is
        # either a no-op (lowering is frozen) or forces a recompile.
        # The legacy shard_map path is only taken when
        # ``backend_type != XLA_GSPMD``; the AOT-prebuild workflow
        # exclusively uses the GSPMD path, so this build-time capture
        # is not exercised under a virtual-devices ``aot_topology``
        # context in practice.
        devices = _get_global_devices()
        local_device_count = len(_get_local_devices())
        if (
            not devices
            or local_device_count == 0
            or len(devices) % local_device_count != 0
        ):
            raise RuntimeError(
                "recir shard_map requires a non-empty local device set that "
                "evenly divides the global device mesh"
            )
        fsdp_host_shard_count = len(devices) // local_device_count
        mesh: Any = Mesh(devices, axis_names=("torch_dist",))
        placeholders = [n for n in self.gm.graph.nodes if n.op == "placeholder"]
        is_batch: List[bool] = [
            n.meta.get("is_batch_input", False) for n in placeholders
        ]
        is_fsdp: List[bool] = [get_state_layout(n) is not None for n in placeholders]
        # FSDP weights are fed as /N per-device shards (Shard(0)); the baked
        # all-gather in the traced body reconstructs the full weight. Batch
        # inputs are Shard(0) along the batch dim. Everything else replicated.
        # Without the is_fsdp branch, weights get P() (replicated) and the
        # body's all-gather double-expands them to N*full.
        in_specs = tuple(
            P("torch_dist") if (is_batch[i] or is_fsdp[i]) else P()
            for i in range(len(placeholders))
        )
        # Output specs mirror ``_jit_gspmd``: FSDP-derived outputs (e.g. the
        # updated weights, which are /N shards after the in-body reduce-scatter)
        # are Shard(0) so shard_map concatenates the per-device pieces back into
        # the global tensor; user (torch) outputs, scalars (e.g. a reduced loss),
        # and replicated outputs stay P() (see the per-leaf handling below). With
        # a blanket P() (the old behavior) an updated /N weight would be read as
        # device-0's shard only and shrink by N every step.
        output_node = next(n for n in self.gm.graph.nodes if n.op == "output")
        torch_indices: set[int] = self.torch_output_indices or set()
        out_leaves, out_treedef = jax.tree.flatten(output_node.args[0])
        placeholder_indices = {n: k for k, n in enumerate(placeholders)}

        def _leaf_ndim(node: Any) -> int:
            val = (
                node.meta.get("val", None) if isinstance(node, torch.fx.Node) else None
            )
            return val.ndim if hasattr(val, "ndim") else 0

        out_spec_list: List[Any] = []
        for i, leaf in enumerate(out_leaves):
            ndim = _leaf_ndim(leaf)
            out_val = (
                leaf.meta.get("val", None) if isinstance(leaf, torch.fx.Node) else None
            )
            out_shape = (
                tuple(out_val.shape)
                if out_val is not None and hasattr(out_val, "shape")
                else None
            )
            fsdp_src = (
                self._find_fsdp_source(leaf, placeholder_indices, is_fsdp, out_shape)
                if isinstance(leaf, torch.fx.Node)
                else None
            )
            # FSDP-derived user outputs, such as full-tensor diagnostics, still
            # carry a /N value on each device. Shard the output so host
            # materialization joins all addressable device shards into /P.
            # Other user outputs preserve the historical replicated contract.
            if i in torch_indices and fsdp_src is None:
                out_spec_list.append(P())
                continue
            if fsdp_src is not None and ndim >= 1:
                out_spec_list.append(P("torch_dist", *([None] * (ndim - 1))))
            elif fsdp_src is not None:
                # An FSDP-derived output with no usable ndim meta is an
                # unexpected meta-propagation gap. User (torch) outputs are
                # already handled above, so this leaf is a carried weight /
                # optimizer state whose /N Shard(0) out-spec we cannot build. A
                # replicated P() fallback would read only device-0's shard and
                # shrink the updated weight by mesh size every step, so fail
                # loudly instead of silently mis-sharding.
                raise RuntimeError(
                    "recir shard_map: FSDP output leaf "
                    f"{getattr(leaf, 'name', leaf)!r} traces to an FSDP source "
                    f"placeholder (#{fsdp_src}) but has no usable ndim meta "
                    f"(ndim={ndim}); cannot build its Shard(0) out-spec."
                )
            else:
                out_spec_list.append(P())
        out_specs = out_treedef.unflatten(out_spec_list)
        sharded_fn = shard_map(  # pyre-fixme[28]
            jax_fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            # pyrefly: ignore [unexpected-keyword]
            check_vma=False,
        )
        # Pin the output layouts to the SAME NamedShardings the next step's
        # _to_sharded_jax expects as input targets. Without this the shard_map
        # (manual-mode, check_vma=False) outputs carry a sharding object that !=
        # the target NamedSharding, so every device_state feedback misses the
        # `tensor.sharding == target` fast path and reshards each of the ~180
        # FSDP weights every step. With it the outputs already match -> zero
        # per-step reshard (the multi-process feedback loop is a no-op copy).
        out_shardings = out_treedef.unflatten(
            [NamedSharding(mesh, spec) for spec in out_spec_list]
        )
        safe_donate = self.safe_donate_indices or set()
        donate_argnums = tuple(
            i
            for i, batch_input in enumerate(is_batch)
            if not batch_input and i in safe_donate
        )
        logger.info(
            "shard_map: donating %d/%d input buffers (state) for in-place update",
            len(donate_argnums),
            len(placeholders),
        )
        jitted: Any = jax.jit(
            sharded_fn,
            out_shardings=out_shardings,
            donate_argnums=donate_argnums,
        )
        example_global_args = tuple(
            self._to_sharded_jax(
                a,
                mesh,
                is_batch_input=is_batch[j],
                is_fsdp_tensor=is_fsdp[j],
                fsdp_host_shard_count=fsdp_host_shard_count,
            )
            for j, a in enumerate(args)
        )
        # Annotate to silence pyre [53] on the ``lowered`` capture below;
        # ``jitted.lower`` returns ``jax.stages.Lowered`` whose stubs are
        # ``Any`` so the inferred type is ``Any`` already.
        lowered: Any = jitted.lower(*example_global_args)

        def run(*run_args: torch.Tensor) -> List[torch.Tensor]:
            # AOT prebuild mode: lower+compile against the closure-
            # captured abstract args; never execute on a compile-only
            # topology. ``_to_sharded_jax`` would raise here too.
            if _AOT_TOPOLOGY.get() is not None:
                if self._jax_compiled is None:
                    self._jax_compiled = lowered.compile()
                    _trim_host_memory_after_compile()
                    self._log_compiled_memory_analysis()
                return []
            global_args = tuple(
                self._to_sharded_jax(
                    a,
                    mesh,
                    is_batch_input=is_batch[j],
                    is_fsdp_tensor=is_fsdp[j],
                    fsdp_host_shard_count=fsdp_host_shard_count,
                )
                for j, a in enumerate(run_args)
            )
            # Prefer an AOT-installed or already-built executable so we
            # capture ``jax.stages.Compiled`` exactly once (the JAX
            # cache otherwise re-invokes lower+compile under the hood).
            # When nothing is cached, build on first invocation.
            if self._jax_compiled is None:
                self._jax_compiled = jitted.lower(*global_args).compile()
                _trim_host_memory_after_compile()
                self._log_compiled_memory_analysis()
            jax_out = self._jax_compiled(*global_args)
            return self._postprocess_jax_output(jax_out, input_device)

        return lowered, run

    @staticmethod
    def _t2j_safe(a: Any, jax_device: Any) -> Any:
        """Convert a torch.Tensor to a JAX array, passing through None.

        AOT prebuild mode: the compile-only client cannot host real
        buffers, so return a ``jax.ShapeDtypeStruct`` instead. The
        ``_build_plain_executable`` ``run`` closure short-circuits
        before invoking the executable so the abstract value is only
        used by lowering.
        """
        if a is None:
            return None
        if not isinstance(a, torch.Tensor):
            return a
        if _AOT_TOPOLOGY.get() is not None:
            jx = t2j(a)
            return jax.ShapeDtypeStruct(tuple(jx.shape), jx.dtype)
        return jax.device_put(t2j(a), jax_device)

    def _jit_plain(
        self,
        jax_fn: Callable[..., Any],
        args: Tuple[torch.Tensor, ...],
        input_device: torch.device,
        jax_device: Any,
    ) -> Any:
        """Plain jax.jit on a single device (no collectives)."""
        example_jax_args = tuple(self._t2j_safe(a, jax_device) for a in args)
        lowered = jax.jit(jax_fn).lower(*example_jax_args)
        compiled: Any = lowered.compile()
        # Stash on the module so AOT serialization can pick it up — but
        # only when nothing was AOT-loaded already.  Mirrors the guard in
        # ``_jit_gspmd`` / ``_jit_shard_map``: an unconditional
        # assignment here would clobber an AOT-deserialized executable
        # installed by ``aot_load_prebuilt_into`` and force a recompile
        # on the next forward pass.
        if self._jax_compiled is None:
            self._jax_compiled = compiled

        def run(*run_args: torch.Tensor) -> List[torch.Tensor]:
            # AOT prebuild mode: skip execution; ``compiled`` is already
            # captured on ``self._jax_compiled`` above.
            if _AOT_TOPOLOGY.get() is not None:
                return []
            jax_args = tuple(self._t2j_safe(a, jax_device) for a in run_args)
            # Prefer the AOT-installed executable when present; fall
            # back to the eagerly built ``compiled`` from the enclosing
            # scope.
            executable = self._jax_compiled or compiled
            jax_out = executable(*jax_args)
            return self._postprocess_jax_output(jax_out, input_device)

        return lowered, run

    @staticmethod
    def _batch_boundary_output_sharding(
        leaf: Any,
        mesh: Mesh,
        batch_axis_resource: Any = "torch_dist",
    ) -> Optional[NamedSharding]:
        if not isinstance(leaf, torch.fx.Node) or not leaf.meta.get(
            "is_batch_input", False
        ):
            return None
        val = leaf.meta.get("val")
        ndim = val.ndim if hasattr(val, "ndim") else 0
        batch_dim = leaf.meta.get("batch_dim", 0)
        if not 0 <= batch_dim < ndim:
            return None
        partitions: List[Any] = [None] * ndim
        partitions[batch_dim] = batch_axis_resource
        return NamedSharding(mesh, P(*partitions))

    @staticmethod
    def _is_replicated_final_output(
        index: int, leaf: Any, final_output_indices: set[int]
    ) -> bool:
        if index not in final_output_indices:
            return False
        return not (
            isinstance(leaf, torch.fx.Node)
            and (
                leaf.meta.get("is_fsdp_sharded", False)
                or leaf.meta.get("is_fsdp_sharded_output", False)
            )
        )

    @staticmethod
    def _find_fsdp_source(
        node: torch.fx.Node,
        placeholder_indices: dict[torch.fx.Node, int],
        is_fsdp_sharded: List[bool],
        out_shape: Optional[Tuple[int, ...]] = None,
    ) -> Optional[int]:
        visited: set[torch.fx.Node] = set()
        queue = [node]
        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            if n.op == "placeholder":
                idx = placeholder_indices.get(n)
                if idx is not None and is_fsdp_sharded[idx]:
                    # Only attribute FSDP sharding to this output if the FSDP
                    # placeholder is the param being CARRIED/updated by the output
                    # (same shape). Without this, a *replicated* param's grad/update
                    # -- which reaches an FSDP param through the shared loss graph --
                    # would be mis-sharded P("torch_dist", None) (breaks when mixing
                    # FSDP + replicated params, e.g. table-only sharding: a small
                    # [100, D] table or a dense weight grad traced to the big table).
                    if out_shape is None:
                        return idx
                    pval = n.meta.get("val", None)
                    pshape = (
                        tuple(getattr(pval, "shape", ())) if pval is not None else None
                    )
                    if pshape == out_shape:
                        return idx
                continue  # don't traverse beyond any placeholder
            for inp in n.all_input_nodes:
                if inp not in visited:
                    queue.append(inp)
        return None

    def _jit_gspmd(  # noqa: C901 — orchestration: builds the GSPMD mesh + per-input/per-output sharding spec + (host, device) routing in one place so the AOT compile + execute paths share a single source of truth.
        self,
        jax_fn: Callable[..., Any],
        args: Tuple[torch.Tensor, ...],
        input_device: torch.device,
    ) -> Any:
        """jax.jit with GSPMD in_shardings/out_shardings."""
        devices = _get_global_devices()
        fsdp_axis_size = int(
            getattr(self.gm, "meta", {}).get("gspmd_fsdp_axis_size", len(devices))
        )
        mesh, batch_axis_resource, hybrid_fsdp = _gspmd_mesh(
            devices,
            fsdp_axis_size,
        )
        fsdp_host_shard_count = fsdp_axis_size // len(_get_local_devices())
        replicated = NamedSharding(mesh, P())
        placeholders = [n for n in self.gm.graph.nodes if n.op == "placeholder"]
        is_batch: List[bool] = [
            n.meta.get("is_batch_input", False) for n in placeholders
        ]
        is_frozen_parameter: List[bool] = [
            n.meta.get("is_frozen_parameter", False) for n in placeholders
        ]
        # Per-placeholder batch_dim. Without _propagate_batch_input_meta this
        # would be 0 for everything with the propagation pass, intermediate FX nodes
        # crossing graph breaks land here with their actual batch axis
        per_input_batch_dim: List[int] = [
            n.meta.get("batch_dim", 0) if is_batch[i] else 0
            for i, n in enumerate(placeholders)
        ]
        is_fsdp_meta: List[bool] = [
            n.meta.get("is_fsdp_sharded", False) for n in placeholders
        ]
        per_input_fsdp_dim: List[int] = [
            n.meta.get("fsdp_shard_dim", 0) if is_fsdp_meta[i] else 0
            for i, n in enumerate(placeholders)
        ]
        # Build per-input sharding specs and track which inputs are physically
        # sharded.
        in_shardings = []
        is_batch_sharded: List[bool] = []
        is_fsdp_sharded: List[bool] = []
        for i, n in enumerate(placeholders):
            if is_batch[i]:
                val = n.meta.get("val", None)
                ndim = val.ndim if hasattr(val, "ndim") else 0
                if ndim == 0 and i < len(args) and hasattr(args[i], "ndim"):
                    ndim = args[i].ndim
                bd = per_input_batch_dim[i]
                if 0 <= bd < ndim:
                    spec_parts: List[Any] = [None] * max(ndim, bd + 1)
                    spec_parts[bd] = batch_axis_resource
                    in_shardings.append(NamedSharding(mesh, P(*spec_parts)))
                    is_batch_sharded.append(True)
                else:
                    in_shardings.append(replicated)
                    is_batch_sharded.append(False)
                is_fsdp_sharded.append(False)
            elif is_fsdp_meta[i] and i < len(args) and getattr(args[i], "ndim", 0) >= 1:
                ndim = args[i].ndim
                shard_dim = per_input_fsdp_dim[i]
                spec_parts = [None] * max(ndim, shard_dim + 1)
                spec_parts[shard_dim] = _GSPMD_FSDP_AXIS
                in_shardings.append(NamedSharding(mesh, P(*spec_parts)))
                is_batch_sharded.append(False)
                is_fsdp_sharded.append(True)
            else:
                in_shardings.append(replicated)
                is_batch_sharded.append(False)
                is_fsdp_sharded.append(False)
        batch_sharded = NamedSharding(mesh, P(batch_axis_resource))
        out_node = next(n for n in self.gm.graph.nodes if n.op == "output")
        torch_indices: set[int] = self.torch_output_indices or set()
        out_leaves, out_treedef = jax.tree.flatten(out_node.args[0])
        # Build per-output sharding specs using data-flow tracing.
        placeholder_indices = {n: i for i, n in enumerate(placeholders)}
        out_sharding_list = []
        for i in range(len(out_leaves)):
            leaf = out_leaves[i]
            if i in torch_indices:
                out_sharding_list.append(batch_sharded)
                continue
            if self._is_replicated_final_output(i, leaf, self.final_output_indices):
                out_sharding_list.append(replicated)
                continue
            batch_sharding = self._batch_boundary_output_sharding(
                leaf,
                mesh,
                batch_axis_resource,
            )
            if batch_sharding is not None:
                out_sharding_list.append(batch_sharding)
                continue
            out_val = (
                leaf.meta.get("val", None) if isinstance(leaf, torch.fx.Node) else None
            )
            explicit_fsdp_output = (
                leaf.meta.get("is_fsdp_sharded_output", None)
                if isinstance(leaf, torch.fx.Node)
                else None
            )
            if explicit_fsdp_output is False:
                out_sharding_list.append(replicated)
                continue
            if isinstance(leaf, torch.fx.Node) and leaf.meta.get(
                "is_fsdp_sharded", False
            ):
                ndim = out_val.ndim if hasattr(out_val, "ndim") else 0
                shard_dim = leaf.meta.get("fsdp_shard_dim", 0)
                if ndim >= 1 and 0 <= shard_dim < ndim:
                    partitions: list[Optional[str]] = [None] * ndim  # pyre-ignore
                    partitions[shard_dim] = _GSPMD_FSDP_AXIS
                    out_sharding_list.append(NamedSharding(mesh, P(*partitions)))
                else:
                    out_sharding_list.append(replicated)
                continue
            if explicit_fsdp_output is True:
                ndim = out_val.ndim if hasattr(out_val, "ndim") else 0
                if ndim >= 1:
                    out_sharding_list.append(
                        NamedSharding(
                            mesh,
                            P(_GSPMD_FSDP_AXIS, *([None] * (ndim - 1))),
                        )
                    )
                else:
                    out_sharding_list.append(replicated)
                continue
            out_shape = (
                tuple(out_val.shape)
                if out_val is not None and hasattr(out_val, "shape")
                else None
            )
            fsdp_src = (
                self._find_fsdp_source(
                    leaf, placeholder_indices, is_fsdp_sharded, out_shape
                )
                if isinstance(leaf, torch.fx.Node)
                else None
            )
            if fsdp_src is not None:
                leaf.meta["is_fsdp_sharded"] = True
                leaf.meta.setdefault("fsdp_shard_dim", 0)
                if out_val is not None and hasattr(out_val, "ndim"):
                    ndim = out_val.ndim
                elif fsdp_src < len(args) and hasattr(args[fsdp_src], "ndim"):
                    ndim = args[fsdp_src].ndim
                else:
                    ndim = 1
                if ndim >= 1:
                    spec = P(_GSPMD_FSDP_AXIS, *([None] * (ndim - 1)))
                    out_sharding_list.append(NamedSharding(mesh, spec))
                else:
                    out_sharding_list.append(replicated)
            else:
                out_sharding_list.append(replicated)
        if len(out_sharding_list) != len(out_leaves):
            raise RuntimeError(
                "GSPMD output sharding ABI must produce exactly one spec per leaf: "
                f"leaves={len(out_leaves)} specs={len(out_sharding_list)}"
            )
        out_shardings = out_treedef.unflatten(out_sharding_list)
        logger.info(
            "Compiling with GSPMD: %d placeholders, %d batch-sharded, %d "
            "weight-sharded, mesh=%s, fsdp_axis_size=%d, hybrid_fsdp=%s",
            len(placeholders),
            sum(is_batch_sharded),
            sum(is_fsdp_sharded),
            mesh.axis_names,
            fsdp_axis_size,
            hybrid_fsdp,
        )
        # Donate state inputs (weights, gradients, optimizer state) so XLA can
        # reuse their HBM buffers for the corresponding outputs (in-place
        # update). Without this, JAX runtime holds both old (W, G, M) and new
        # (W', G', M') simultaneously during the train step, doubling
        # persistent state from ~45 GB to ~90 GB per device on Llama-3-8B
        # and triggering HBM OOM at LoadProgram.
        safe_donate = self.safe_donate_indices or set()
        donate_argnums = tuple(
            i for i, b in enumerate(is_batch) if not b and i in safe_donate
        )
        logger.info(
            "GSPMD: donating %d/%d input buffers (state) for in-place update",
            len(donate_argnums),
            len(placeholders),
        )
        jitted: Any = jax.jit(
            jax_fn,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            donate_argnums=donate_argnums,
        )

        _hlo_captured: List[bool] = [False]

        def run(*run_args: torch.Tensor) -> List[torch.Tensor]:
            # AOT prebuild mode: lower+compile against the closure-
            # captured abstract args (see ``lowered`` below). Skip
            # execution — the caller (``aot_prebuild_and_serialize``)
            # only reads ``self._jax_compiled``.
            if _AOT_TOPOLOGY.get() is not None:
                if self._jax_compiled is None:
                    self._jax_compiled = lowered.compile()
                    _trim_host_memory_after_compile()
                    self._log_compiled_memory_analysis()
                if self.capture_hlo_text and not _hlo_captured[0]:
                    _hlo_captured[0] = True
                    run.hlo_text = self._jax_compiled.as_text()  # pyre-ignore[16]
                return []
            # IMPORTANT: rebuild the mesh from live JAX devices at call
            # time rather than reusing the closure-captured `mesh`. When
            # this _XLAModule was first instantiated under an
            # `aot_topology(virtual_topo)` context (the AOT-prebuild
            # warmup pattern), the captured `mesh` holds the VIRTUAL
            # devices. Subsequent real-runtime calls happen OUTSIDE
            # `aot_topology` and must use the real
            # `jax.devices()`-backed mesh, otherwise `_to_sharded_jax`
            # builds NamedShardings against virtual devices that don't
            # correspond to any addressable real device, and JAX SIGSEGVs
            # silently on first call.
            runtime_mesh, runtime_batch_axis_resource, runtime_hybrid_fsdp = (
                _gspmd_mesh(
                    _get_global_devices(),
                    fsdp_axis_size,
                )
            )

            runtime_fsdp_host_shard_count = fsdp_axis_size // len(_get_local_devices())
            runtime_cache_mesh = (
                tuple(runtime_mesh.axis_names),
                tuple(runtime_mesh.devices.flat),
                runtime_fsdp_host_shard_count,
            )

            def _convert_run_arg(j: int, a: Any) -> Any:
                cache_layout = (
                    runtime_cache_mesh,
                    is_fsdp_sharded[j],
                    per_input_fsdp_dim[j],
                )
                return _reuse_unchanged_state_input(
                    self._state_input_cache,
                    (self._state_input_cache_namespace, j),
                    a,
                    is_batch[j],
                    is_frozen_parameter[j],
                    functools.partial(
                        self._to_sharded_jax,
                        a,
                        runtime_mesh,
                        is_batch[j],
                        per_input_batch_dim[j],
                        is_fsdp_tensor=is_fsdp_sharded[j],
                        batch_axis_resource=runtime_batch_axis_resource,
                        fsdp_axis_name=_GSPMD_FSDP_AXIS,
                        fsdp_host_shard_count=runtime_fsdp_host_shard_count,
                    ),
                    cache_layout,
                )

            jax_args = tuple(_convert_run_arg(j, a) for j, a in enumerate(run_args))
            # First call materializes and stashes the compiled
            # executable so AOT serializers can pick it up (and so
            # subsequent calls bypass JAX's internal compile cache
            # lookup, which avoids a second compile during HLO capture).
            if self._jax_compiled is None:
                with gspmd_mesh_context(
                    runtime_mesh,
                    runtime_batch_axis_resource,
                    fsdp_axis_size,
                    runtime_hybrid_fsdp,
                ):
                    self._jax_compiled = jitted.lower(*jax_args).compile()
                _trim_host_memory_after_compile()
                self._log_compiled_memory_analysis()
            if self.capture_hlo_text and not _hlo_captured[0]:
                _hlo_captured[0] = True
                run.hlo_text = self._jax_compiled.as_text()  # pyre-ignore[16]
            jax_out = self._jax_compiled(*jax_args)
            return self._postprocess_jax_output(jax_out, input_device)

        run.hlo_text = None  # pyre-ignore[16]

        example_args = tuple(
            self._to_sharded_jax(
                a,
                mesh,
                is_batch[j],
                per_input_batch_dim[j],
                is_fsdp_tensor=is_fsdp_sharded[j],
                batch_axis_resource=batch_axis_resource,
                fsdp_axis_name=_GSPMD_FSDP_AXIS,
                fsdp_host_shard_count=fsdp_host_shard_count,
            )
            for j, a in enumerate(args)
        )
        # Annotate to silence pyre [53] on the ``lowered`` capture in the
        # ``run`` closure above (late binding); ``jitted.lower`` returns
        # ``jax.stages.Lowered`` whose stubs are ``Any``.
        with gspmd_mesh_context(
            mesh,
            batch_axis_resource,
            fsdp_axis_size,
            hybrid_fsdp,
        ):
            lowered: Any = jitted.lower(*example_args)
        return lowered, run

    def forward(self, *args: torch.Tensor, tag: str = "") -> List[torch.Tensor]:
        if self.compiled is None:
            logger.info(
                "Compiling subgraph (%s) to XLA [%s]...",
                tag,
                self.backend_type.value,
            )
            if self.backend_type == BackendType.XLA_DYNAMO_BRIDGE:
                self.compiled = self._compile_dynamo_bridge(*args)
            elif self.backend_type in (
                BackendType.XLA_SHARD_MAP,
                BackendType.XLA_GSPMD,
            ):
                self.compiled = self._compile_xla(*args)
            else:
                raise ValueError(f"Unknown XLA backend type: {self.backend_type}")
            logger.info("Completed XLA compilation for subgraph (%s)", tag)
        with torch.profiler.record_function(tag):
            assert self.compiled is not None
            result = self.compiled(*args)
            if self.xla_hlo_text is None and hasattr(self.compiled, "hlo_text"):
                self.xla_hlo_text = self.compiled.hlo_text  # pyre-ignore[16]
            try:
                self._maybe_dump_graphs(tag)
            except Exception as e:  # diagnostics must never break training
                logger.warning("[RECIR_DUMP_GRAPHS] dump failed: %s", e)
            return result


def _is_xla_compatible(node: torch.fx.Node) -> Tuple[bool, str]:
    """Determine whether an FX node can be lowered to XLA.

    XLA has a different op-coverage profile than Inductor:
      - Fused Adam IS supported (unlike Inductor).
      - Foreach ops ARE supported (registered in torchax).
      - _c10d_functional ops ARE supported (via shard_map in torchax).
      - Profiler ops are no-ops at execution time and must not cause
        graph breaks.
      - Custom C++ ops without XLA lowerings are NOT supported.
      - All ops must have a Meta kernel for fake-tensor tracing.

    This function is deliberately conservative; ops can be added to the
    allowlist as XLA coverage expands.
    """
    # Ops that torchax supports but lack a Meta kernel registration.
    _XLA_ALLOWLIST = {
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.full_like.default,
        torch.ops.aten.select.int,
        torch.ops.aten.lift_fresh_copy.default,
    }
    if node.target in _XLA_ALLOWLIST:
        return True, ""

    if isinstance(node.target, torch._ops.OpOverload):
        target: torch._ops.OpOverload = node.target
        op_name = target.name()
        if op_name.startswith("aten::_foreach_"):
            return True, ""
        if op_name.startswith("_c10d_functional::"):
            return True, ""
        if op_name.startswith("profiler::"):
            return True, ""
        if not has_meta_kernel(target):
            return False, f"{target} doesn't have a meta kernel registered"

    # Override dispatch functions inserted by install_graph_break_dispatch
    # are marked with xla_graph_break metadata — they must not be absorbed.
    if node.meta.get("xla_graph_break", False):
        return False, "graph break (override dispatch)"

    return True, ""


@functools.lru_cache(maxsize=1)
def _warn_uncached_transfer_unavailable() -> None:
    logger.warning(
        "jax.Array._single_device_array_to_np_array_did_copy is gone; falling "
        "back to jax.device_get, which caches each shard's host array and so "
        "holds every shard of a state entry resident at once. Re-point "
        "_single_device_array_to_numpy_uncached at the renamed entry point."
    )


def _single_device_array_to_numpy_uncached(t: Any) -> Any:
    """One device's array to numpy, without JAX caching the result on the source."""
    import numpy as np

    shards = getattr(t, "addressable_shards", ())
    if len(shards) > 1:
        raise ValueError(
            f"expected one device's array, got {len(shards)} shards; the JAX "
            "entry point below reads only the first. Iterate the shards, or "
            "use _jax_to_numpy, which handles multi-device arrays."
        )

    to_np = getattr(t, "_single_device_array_to_np_array_did_copy", None)
    if to_np is None:
        # Private JAX API; a version bump may rename it. Degrading to the
        # caching transfer keeps the flush correct, but silently reinstates the
        # pinning this exists to avoid -- which on 64 chips surfaces as an OOM
        # that looks like it came from somewhere else. Say so, once.
        _warn_uncached_transfer_unavailable()
        return _jax_to_numpy(t)
    host_arr = np.asarray(to_np()[0])
    if host_arr.dtype.name == "bfloat16":
        host_arr = host_arr.astype(np.float32)
    return host_arr


def _jax_to_numpy(t: Any) -> Any:
    """Convert a JAX array to a numpy array, handling bfloat16.

    Transfers data to host (CPU) first via jax.device_get() to avoid
    allocating extra device memory for dtype conversion. Then converts
    bfloat16 to float32 on CPU since torch.tensor() doesn't accept
    ml_dtypes.bfloat16.
    """
    import numpy as np

    # If `t` is a torch.Tensor (e.g. torchax-wrapped), np.asarray below
    # would route through Tensor.numpy() which raises TypeError for bf16 tensors.
    # Cast to fp32 in torch first; the bf16 fallback at the end of this
    # function preserves the same return-shape contract.
    if isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16:
        t = t.float()

    # Transfer from device to host first to avoid OOM on device
    host_arr = np.asarray(jax.device_get(t))
    # ml_dtypes.bfloat16 dtype name is 'bfloat16'
    if host_arr.dtype.name == "bfloat16":
        host_arr = host_arr.astype(np.float32)
    return host_arr


def _sharded_axis_and_shards(t: Any) -> Optional[Tuple[int, List[Any]]]:
    """The axis ``t``'s addressable shards are split on, and those shards in index order."""
    shards = t.addressable_shards
    if len(shards) <= 1 or shards[0].data.shape == t.shape:
        return None
    sharding_spec = getattr(getattr(t, "sharding", None), "spec", ())
    sharded_axes = [axis for axis, partition in enumerate(sharding_spec) if partition]
    if len(sharded_axes) > 1:
        # The shards are concatenated along a single axis below. Reassembling a
        # multi-axis sharded array that way places every shard at the wrong
        # offset and reports success -- a silently corrupt checkpoint.
        raise ValueError(
            f"cannot reassemble an array sharded on axes {sharded_axes}; "
            "only single-axis sharding is supported "
            f"(spec {tuple(sharding_spec)}, shape {tuple(t.shape)})"
        )
    axis = sharded_axes[0] if sharded_axes else 0
    return axis, sorted(shards, key=lambda s: s.index)


def _local_shard_shape(t: Any) -> Tuple[int, ...]:
    """Shape of the tensor this process's addressable shards reassemble into."""
    split = _sharded_axis_and_shards(t)
    if split is None:
        return tuple(t.addressable_shards[0].data.shape)
    axis, shards = split
    shape = list(shards[0].data.shape)
    shape[axis] = sum(s.data.shape[axis] for s in shards)
    return tuple(shape)


def _copy_jax_addressable_shards_to_torch(t: Any, dest: torch.Tensor) -> None:
    """Reassemble the addressable shards of JAX array ``t`` into ``dest``, in place."""
    local_shape = _local_shard_shape(t)
    if tuple(dest.shape) != local_shape:
        raise ValueError(
            f"destination shape {tuple(dest.shape)} does not match the shape "
            f"this process's shards reassemble into, {local_shape} "
            f"(jax array is globally {tuple(t.shape)})"
        )

    split = _sharded_axis_and_shards(t)
    if split is None:
        dest.copy_(
            torch.from_numpy(
                _single_device_array_to_numpy_uncached(t.addressable_shards[0].data)
            )
        )
        return

    axis, ordered_shards = split
    offset = 0
    for shard in ordered_shards:
        arr = _single_device_array_to_numpy_uncached(shard.data)
        length = arr.shape[axis]
        # `arr` is the only extra buffer alive, and only until the next
        # iteration rebinds it -- that is the whole point of the loop, and it
        # only holds because `_single_device_array_to_numpy_uncached` keeps JAX from caching it.
        dest.narrow(axis, offset, length).copy_(torch.from_numpy(arr))
        offset += length


def _jax_to_torch(t: Any, device: torch.device) -> Any:
    """Convert a JAX output (array, None, or nested list/tuple) to a torch tensor.

    For arrays with addressable shards, allocates the per-process host tensor
    (this process's shards only, not the global shape) and hands it to
    `_copy_jax_addressable_shards_to_torch`, which fills it shard by shard.

    Preserves bf16 dtype: numpy doesn't natively represent bf16 so the
    transfer round-trips through fp32 in `_jax_to_numpy`; the inverse
    cast back to bf16 happens here.  Without this cast, JAX outputs from
    a bf16-compiled program would be returned as fp32 torch tensors,
    which (when cached in `XLADeviceStateManager.device_state`) would
    feed fp32 inputs into the NEXT compiled step — triggering a JAX
    cache miss + re-compile of the same train_step in fp32.  See the
    "OOM #3" subsection of
    `mrs/rankevolve/rankagi/post_train/scripts/gpu/benchmark_results.md`
    (llama3-8b section) for the runtime evidence.
    """
    import numpy as np

    if t is None:
        return None
    if isinstance(t, list):
        return [_jax_to_torch(x, device) for x in t]
    if isinstance(t, tuple):
        return tuple(_jax_to_torch(x, device) for x in t)
    if isinstance(t, dict):
        return {k: _jax_to_torch(v, device) for k, v in t.items()}
    src_is_bf16 = hasattr(t, "dtype") and getattr(t.dtype, "name", "") == "bfloat16"
    if hasattr(t, "addressable_shards") and t.addressable_shards:
        if not t.shape:
            # A scalar has no shards to reassemble, and routing it through the
            # shard path would change its dtype: `.item()` has always
            # normalised to a python scalar's type, dropping bf16 to fp32.
            return torch.tensor(
                _jax_to_numpy(t.addressable_shards[0].data).item(), device=device
            )
        out = torch.empty(
            _local_shard_shape(t),
            dtype=(
                torch.bfloat16
                if src_is_bf16
                else torch.from_numpy(np.empty(0, dtype=t.dtype)).dtype
            ),
            device=device,
        )
        _copy_jax_addressable_shards_to_torch(t, out)
        return out
    np_arr = _jax_to_numpy(t)
    if np_arr.ndim == 0:
        return torch.tensor(np_arr.item(), device=device)
    out = torch.tensor(np_arr, device=device)
    if src_is_bf16:
        out = out.to(torch.bfloat16)
    return out


def _annotate_safe_donate_indices(gm: torch.fx.GraphModule) -> None:
    for call_node in gm.graph.nodes:
        if call_node.op != "call_module":
            continue
        target = getattr(gm, call_node.target, None)
        if not isinstance(target, _XLAModule):
            continue
        safe: set[int] = set()
        for i, arg in enumerate(call_node.args):
            if not isinstance(arg, torch.fx.Node):
                continue
            if set(arg.users.keys()) == {call_node}:
                safe.add(i)
        target.safe_donate_indices = safe


def _strip_mark_boundary_nodes(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Remove ``recir::mark_boundary`` identity ops before XLA segmentation.

    In the ``compile()`` decorator path, ``install_graph`` is never called so
    ``mark_boundary`` ops planted by ``install_eager`` remain in the graph.
    They have no torchax/JAX lowering, but marking them XLA-incompatible would
    create graph breaks that expose jax arrays to eager torch ops.  Since
    ``mark_boundary`` is pure identity (returns its first arg unchanged),
    we simply replace each call with its tensor input and erase the node.
    """
    changed = False
    for node in reversed(list(gm.graph.nodes)):
        if (
            node.op == "call_function"
            and isinstance(node.target, torch._ops.OpOverload)
            and node.target.name() == "recir::mark_boundary"
        ):
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)
            changed = True
    if changed:
        gm.graph.eliminate_dead_code()
        gm.recompile()
    return gm


def partial_lower_xla(
    gm: torch.fx.GraphModule,
    backend_type: BackendType = BackendType.XLA_SHARD_MAP,
    capture_hlo_text: bool = False,
    node_predicate: Callable[[torch.fx.Node], bool] = lambda x: True,
    subgraph_predicate: Callable[[List[torch.fx.Node]], bool] = lambda x: True,
    dumper: Callable[[str], str] = lambda x: "subgraph",
    op_swap_map: Optional[OpSwapRegistry] = None,
) -> torch.fx.GraphModule:
    if op_swap_map is not None:
        gm = swap_ops(gm, op_swap_map)
    # Strip profiler ops before segmentation so their ScriptObject
    # handles never become XLA subgraph inputs.
    gm = strip_profiler_nodes(gm)
    gm = _strip_mark_boundary_nodes(gm)
    # Re-propagate is_batch_input/batch_dim. The first pass runs at trace
    # time in ``recir.api._finalize_graph`` / ``recir.trace`` before any
    # post-trace mutation (for example, surgery that splices in new subgraphs).
    # Surgery-inserted nodes inherit only the component's traced meta -- no
    # ``is_batch_input`` tag -- so without re-propagation they become
    # replicated subgraph placeholders after ``segment_and_lower`` and
    # ``_jit_gspmd`` builds per-process shardings for them while the rest
    # of the graph carries global shapes -> JAX broadcast-shape mismatch
    # (e.g. ``[4096,1,128]`` vs ``[128,1,128]``).
    #
    # External surgery integrations may re-tag after every splice, but this
    # call also covers any caller that mutates the graph before reaching
    # ``partial_lower_xla``;
    # ``propagate_batch_input_meta`` is idempotent (skips already-tagged
    # nodes) so re-running is free for the surgery-tagged callers.
    propagate_batch_input_meta(gm)
    propagate_fsdp_sharded_meta(gm)
    gspmd_fsdp_axis_size = getattr(gm, "meta", {}).get("gspmd_fsdp_axis_size")

    def _module_factory(sub_gm: torch.fx.GraphModule) -> _XLAModule:
        if gspmd_fsdp_axis_size is not None:
            sub_gm.meta["gspmd_fsdp_axis_size"] = gspmd_fsdp_axis_size
        return _XLAModule(
            sub_gm,
            backend_type,
            capture_hlo_text=capture_hlo_text,
        )

    lowered = segment_and_lower(
        gm,
        compatibility_fn=_is_xla_compatible,
        module_factory=_module_factory,
        subgraph_prefix="xla_subgraph",
        node_predicate=node_predicate,
        subgraph_predicate=subgraph_predicate,
        dumper=dumper,
    )
    _annotate_safe_donate_indices(lowered)
    return lowered
