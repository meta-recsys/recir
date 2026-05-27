# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
from typing import Any, Callable, List, Optional, Tuple

import jax  # @manual
import jax.numpy as jnp  # @manual
import recir.torchax as torchax
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from recir.backend import BackendType
from recir.lowering.lowering_utils import segment_and_lower
from recir.lowering.op_swap import OpSwapRegistry, swap_ops
from recir.torchax.ops.mappings import t2j

try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

logger: logging.Logger = logging.getLogger(__name__)


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
    ) -> None:
        super().__init__()
        self.gm = gm
        self.backend_type = backend_type
        self.compiled: Optional[Callable[..., List[torch.Tensor]]] = None
        self.xla_hlo_text: Optional[str] = None
        self.keep_jax_output: bool = False
        self.torch_output_indices: Optional[set[int]] = None
        self.safe_donate_indices: Optional[set[int]] = None

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
        """
        torchax.default_env()
        gm: torch.fx.GraphModule = self.gm
        input_device: torch.device = args[0].device if args else torch.device("cpu")
        local_devs = jax.local_devices()
        jax_device: Any = local_devs[0] if local_devs else jax.devices()[0]

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
        if self.backend_type != BackendType.XLA_GSPMD:
            self.xla_hlo_text = lowered.as_text()
        logger.info(
            "XLA HLO compiled (backend=%s)",
            self.backend_type.value,
        )
        return run

    @staticmethod
    def _to_sharded_jax(
        tensor: Any,
        mesh: Any,
        is_batch_input: bool = False,
        input_batch_dim: int = 0,
        is_fsdp_weight: bool = False,
    ) -> Any:
        """Convert a PyTorch tensor to a properly sharded JAX array."""
        if tensor is None:
            return None
        if not isinstance(tensor, torch.Tensor):
            assert isinstance(tensor, jax.Array), (
                f"Expected torch.Tensor or jax.Array, got {type(tensor)}"
            )
            return tensor
        t = tensor.detach()
        if t.dtype == torch.bool:
            t = t.to(torch.int8)
        elif t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        nparray = t.cpu().numpy()
        jax_array = jnp.asarray(nparray)
        if tensor.dtype == torch.bool:
            jax_array = jax_array.astype(jnp.bool_)
        elif tensor.dtype == torch.bfloat16:
            jax_array = jax_array.astype(jnp.bfloat16)

        local_devices = jax.local_devices()
        num_local = len(local_devices)
        num_global = jax.device_count()
        if is_batch_input:
            local_shards = jnp.split(jax_array, num_local, axis=input_batch_dim)
            local_buffers = [
                jax.device_put(s, d) for s, d in zip(local_shards, local_devices)
            ]
            global_shape = list(jax_array.shape)
            global_shape[input_batch_dim] *= num_global // num_local
            # Build a PartitionSpec naming "torch_dist" on the actual batch
            # axis (input_batch_dim), with None on every other axis.
            ndim = len(global_shape)
            spec_parts: List[Any] = [None] * max(ndim, input_batch_dim + 1)
            spec_parts[input_batch_dim] = "torch_dist"
            return jax.make_array_from_single_device_arrays(
                tuple(global_shape),
                NamedSharding(mesh, P(*spec_parts)),
                local_buffers,
            )
        if is_fsdp_weight:
            # FSDP: input is a local shard (after shard_params).
            local_shards = jnp.split(jax_array, num_local, axis=0)
            local_buffers = [
                jax.device_put(s, d) for s, d in zip(local_shards, local_devices)
            ]
            global_shape = list(jax_array.shape)
            global_shape[0] *= num_global // num_local
            ndim = len(global_shape)
            spec = P("torch_dist", *([None] * (ndim - 1)))
            return jax.make_array_from_single_device_arrays(
                tuple(global_shape), NamedSharding(mesh, spec), local_buffers
            )
        local_buffers = [jax.device_put(jax_array, d) for d in local_devices]
        return jax.make_array_from_single_device_arrays(
            jax_array.shape, NamedSharding(mesh, P()), local_buffers
        )

    def _jit_shard_map(
        self,
        jax_fn: Callable[..., Any],
        args: Tuple[torch.Tensor, ...],
        input_device: torch.device,
    ) -> Any:
        """jax.jit wrapped in shard_map for collectives."""
        devices = jax.devices()
        mesh: Any = Mesh(devices, axis_names=("torch_dist",))
        placeholders = [n for n in self.gm.graph.nodes if n.op == "placeholder"]
        is_batch: List[bool] = [
            n.meta.get("is_batch_input", False) for n in placeholders
        ]
        in_specs = tuple(P("torch_dist") if b else P() for b in is_batch)
        output_node = next(n for n in self.gm.graph.nodes if n.op == "output")
        out_specs = jax.tree.map(lambda _: P(), output_node.args[0])
        sharded_fn = shard_map(  # pyre-fixme[28]
            jax_fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )
        jitted: Any = jax.jit(sharded_fn)
        example_global_args = tuple(
            self._to_sharded_jax(a, mesh, is_batch_input=is_batch[j])
            for j, a in enumerate(args)
        )
        lowered = jitted.lower(*example_global_args)

        def run(*run_args: torch.Tensor) -> List[torch.Tensor]:
            global_args = tuple(
                self._to_sharded_jax(a, mesh, is_batch_input=is_batch[j])
                for j, a in enumerate(run_args)
            )
            jax_out = jitted(*global_args)
            return self._postprocess_jax_output(jax_out, input_device)

        return lowered, run

    @staticmethod
    def _t2j_safe(a: Any, jax_device: Any) -> Any:
        """Convert a torch.Tensor to a JAX array, passing through None."""
        if a is None:
            return None
        if not isinstance(a, torch.Tensor):
            return a
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

        def run(*run_args: torch.Tensor) -> List[torch.Tensor]:
            jax_args = tuple(self._t2j_safe(a, jax_device) for a in run_args)
            jax_out = compiled(*jax_args)
            return self._postprocess_jax_output(jax_out, input_device)

        return lowered, run

    @staticmethod
    def _find_fsdp_source(
        node: torch.fx.Node,
        placeholder_indices: dict[torch.fx.Node, int],
        is_fsdp_sharded: List[bool],
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
                    return idx
                continue  # don't traverse beyond any placeholder
            for inp in n.all_input_nodes:
                if inp not in visited:
                    queue.append(inp)
        return None

    def _jit_gspmd(
        self,
        jax_fn: Callable[..., Any],
        args: Tuple[torch.Tensor, ...],
        input_device: torch.device,
    ) -> Any:
        """jax.jit with GSPMD in_shardings/out_shardings."""
        devices = jax.devices()
        mesh: Any = Mesh(devices, axis_names=("torch_dist",))
        replicated = NamedSharding(mesh, P())
        placeholders = [n for n in self.gm.graph.nodes if n.op == "placeholder"]
        is_batch: List[bool] = [
            n.meta.get("is_batch_input", False) for n in placeholders
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
        # Build per-input sharding specs and track which inputs are FSDP-sharded.
        in_shardings = []
        is_fsdp_sharded: List[bool] = []
        for i, n in enumerate(placeholders):
            if is_batch[i]:
                val = n.meta.get("val", None)
                ndim = val.ndim if hasattr(val, "ndim") else 1
                bd = per_input_batch_dim[i]
                spec_parts: List[Any] = [None] * max(ndim, bd + 1)
                spec_parts[bd] = "torch_dist"
                in_shardings.append(NamedSharding(mesh, P(*spec_parts)))
                is_fsdp_sharded.append(False)
            elif is_fsdp_meta[i] and i < len(args) and hasattr(args[i], "ndim"):
                ndim = args[i].ndim
                spec = P("torch_dist", *([None] * (ndim - 1)))
                in_shardings.append(NamedSharding(mesh, spec))
                is_fsdp_sharded.append(True)
            else:
                in_shardings.append(replicated)
                is_fsdp_sharded.append(False)
        batch_sharded = NamedSharding(mesh, P("torch_dist"))
        out_node = next(n for n in self.gm.graph.nodes if n.op == "output")
        torch_indices: set[int] = self.torch_output_indices or set()
        out_leaves, out_treedef = jax.tree.flatten(out_node.args[0])
        # Build per-output sharding specs using data-flow tracing.
        placeholder_indices = {n: i for i, n in enumerate(placeholders)}
        out_sharding_list = []
        for i in range(len(out_leaves)):
            if i in torch_indices:
                out_sharding_list.append(batch_sharded)
                continue
            leaf = out_leaves[i]
            fsdp_src = (
                self._find_fsdp_source(leaf, placeholder_indices, is_fsdp_sharded)
                if isinstance(leaf, torch.fx.Node)
                else None
            )
            if fsdp_src is not None:
                out_val = (
                    leaf.meta.get("val", None)
                    if isinstance(leaf, torch.fx.Node)
                    else None
                )
                if out_val is not None and hasattr(out_val, "ndim"):
                    ndim = out_val.ndim
                elif fsdp_src < len(args) and hasattr(args[fsdp_src], "ndim"):
                    ndim = args[fsdp_src].ndim
                else:
                    ndim = 1
                if ndim >= 1:
                    spec = P("torch_dist", *([None] * (ndim - 1)))
                    out_sharding_list.append(NamedSharding(mesh, spec))
                else:
                    out_sharding_list.append(replicated)
            else:
                out_sharding_list.append(replicated)
        out_shardings = out_treedef.unflatten(out_sharding_list)
        logger.info(
            "Compiling with GSPMD: %d placeholders, %d batch-sharded, %d weight-sharded, mesh=%s",
            len(placeholders),
            sum(is_batch),
            sum(is_fsdp_sharded),
            mesh.axis_names,
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
            jax_args = tuple(
                self._to_sharded_jax(
                    a,
                    mesh,
                    is_batch[j],
                    per_input_batch_dim[j],
                    is_fsdp_weight=is_fsdp_sharded[j],
                )
                for j, a in enumerate(run_args)
            )
            if not _hlo_captured[0]:
                _hlo_captured[0] = True
                compiled = jitted.lower(*jax_args).compile()
                run.hlo_text = compiled.as_text()  # pyre-ignore[16]
            jax_out = jitted(*jax_args)
            return self._postprocess_jax_output(jax_out, input_device)

        run.hlo_text = None  # pyre-ignore[16]

        example_args = tuple(
            self._to_sharded_jax(
                a,
                mesh,
                is_batch[j],
                per_input_batch_dim[j],
                is_fsdp_weight=is_fsdp_sharded[j],
            )
            for j, a in enumerate(args)
        )
        lowered = jitted.lower(*example_args)
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
        if not target.has_kernel_for_dispatch_key(torch._C.DispatchKey.Meta):
            return False, f"{target} doesn't have a meta kernel registered"

    # Override dispatch functions inserted by replace_override_anchors
    # are marked with xla_graph_break metadata — they must not be absorbed.
    if node.meta.get("xla_graph_break", False):
        return False, "graph break (override dispatch)"

    return True, ""


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


def _jax_to_torch(t: Any, device: torch.device) -> Any:
    """Convert a JAX output (array, None, or nested list/tuple) to a torch tensor.

    For batch-sharded arrays (shards differ), concatenates all addressable
    shards to reconstruct the per-process tensor.  For replicated arrays
    (all shards identical), takes a single shard.

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
    src_is_bf16 = hasattr(t, "dtype") and getattr(t.dtype, "name", "") == "bfloat16"
    if hasattr(t, "addressable_shards") and t.addressable_shards:
        shards = t.addressable_shards
        if len(shards) > 1 and shards[0].data.shape != t.shape:
            # Batch-sharded: each shard is a slice. Concatenate to
            # reconstruct the per-process tensor.
            sorted_shards = sorted(shards, key=lambda s: s.index)
            arrays = [_jax_to_numpy(s.data) for s in sorted_shards]
            np_arr = np.concatenate(arrays, axis=0)
        else:
            # Replicated: all shards identical, take one.
            np_arr = _jax_to_numpy(shards[0].data)
        if np_arr.ndim == 0:
            return torch.tensor(np_arr.item(), device=device)
        out = torch.tensor(np_arr, device=device)
        if src_is_bf16:
            out = out.to(torch.bfloat16)
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


def _strip_profiler_nodes(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Remove profiler ops from the graph.

    Profiler ops like ``_record_function_enter_new`` and
    ``_record_function_exit`` return ``torch.ScriptObject`` handles
    that JAX cannot represent.  They are no-ops at execution time,
    so stripping them is safe.

    Must run on the parent graph *before* segmentation so that the
    ScriptObject handles never become XLA subgraph inputs.

    Implementation note: profiler nodes form ``enter -> exit`` pairs,
    where ``_record_function_exit`` consumes the handle returned by
    ``_record_function_enter_new``. We iterate in REVERSE topological
    order so each ``_record_function_exit`` is erased before its
    matching ``_record_function_enter_new``. This matters for
    ``IterGraphModule``: its overridden ``erase_node`` cascades the
    erase into the mirrored setup/cleanup graphs without first
    calling ``replace_all_uses_with(None)`` on the mirror node, so
    erasing the enter first while its exit mirror still references
    it raises ``RuntimeError: Tried to erase Node ... but it still
    had 1 users in the graph``.
    """
    changed = False
    for node in reversed(list(gm.graph.nodes)):
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            if node.target.name().startswith("profiler::"):
                node.replace_all_uses_with(None)  # type: ignore[arg-type]
                gm.graph.erase_node(node)
                changed = True
    if changed:
        gm.graph.eliminate_dead_code()
        gm.recompile()
    return gm


def partial_lower_xla(
    gm: torch.fx.GraphModule,
    backend_type: BackendType = BackendType.XLA_SHARD_MAP,
    node_predicate: Callable[[torch.fx.Node], bool] = lambda x: True,
    subgraph_predicate: Callable[[List[torch.fx.Node]], bool] = lambda x: True,
    dumper: Callable[[str], str] = lambda x: "subgraph",
    op_swap_map: Optional[OpSwapRegistry] = None,
) -> torch.fx.GraphModule:
    if op_swap_map is not None:
        gm = swap_ops(gm, op_swap_map)
    # Strip profiler ops before segmentation so their ScriptObject
    # handles never become XLA subgraph inputs.
    gm = _strip_profiler_nodes(gm)
    lowered = segment_and_lower(
        gm,
        compatibility_fn=_is_xla_compatible,
        module_factory=lambda sub_gm: _XLAModule(sub_gm, backend_type),
        subgraph_prefix="xla_subgraph",
        node_predicate=node_predicate,
        subgraph_predicate=subgraph_predicate,
        dumper=dumper,
    )
    _annotate_safe_donate_indices(lowered)
    return lowered
