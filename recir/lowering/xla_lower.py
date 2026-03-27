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

    def _postprocess_jax_output(
        self, jax_out: Any, input_device: torch.device
    ) -> List[torch.Tensor]:
        if self.keep_jax_output:
            if isinstance(jax_out, (list, tuple)):
                return list(jax_out)
            return [jax_out]
        return _jax_to_torch(jax_out, input_device)

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

    @staticmethod
    def _strip_profiler_nodes(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Remove profiler ops from the graph.

        Profiler ops like ``_record_function_enter_new`` and
        ``_record_function_exit`` return ``torch.ScriptObject`` handles
        that JAX cannot represent.  They are no-ops at execution time,
        so stripping them is safe.
        """
        changed = False
        for node in list(gm.graph.nodes):
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
        self.gm = self._strip_profiler_nodes(self.gm)
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
        tensor: Any, mesh: Any, is_batch_input: bool = False, input_batch_dim: int = 0
    ) -> Any:
        """Convert a PyTorch tensor to a properly sharded JAX array."""
        if not isinstance(tensor, torch.Tensor):
            assert isinstance(tensor, jax.Array), (
                f"Expected torch.Tensor or jax.Array, got {type(tensor)}"
            )
            return tensor
        t = tensor.detach()
        if t.dtype == torch.bool:
            t = t.to(torch.int8)
        nparray = t.cpu().numpy()
        jax_array = jnp.asarray(nparray)
        if tensor.dtype == torch.bool:
            jax_array = jax_array.astype(jnp.bool_)

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
            return jax.make_array_from_single_device_arrays(
                tuple(global_shape), NamedSharding(mesh, P("torch_dist")), local_buffers
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

    def _jit_plain(
        self,
        jax_fn: Callable[..., Any],
        args: Tuple[torch.Tensor, ...],
        input_device: torch.device,
        jax_device: Any,
    ) -> Any:
        """Plain jax.jit on a single device (no collectives)."""
        example_jax_args = tuple(jax.device_put(t2j(a), jax_device) for a in args)
        lowered = jax.jit(jax_fn).lower(*example_jax_args)
        compiled: Any = lowered.compile()

        def run(*run_args: torch.Tensor) -> List[torch.Tensor]:
            jax_args = tuple(
                a
                if not isinstance(a, torch.Tensor)
                else jax.device_put(t2j(a), jax_device)
                for a in run_args
            )
            jax_out = compiled(*jax_args)
            return self._postprocess_jax_output(jax_out, input_device)

        return lowered, run

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
        batch_sharded = NamedSharding(mesh, P("torch_dist"))
        placeholders = [n for n in self.gm.graph.nodes if n.op == "placeholder"]
        is_batch: List[bool] = [
            n.meta.get("is_batch_input", False) for n in placeholders
        ]
        input_batch_dim: int = 0
        for n in placeholders:
            if n.meta.get("is_batch_input", False):
                input_batch_dim = n.meta.get("batch_dim", 0)
                break
        in_shardings = [batch_sharded if b else replicated for b in is_batch]
        out_node = next(n for n in self.gm.graph.nodes if n.op == "output")
        out_shardings = jax.tree.map(lambda _: replicated, out_node.args[0])
        logger.info(
            "Compiling with GSPMD: %d inputs (%d batch-sharded), mesh=%s",
            len(args),
            sum(is_batch),
            mesh.axis_names,
        )
        jitted: Any = jax.jit(
            jax_fn, in_shardings=in_shardings, out_shardings=out_shardings
        )

        _hlo_captured: List[bool] = [False]

        def run(*run_args: torch.Tensor) -> List[torch.Tensor]:
            jax_args = tuple(
                self._to_sharded_jax(a, mesh, is_batch[j], input_batch_dim)
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
            self._to_sharded_jax(a, mesh, is_batch[j], input_batch_dim)
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
    if node.target == torch.ops.aten.flatten.using_ints:
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

    return True, ""


def _jax_to_torch(t: Any, device: torch.device) -> Any:
    """Convert a JAX output (array, None, or nested list/tuple) to a torch tensor."""
    if t is None:
        return None
    if isinstance(t, list):
        return [_jax_to_torch(x, device) for x in t]
    if isinstance(t, tuple):
        return tuple(_jax_to_torch(x, device) for x in t)
    if hasattr(t, "addressable_shards") and t.addressable_shards:
        return torch.tensor(t.addressable_shards[0].data.__array__(), device=device)
    return torch.tensor(t.__array__(), device=device)


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
    return segment_and_lower(
        gm,
        compatibility_fn=_is_xla_compatible,
        module_factory=lambda sub_gm: _XLAModule(sub_gm, backend_type),
        subgraph_prefix="xla_subgraph",
        node_predicate=node_predicate,
        subgraph_predicate=subgraph_predicate,
        dumper=dumper,
    )
