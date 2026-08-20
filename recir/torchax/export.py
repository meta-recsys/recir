# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# pylint: disable
"""Utilities for exporting a torch program to jax/stablehlo."""

import copy
import logging
from typing import Any

import jax
import jax.export
import recir.torchax
import sympy
import torch
import torch._refs
from recir.torchax import decompositions
from recir.torchax.ops import mappings, ops_registry
from torch._decomp import get_decompositions
from torch.utils import _pytree as pytree

logger: logging.Logger = logging.getLogger(__name__)


class JaxInterpreter(torch.fx.Interpreter):
    """Experimental."""

    def __init__(self, graph_module: Any) -> None:
        super().__init__(graph_module)

    def call_function(
        self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if not isinstance(target, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)):
            return super().call_function(target, args, kwargs)

        logger.debug("Running %s", target.name())

        op = ops_registry.all_aten_ops.get(target)
        if op is None:
            op = ops_registry.all_aten_ops.get(target.overloadpacket)
        assert op is not None, target
        assert op.is_jax_function, op
        if op is None:
            op = ops_registry.all_aten_ops.get(target.overloadpacket)
        if op is None:
            print(target.name(), target.tags)
            raise RuntimeError("No lowering found for", target.name())
        return op.func(*args, **kwargs)

    def run_node(self, n: Any) -> Any:
        res = super().run_node(n)
        if n.op == "call_function" and hasattr(res, "shape"):
            logger.debug("Meta: %s REAL: %s", n.meta.get("val").shape, res.shape)
        return res


_extra_decomp: dict[Any, Any] = get_decompositions([torch.ops.aten.unfold])


def _extract_states_from_exported_program(
    exported_model: Any,
) -> tuple[list[str], list[Any]]:
    # NOTE call convention: (parameters, buffers, user_inputs)
    param_and_buffer_keys = (
        exported_model.graph_signature.parameters
        + exported_model.graph_signature.buffers
    )
    state_dict = copy.copy(exported_model.state_dict)
    if (constants := getattr(exported_model, "constants", None)) is not None:
        state_dict.update(constants)
    param_buffer_values = [state_dict[key] for key in param_and_buffer_keys]

    if hasattr(exported_model.graph_signature, "lifted_tensor_constants"):
        for name in exported_model.graph_signature.lifted_tensor_constants:
            param_buffer_values.append(exported_model.tensor_constants[name])

    return param_and_buffer_keys, param_buffer_values


def exported_program_to_jax(exported_program: Any, export_raw: bool = False) -> Any:
    """returns a pytree of jax arrays(state), and

    a callable(func) that is jax function.

    func(state, input) would be how you call it.
    """
    if torch.__version__ >= "2.2":
        # torch version 2.1 didn't expose this yet
        exported_program = exported_program.run_decompositions()
        exported_program = exported_program.run_decompositions(
            decompositions.DECOMPOSITIONS
        )
    logger.debug("%s", exported_program.graph_module.code)

    names, states = _extract_states_from_exported_program(exported_program)

    def _extract_args(args: Any, kwargs: Any) -> list[Any]:
        flat_args, received_spec = pytree.tree_flatten((args, kwargs))  # type: ignore[possibly-undefined]
        return flat_args

    num_mutations: int = len(exported_program.graph_signature.buffers_to_mutate)

    def func(states: Any, inputs: Any) -> Any:
        args = _extract_args(inputs, {})
        res = JaxInterpreter(exported_program.graph_module).run(
            *states,
            *args,
            enable_io_processing=False,
        )
        res = res[num_mutations:]
        return res

    if export_raw:
        return names, states, func
    env = recir.torchax.default_env()
    states = env.t2j_copy(states)
    return states, func


def extract_avals(exported: Any) -> list[Any]:
    """Return JAX Abstract Value shapes for all input parameters of the exported
    program. This supports dynamic batch dimensions, including with constraints.
    """

    def _to_aval(
        arg_meta: dict[str, Any], symbolic_shapes: dict[str, Any] | None
    ) -> Any:
        """Convet from torch type to jax abstract value for export tracing"""

        def _get_dim(d: Any) -> Any:
            if isinstance(d, torch.SymInt):
                return symbolic_shapes[str(d)]  # pyre-ignore[16]
            return d

        val = arg_meta["val"]
        is_scalar = (
            isinstance(val, float) or isinstance(val, int) or isinstance(val, bool)
        )
        if is_scalar:
            return jax.ShapeDtypeStruct([], type(arg_meta["val"]))

        tensor_meta = arg_meta["tensor_meta"]
        shape = [_get_dim(d) for d in tensor_meta.shape]
        return jax.ShapeDtypeStruct(shape, mappings.t2j_dtype(tensor_meta.dtype))

    def _get_inputs(exported: Any) -> list[Any]:
        """Return placeholders with input metadata"""
        placeholders = [p for p in exported.graph.nodes if p.op == "placeholder"]
        input_placeholders = [
            p
            for p, s in zip(
                placeholders, exported.graph_signature.input_specs, strict=False
            )
            if s.kind == torch.export.graph_signature.InputKind.USER_INPUT
        ]
        return input_placeholders

    def _build_symbolic_shapes(
        range_constraints: dict[Any, Any],
    ) -> dict[str, Any] | None:
        """Convert torch SymInt to JAX symbolic_shape and stores in a map using the
        string name of the torch symbolic int.

        TODO: There is probably a better way of storing a key for a symbolic int.
        This value needs to be looked up again in `_to_aval` to figure out which
        JAX symbolic to map to for a given torch tensor.
        """
        if len(range_constraints) == 0:
            return None

        def _build_symbolic_constraints(
            symbol_name: str, torch_constraint: Any
        ) -> tuple[str, ...]:
            """Convert torch SymInt constraints to string for JAX symbolic_shape
            Using sympy may be overkill here, currently PyTorch only uses ValueRanges
            which allow specifying the min and the max of a value, for example:
              torch.export.Dim("a", min=5, max=10)
               ==> ("a >= 5", "a <= 10",)
            """
            if (
                not isinstance(
                    torch_constraint, torch.utils._sympy.value_ranges.ValueRanges
                )
                or torch_constraint.is_bool
            ):
                raise TypeError(
                    f"No symbolic constraint handler for: {torch_constraint}"
                )

            constraints = []
            symbol = sympy.Symbol(symbol_name)
            if torch_constraint.lower != 2:
                constraints.append(symbol >= torch_constraint.lower)
            from sympy.core.singleton import S

            if (
                not torch_constraint.upper.is_infinite
                and torch_constraint.upper is not S.IntInfinity
            ):
                constraints.append(symbol <= torch_constraint.upper)

            return tuple(sympy.pretty(c, use_unicode=False) for c in constraints)

        def _build_symbolic_shape(
            sym: Any, constraint: Any, free_symbols: dict[str, Any]
        ) -> Any:
            """Returns a JAX symbolic shape for a given symbol and constraint

            There are two possible sympy `sym` inputs:
              1. Symbol - (s0) These can have custom constraints.
              2. Expr - (s0*2) These apply the expr to s0's constraints, cannot override.

              Currently support is limited to operations with a symbol and and int,
              in `torch/export/dynamic_shapes.py`:
              "Only increasing linear operations with integer coefficients are supported."
            """
            symbol_name = str(sym)
            constraints = _build_symbolic_constraints(symbol_name, constraint)
            if sym.is_symbol:
                symbolic_shape = jax.export.symbolic_shape(
                    symbol_name, constraints=constraints
                )
            else:
                assert len(sym.free_symbols) > 0
                scope = free_symbols[str(list(sym.free_symbols)[0])].scope
                symbolic_shape = jax.export.symbolic_shape(symbol_name, scope=scope)
            assert len(symbolic_shape) == 1
            return symbolic_shape[0]

        # Populate symbol variables before expressions, exprs need to use the same
        # Symbolic scope as the variable they operate on. Expressions can only be
        # integer compuations on symbol variables, so each symbol variable is OK to
        # have its own scope.
        symbolic_shapes = {}
        symbol_variables = [(s, v) for s, v in range_constraints.items() if s.is_symbol]
        symbol_exprs = [(s, v) for s, v in range_constraints.items() if not s.is_symbol]
        for sym, constraint in symbol_variables + symbol_exprs:
            symbolic_shape = _build_symbolic_shape(sym, constraint, symbolic_shapes)
            symbolic_shapes[str(sym)] = symbolic_shape
        return symbolic_shapes

    symbolic_shapes = _build_symbolic_shapes(exported.range_constraints)
    args = _get_inputs(exported)

    logger.debug("Inputs to aval: %s", args)
    logger.debug("Symbolic shapes: %s", symbolic_shapes)
    for arg in args:
        logger.debug(
            "Meta2Aval %s --> %s",
            arg.meta,
            _to_aval(arg.meta, symbolic_shapes),
        )

    return [_to_aval(arg.meta, symbolic_shapes) for arg in args]


def exported_program_to_stablehlo(exported_program: Any) -> tuple[Any, Any]:
    """Replacement for torch_xla.stablehlo.exported_program_to_stablehlo

    Convert a program exported via torch.export to StableHLO.

    This supports dynamic dimension sizes and generates explicit checks for
    dynamo guards in the IR using shape_assertion custom_call ops.
    """
    weights, func = exported_program_to_jax(exported_program)
    jax_avals = extract_avals(exported_program)
    jax_export = jax.export.export(jax.jit(func))(weights, (jax_avals,))
    return weights, jax_export
