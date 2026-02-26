import torch
import inspect
import ast
import textwrap
from contextlib import contextmanager
from zrth import DType, Wire, Term, IType


def convert_method(
    method,
    wires: dict[str, tuple[Wire, Wire]],
    result: list[Wire],
    cls=None,
    layers: dict[str, int] | None = None,
    params: dict[str, Wire] | None = None,
) -> list[Term]:
    """Convert a Python method to a list of Terms.

    Args:
        method: Unbound method to convert (e.g. cls.reset, cls.step, cls.forward)
        wires: All named wire pairs available to the method - both method
               parameters (by parameter name) and self.* attributes (by attr name)
        result: Ordered list of next-wires matched positionally to the return tuple
        cls: Class owning the method, needed only for inlining self.helper() calls
        params: Read-only parameter wires (single wires, not pairs) for self.* constants

    Returns:
        List of Terms representing the method as a reactive diagram
    """
    source = textwrap.dedent(inspect.getsource(method))
    func_def = ast.parse(source).body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError(f"Expected function definition, got {type(func_def).__name__}")

    # Normalize early returns to single-exit form before visiting
    func_def.body = _normalize_early_returns(func_def.body)

    param_names = [arg.arg for arg in func_def.args.args if arg.arg != "self"]
    visitor = MethodVisitor(wires, result, cls=cls, layers=layers, params=params)
    visitor.temp_vars.update(
        {name: wires[name][0] for name in param_names if name in wires}
    )

    for stmt in func_def.body:
        visitor.visit(stmt)

    for i, result_wire in enumerate(result):
        src = visitor.temp_vars.get(f"_ret_{i}")
        if src is None:
            raise ValueError(f"Method has no return value for result {i}")
        visitor.terms.append(Term(IType.Id(), [result_wire], [src]))

    return visitor.terms


def _normalize_early_returns(stmts: list) -> list:
    """Convert early-return patterns to single-exit form.

    Transforms:
        if cond:
            ...
            return val      ← replaced with _ret_i = val assignments
        rest...
        return final
    Into:
        if cond:
            ...
            _ret_0 = val0
            _ret_1 = val1
        else:
            rest...
            return final    ← visit_Return stores _ret_i in temp_vars

    Both branches then set _ret_i, so visit_If SSA-merges them via Ite.
    Applied recursively so multiple consecutive early returns are all folded in.
    """
    result = []
    idx = 0
    while idx < len(stmts):
        stmt = stmts[idx]

        # Detect: if without else whose body ends with a return
        if (
            isinstance(stmt, ast.If)
            and not stmt.orelse
            and stmt.body
            and isinstance(stmt.body[-1], ast.Return)
        ):
            ret = stmt.body[-1]
            values = (
                ret.value.elts if isinstance(ret.value, ast.Tuple)
                else ([ret.value] if ret.value else [])
            )
            # Replace return with _ret_i = value assignments so SSA sees both branches
            ret_assigns = [
                ast.fix_missing_locations(ast.copy_location(
                    ast.Assign(
                        targets=[ast.Name(id=f"_ret_{i}", ctx=ast.Store())],
                        value=val,
                        type_comment=None,
                    ),
                    ret,
                ))
                for i, val in enumerate(values)
            ]
            body = _normalize_early_returns(stmt.body[:-1]) + ret_assigns
            rest = _normalize_early_returns(stmts[idx + 1:])
            result.append(ast.copy_location(
                ast.If(test=stmt.test, body=body or [ast.Pass()], orelse=rest),
                stmt,
            ))
            return result  # rest is consumed into the else; nothing left to process

        if isinstance(stmt, ast.If):
            stmt = ast.copy_location(
                ast.If(
                    test=stmt.test,
                    body=_normalize_early_returns(stmt.body),
                    orelse=_normalize_early_returns(stmt.orelse) if stmt.orelse else [],
                ),
                stmt,
            )

        result.append(stmt)
        idx += 1

    return result


# ============================================================================
# PyTorch Layer Helpers (used by NN conversion)
# ============================================================================


def _translate_linear(input_wire: Wire, out_features: int, terms: list[Term]):
    """Translate a linear layer to a single IType.Linear term.

    Creates: output = input @ weight + bias
    Weight and bias wires are dangling (no term writes them) — they are
    external parameters to be connected later.

    Args:
        input_wire: Input Wire
        out_features: Number of output features
        terms: List to append Terms to

    Returns:
        (output_wire, weight_wire, bias_wire)
    """
    in_features = input_wire.dtype().shape[-1]

    weight_wire = Wire(DType.Float([in_features, out_features]))
    bias_wire = Wire(DType.Float([out_features]))
    output_wire = Wire(DType.Float([out_features]))

    terms.append(Term(IType.Linear(), [output_wire], [input_wire, weight_wire, bias_wire]))

    return output_wire, weight_wire, bias_wire


def _translate_relu(input_wire: Wire, terms: list[Term]) -> Wire:
    """Translate ReLU activation to reactive operations

    Implements: max(0, x)

    Args:
        input_wire: Input Wire
        terms: List to append Terms to

    Returns:
        Output Wire
    """
    output_wire = Wire(input_wire.dtype())
    relu_term = Term(IType.ReLU(), [output_wire], [input_wire])
    terms.append(relu_term)

    return output_wire



class MethodVisitor(ast.NodeVisitor):
    """AST visitor to convert Python methods to reactive Terms

    Intermediate values are stored as Wire objects in temp_vars.
    """

    BINARY_OPS = {
        ast.Add: IType.Add,
        ast.Sub: IType.Sub,
        ast.Mult: IType.Mul,
        ast.Div: IType.Div,
    }

    COMPARE_OPS = {
        ast.Eq: IType.Eq,
        ast.NotEq: IType.Neq,
        ast.Lt: IType.Lt,
        ast.LtE: IType.Le,
        ast.Gt: IType.Gt,
        ast.GtE: IType.Ge,
    }

    def __init__(
        self,
        wire_pairs: dict[str, tuple[Wire, Wire]],
        result_wires: list[Wire],
        cls=None,
        layers: dict[str, int] | None = None,
        params: dict[str, Wire] | None = None,
    ):
        self.wire_pairs = wire_pairs
        self.result_wires = result_wires
        self.cls = cls
        self.layers = layers or {}
        self.params = params or {}
        self.terms = []
        self.temp_vars = {}
        self.scopes = []
        self.written_wires = set()

    @contextmanager
    def _scope(self, scope_name):
        """Push/pop scope for top-level write detection."""
        self.scopes.append(scope_name)
        try:
            yield
        finally:
            self.scopes.pop()

    def visit_If(self, node):
        """Handle if/else with SSA: evaluate both branches, merge with Ite."""
        cond_wire = self._convert_expr(node.test)

        parent_scope = dict(self.temp_vars)

        with self._scope("if"):
            for stmt in node.body:
                self.visit(stmt)
        if_scope_after = dict(self.temp_vars)

        self.temp_vars = dict(parent_scope)

        with self._scope("else"):
            if node.orelse:
                for stmt in node.orelse:
                    self.visit(stmt)
        else_scope_after = dict(self.temp_vars)

        # In a normalized early-return branch (contains _ret_*), plain locals are dead
        # after the return — only wire_pairs and _ret_* should escape.
        if any(k.startswith("_ret_") for k in if_scope_after):
            if_scope_after = {k: v for k, v in if_scope_after.items() if k in self.wire_pairs or k.startswith("_ret_")}
        if any(k.startswith("_ret_") for k in else_scope_after):
            else_scope_after = {k: v for k, v in else_scope_after.items() if k in self.wire_pairs or k.startswith("_ret_")}

        all_vars = set(if_scope_after.keys()) | set(else_scope_after.keys())

        for var in all_vars:
            if_wire = if_scope_after.get(var)
            if if_wire is None:
                if_wire = parent_scope.get(var)
            if if_wire is None and var in self.wire_pairs:
                if_wire = self.wire_pairs[var][0]

            else_wire = else_scope_after.get(var)
            if else_wire is None:
                else_wire = parent_scope.get(var)
            if else_wire is None and var in self.wire_pairs:
                else_wire = self.wire_pairs[var][0]

            if if_wire != else_wire and if_wire is not None and else_wire is not None:
                merged_wire = Wire(if_wire.dtype())
                self.terms.append(
                    Term(IType.Ite(), [merged_wire], [cond_wire, if_wire, else_wire])
                )
                self.temp_vars[var] = merged_wire

                # If this is a state wire and we're at top level, write the merged value to output
                if (
                    var in self.wire_pairs
                    and len(self.scopes) == 0
                    and var not in self.written_wires
                ):
                    output_wire = self.wire_pairs[var][1]
                    term = Term(IType.Id(), [output_wire], [merged_wire])
                    self.terms.append(term)
                    self.written_wires.add(var)
            elif if_wire is not None:
                self.temp_vars[var] = if_wire

                # Write only if top-level, not yet written, and newly assigned in this branch
                if (
                    var in self.wire_pairs
                    and len(self.scopes) == 0
                    and var not in parent_scope
                    and var not in self.written_wires
                ):
                    output_wire = self.wire_pairs[var][1]
                    term = Term(IType.Id(), [output_wire], [if_wire])
                    self.terms.append(term)
                    self.written_wires.add(var)

    def visit_Assign(self, node):
        """Handle variable assignment"""
        if len(node.targets) != 1:
            raise ValueError("Only single assignment supported")

        target = node.targets[0]

        if isinstance(target, ast.Name):
            var_name = target.id
            result_wire = self._convert_expr(node.value)
            self.temp_vars[var_name] = result_wire

        elif isinstance(target, ast.Attribute) and target.attr in self.wire_pairs:
            wire_name = target.attr
            target_dtype = self.wire_pairs[wire_name][1].dtype()
            result_wire = self._convert_expr(node.value, target_dtype=target_dtype)
            self.temp_vars[wire_name] = result_wire

            if len(self.scopes) == 0:
                output_wire = self.wire_pairs[wire_name][1]
                term = Term(IType.Id(), [output_wire], [result_wire])
                self.terms.append(term)
                self.written_wires.add(wire_name)

    def visit_AugAssign(self, node):
        """Handle augmented assignment (+=, -=, *=, /=)."""
        if (
            isinstance(node.target, ast.Attribute)
            and node.target.attr in self.wire_pairs
        ):
            wire_name = node.target.attr

            if wire_name in self.temp_vars:
                left_wire = self.temp_vars[wire_name]
            else:
                left_wire = self.wire_pairs[wire_name][0]

            target_dtype = self.wire_pairs[wire_name][1].dtype()
            right_wire = self._convert_expr(node.value, target_dtype=target_dtype)

            op_type = type(node.op)
            if op_type not in self.BINARY_OPS:
                raise ValueError(
                    f"Unsupported augmented assignment operator: {op_type.__name__}"
                )

            result_wire = Wire(left_wire.dtype())
            itype_cls = self.BINARY_OPS[op_type]
            self.terms.append(Term(itype_cls(), [result_wire], [left_wire, right_wire]))

            self.temp_vars[wire_name] = result_wire

            if len(self.scopes) == 0:
                output_wire = self.wire_pairs[wire_name][1]
                term = Term(IType.Id(), [output_wire], [result_wire])
                self.terms.append(term)
                self.written_wires.add(wire_name)
        else:
            raise ValueError(
                "Augmented assignment only supported for self.attribute wires"
            )

    def visit_Expr(self, node):
        """Silently skip bare expression statements (e.g. super().reset(seed=seed))"""
        pass

    def visit_Return(self, node):
        """Handle return statement: zip return values positionally with result_wires"""
        if node.value is None:
            return

        value_nodes = (
            node.value.elts
            if isinstance(node.value, ast.Tuple)
            else [node.value]
        )

        if len(value_nodes) != len(self.result_wires):
            raise ValueError(
                f"Return has {len(value_nodes)} value(s) but "
                f"{len(self.result_wires)} result wire(s) were declared"
            )

        for i, (result_wire, value_node) in enumerate(zip(self.result_wires, value_nodes)):
            src_wire = self._convert_expr(value_node, target_dtype=result_wire.dtype())
            self.temp_vars[f"_ret_{i}"] = src_wire

    def _convert_expr(self, expr, target_dtype=None):
        """Convert AST expression node to Wire object

        Args:
            expr: AST expression node
            target_dtype: Optional target dtype for type propagation (used for constants)
        """
        if isinstance(expr, ast.Call):
            return self._convert_call(expr, target_dtype=target_dtype)
        elif isinstance(expr, ast.BinOp):
            return self._convert_binop(expr, target_dtype=target_dtype)
        elif isinstance(expr, ast.UnaryOp):
            return self._convert_unaryop(expr, target_dtype=target_dtype)
        elif isinstance(expr, ast.BoolOp):
            return self._convert_boolop(expr)
        elif isinstance(expr, ast.Compare):
            return self._convert_compare(expr)
        elif isinstance(expr, ast.IfExp):
            return self._convert_ifexp(expr, target_dtype=target_dtype)
        elif isinstance(expr, ast.Attribute):
            return self._convert_attribute(expr)
        elif isinstance(expr, ast.Name):
            return self._convert_name(expr)
        elif isinstance(expr, ast.Constant):
            return self._convert_constant(expr, target_dtype)
        elif isinstance(expr, (ast.List, ast.Tuple)):
            return self._convert_list_literal(expr, target_dtype)
        else:
            raise ValueError(f"Unsupported expression type: {type(expr).__name__}")

    def _convert_call(self, call, target_dtype=None):
        """Convert method/function call"""
        if isinstance(call.func, ast.Attribute):
            method = call.func.attr
            obj = call.func.value

            # Handle module/name-based calls first (np.*, self.*)
            # TODO: This only handles NumPy common aliases. For arbitrary aliases (import numpy as mynp),
            # would need to track imports and resolve module names.
            if isinstance(obj, ast.Name):
                name = obj.id
                if name in ("np", "numpy"):
                    return self._convert_numpy_creation(
                        method, call.args, target_dtype=target_dtype
                    )
                elif name == "torch":
                    if method == "relu":
                        input_wire = self._convert_expr(call.args[0], target_dtype=target_dtype)
                        return _translate_relu(input_wire, self.terms)
                    else:
                        raise ValueError(f"Unsupported torch function: torch.{method}")
                elif name == "self":
                    return self._inline_method(method, call.args)

            if method == "argmax":
                obj_wire = self._convert_expr(obj)
                result = Wire(DType.Float([1]))
                self.terms.append(Term(IType.Argmax(), [result], [obj_wire]))
                return result
            elif method == "item":
                return self._convert_expr(obj)
            else:
                raise ValueError(f"Unsupported method: {method}")

        elif isinstance(call.func, ast.Name):
            func_name = call.func.id

            if func_name in ("min", "max"):
                return self._convert_minmax(call.args, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")
        else:
            raise ValueError(f"Unsupported call type: {type(call.func).__name__}")

    def _convert_minmax(self, args, func_name):
        """Convert min/max to conditional: min(a,b) -> Ite(Lt(a,b), a, b)"""
        if len(args) != 2:
            raise ValueError(f"{func_name}() requires 2 arguments")

        a_wire = self._convert_expr(args[0])
        b_wire = self._convert_expr(args[1])

        cmp_type = IType.Lt() if func_name == "min" else IType.Gt()
        cmp_wire = Wire(DType.Bool())
        self.terms.append(Term(cmp_type, [cmp_wire], [a_wire, b_wire]))

        result = Wire(a_wire.dtype())
        self.terms.append(Term(IType.Ite(), [result], [cmp_wire, a_wire, b_wire]))
        return result

    def _convert_numpy_creation(self, func_name, args, target_dtype=None):
        """Convert NumPy array creation to torch equivalent.

        Supports compile-time conversion of:
        - np.zeros(shape) -> Wire with Tensor term
        - np.ones(shape) -> Wire with Tensor term
        - np.array(data) -> Wire with Tensor term

        Shape and data must be Python literals (evaluated at conversion time).
        Respects target_dtype for creating Int/Float/Bool arrays.
        """
        if func_name == "zeros":
            if len(args) != 1:
                raise ValueError(f"np.zeros() requires 1 argument, got {len(args)}")
            shape = self._eval_shape(args[0])

            if target_dtype and target_dtype.kind() == "Int":
                tensor_data = torch.zeros(*shape, dtype=torch.long)
            elif target_dtype and target_dtype.kind() == "Bool":
                tensor_data = torch.zeros(*shape, dtype=torch.bool)
            else:
                tensor_data = torch.zeros(*shape, dtype=torch.float32)

        elif func_name == "ones":
            if len(args) != 1:
                raise ValueError(f"np.ones() requires 1 argument, got {len(args)}")
            shape = self._eval_shape(args[0])

            if target_dtype and target_dtype.kind() == "Int":
                tensor_data = torch.ones(*shape, dtype=torch.long)
            elif target_dtype and target_dtype.kind() == "Bool":
                tensor_data = torch.ones(*shape, dtype=torch.bool)
            else:
                tensor_data = torch.ones(*shape, dtype=torch.float32)

        elif func_name == "array":
            if len(args) != 1:
                raise ValueError(f"np.array() requires 1 argument, got {len(args)}")
            data = self._eval_literal(args[0])

            if target_dtype and target_dtype.kind() == "Int":
                tensor_data = torch.tensor(data, dtype=torch.long)
            elif target_dtype and target_dtype.kind() == "Bool":
                tensor_data = torch.tensor(data, dtype=torch.bool)
            else:
                tensor_data = torch.tensor(data, dtype=torch.float32)

        else:
            raise ValueError(f"Unsupported NumPy function: np.{func_name}()")

        shape = list(tensor_data.size())
        if target_dtype:
            dtype = target_dtype.reshape(shape)
        else:
            dtype = DType.Float(shape)

        const_wire = Wire(dtype)
        self.terms.append(Term(IType.Tensor(tensor_data), [const_wire]))
        return const_wire

    def _eval_shape(self, shape_node):
        """Evaluate shape argument at compile time.

        Accepts:
        - Single int: np.zeros(5) -> (5,)
        - Tuple: np.zeros((3, 3)) -> (3, 3)
        - List: np.zeros([10]) -> (10,)

        Returns:
            Tuple of ints representing the shape
        """
        if isinstance(shape_node, ast.Constant):
            return (shape_node.value,)
        elif isinstance(shape_node, (ast.Tuple, ast.List)):
            shape = []
            for elt in shape_node.elts:
                if not isinstance(elt, ast.Constant):
                    raise ValueError(
                        f"Shape must be constant, got {type(elt).__name__}"
                    )
                shape.append(elt.value)
            return tuple(shape)
        else:
            raise ValueError(
                f"Shape must be int or tuple, got {type(shape_node).__name__}"
            )

    def _eval_literal(self, data_node):
        """Recursively evaluate Python literal for np.array() data.

        Supports nested lists/tuples of constants only.
        Examples: [1, 2, 3], [[1, 2], [3, 4]], (1.0, 2.0, 3.0)

        Returns:
            Python list/scalar that can be passed to torch.tensor()
        """
        if isinstance(data_node, ast.Constant):
            return data_node.value
        elif isinstance(data_node, (ast.List, ast.Tuple)):
            return [self._eval_literal(elt) for elt in data_node.elts]
        else:
            raise ValueError(
                f"np.array() data must be literal, got {type(data_node).__name__}"
            )

    def _convert_binop(self, binop, target_dtype=None):
        """Convert binary operation

        Dtype propagation:
        - With target_dtype: both operands inherit it (e.g., self.x = 2 + 3 → both Int)
        - Without: right operand inherits from left (e.g., self.x + 3 → 3 matches x's dtype)
        """
        if target_dtype:
            left_wire = self._convert_expr(binop.left, target_dtype=target_dtype)
            right_wire = self._convert_expr(binop.right, target_dtype=target_dtype)
            result_dtype = target_dtype
        else:
            left_wire = self._convert_expr(binop.left)
            right_wire = self._convert_expr(binop.right, target_dtype=left_wire.dtype())
            result_dtype = left_wire.dtype()

        op_type = type(binop.op)
        if op_type not in self.BINARY_OPS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

        result = Wire(result_dtype)
        itype_cls = self.BINARY_OPS[op_type]
        self.terms.append(Term(itype_cls(), [result], [left_wire, right_wire]))
        return result

    def _convert_unaryop(self, unaryop, target_dtype=None):
        """Convert unary operation (not, -, +)

        - not x: Always returns Bool
        - -x: Propagates target_dtype to operand and zero constant
        - +x: No-op, returns operand with target_dtype propagated
        """
        op_type = type(unaryop.op)
        if op_type == ast.Not:
            # not x -> Ite(x, False, True) - always Bool
            operand_wire = self._convert_expr(unaryop.operand)
            false_wire = self._convert_constant(ast.Constant(False))
            true_wire = self._convert_constant(ast.Constant(True))
            result = Wire(DType.Bool())
            self.terms.append(
                Term(IType.Ite(), [result], [operand_wire, false_wire, true_wire])
            )
            return result
        elif op_type == ast.USub:
            # -x -> 0 - x
            if target_dtype:
                operand_wire = self._convert_expr(
                    unaryop.operand, target_dtype=target_dtype
                )
                zero_wire = self._convert_constant(
                    ast.Constant(0), target_dtype=target_dtype
                )
                result_dtype = target_dtype
            else:
                operand_wire = self._convert_expr(unaryop.operand)
                zero_wire = self._convert_constant(
                    ast.Constant(0), target_dtype=operand_wire.dtype()
                )
                result_dtype = operand_wire.dtype()
            result = Wire(result_dtype)
            self.terms.append(Term(IType.Sub(), [result], [zero_wire, operand_wire]))
            return result
        elif op_type == ast.UAdd:
            return self._convert_expr(unaryop.operand, target_dtype=target_dtype)
        else:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

    def _convert_boolop(self, boolop):
        """Convert boolean operation (and/or) to nested Ite expressions

        Python: a and b -> Ite(a, b, False)
        Python: a or b  -> Ite(a, True, b)
        """
        if len(boolop.values) < 2:
            raise ValueError("BoolOp must have at least 2 operands")

        wires = [self._convert_expr(val) for val in boolop.values]
        is_and = isinstance(boolop.op, ast.And)

        if not is_and and not isinstance(boolop.op, ast.Or):
            raise ValueError(
                f"Unsupported boolean operator: {type(boolop.op).__name__}"
            )

        # Build nested Ite from right to left
        result = wires[-1]
        for wire in reversed(wires[:-1]):
            false_wire = self._convert_constant(ast.Constant(False))
            true_wire = self._convert_constant(ast.Constant(True))

            merged = Wire(DType.Bool())
            if is_and:
                # a and b -> Ite(a, b, False)
                self.terms.append(
                    Term(IType.Ite(), [merged], [wire, result, false_wire])
                )
            else:
                # a or b -> Ite(a, True, b)
                self.terms.append(
                    Term(IType.Ite(), [merged], [wire, true_wire, result])
                )
            result = merged
        return result

    def _convert_compare(self, compare):
        """Convert comparison operation

        Handles both simple comparisons (a < b) and chains (a < b < c).
        Chains are expanded: a < b < c becomes (a < b) and (b < c)
        """
        comparison_wires = []
        left = compare.left

        for op, comparator in zip(compare.ops, compare.comparators):
            left_wire = self._convert_expr(left)
            right_wire = self._convert_expr(comparator, target_dtype=left_wire.dtype())

            op_type = type(op)
            if op_type not in self.COMPARE_OPS:
                raise ValueError(f"Unsupported comparison operator: {op_type.__name__}")

            cmp_wire = Wire(DType.Bool())
            itype_cls = self.COMPARE_OPS[op_type]
            self.terms.append(Term(itype_cls(), [cmp_wire], [left_wire, right_wire]))
            comparison_wires.append(cmp_wire)
            left = comparator

        # Combine with AND (single comparison returns directly)
        result = comparison_wires[0]
        for comp_wire in comparison_wires[1:]:
            false_wire = self._convert_constant(ast.Constant(False))
            merged = Wire(DType.Bool())
            self.terms.append(
                Term(IType.Ite(), [merged], [result, comp_wire, false_wire])
            )
            result = merged

        return result

    def _convert_ifexp(self, ifexp, target_dtype=None):
        """Convert ternary conditional to Ite

        Propagates target_dtype to both branches (e.g., self.x = 5 if c else 10 → both Int)
        """
        cond_wire = self._convert_expr(ifexp.test)
        true_wire = self._convert_expr(ifexp.body, target_dtype=target_dtype)
        false_wire = self._convert_expr(ifexp.orelse, target_dtype=target_dtype)

        result_dtype = target_dtype if target_dtype else true_wire.dtype()
        result = Wire(result_dtype)
        self.terms.append(
            Term(IType.Ite(), [result], [cond_wire, true_wire, false_wire])
        )
        return result

    def _convert_name(self, name):
        """Convert variable reference"""
        var_name = name.id
        if var_name in self.temp_vars:
            return self.temp_vars[var_name]
        else:
            raise ValueError(f"Unknown variable: {var_name}")

    def _convert_constant(self, constant, target_dtype=None):
        """Convert constant literal to Wire with Tensor term

        Args:
            constant: AST Constant node
            target_dtype: Optional dtype (inferred from context). Defaults to Float if None.
        """
        value = constant.value

        if isinstance(value, bool):
            tensor_data = torch.tensor([value])
            dtype = DType.Bool([1])

        elif isinstance(value, (int, float)):
            if target_dtype is None:
                # TODO: Consider inferring dtype from value (int vs float) or raising error instead of defaulting
                target_dtype = DType.Float([])
                tensor_data = torch.tensor([float(value)], dtype=torch.float32)
            elif target_dtype.kind() == "TensorInt":
                tensor_data = torch.tensor([int(value)], dtype=torch.long)
            elif target_dtype.kind() == "TensorFloat":
                tensor_data = torch.tensor([float(value)], dtype=torch.float32)
            elif target_dtype.kind() == "TensorBool":
                tensor_data = torch.tensor([bool(value)])
            else:
                raise ValueError(
                    f"Unsupported target dtype kind: {target_dtype.kind()}"
                )

            dtype = target_dtype.reshape([1])

        elif isinstance(value, (list, tuple, torch.Tensor)):
            if isinstance(value, torch.Tensor):
                tensor_data = value
            else:
                if target_dtype and target_dtype.kind() == "Int":
                    tensor_data = torch.tensor(value, dtype=torch.long)
                elif target_dtype and target_dtype.kind() == "Bool":
                    tensor_data = torch.tensor(value, dtype=torch.bool)
                else:
                    tensor_data = torch.tensor(value, dtype=torch.float32)

            if target_dtype:
                dtype = target_dtype.reshape(list(tensor_data.size()))
            else:
                dtype = DType.Float(list(tensor_data.size()))

        else:
            raise ValueError(f"Unsupported constant type: {type(value)}")

        const_wire = Wire(dtype)
        self.terms.append(Term(IType.Tensor(tensor_data), [const_wire]))
        return const_wire

    def _convert_list_literal(self, node, target_dtype=None):
        """Convert a list/tuple literal to a Wire with Tensor term.

        Evaluates the nested literal at compile time and creates a constant tensor.
        """
        data = self._eval_literal(node)

        if target_dtype and target_dtype.kind() == "Int":
            tensor_data = torch.tensor(data, dtype=torch.long)
        elif target_dtype and target_dtype.kind() == "Bool":
            tensor_data = torch.tensor(data, dtype=torch.bool)
        else:
            tensor_data = torch.tensor(data, dtype=torch.float32)

        shape = list(tensor_data.size())
        if target_dtype:
            dtype = target_dtype.reshape(shape)
        else:
            dtype = DType.Float(shape)

        const_wire = Wire(dtype)
        self.terms.append(Term(IType.Tensor(tensor_data), [const_wire]))
        return const_wire

    def _convert_attribute(self, attr):
        """Convert self.attr: returns locally-assigned wire if available, else the input wire."""
        if isinstance(attr.value, ast.Name) and attr.value.id == "self":
            wire_name = attr.attr

            if wire_name in self.temp_vars:
                return self.temp_vars[wire_name]

            if wire_name in self.wire_pairs:
                return self.wire_pairs[wire_name][0]

            if wire_name in self.params:
                return self.params[wire_name]

            raise ValueError(f"Unknown wire: {wire_name}")
        else:
            raise ValueError(f"Unsupported attribute access: {ast.unparse(attr)}")

    def _inline_method(self, method_name, args):
        """Inline simple method or dispatch to layer translator."""
        if method_name in self.layers:
            input_wire = self._convert_expr(args[0])
            out_features = self.layers[method_name]
            output_wire, _, _ = _translate_linear(input_wire, out_features, self.terms)
            return output_wire

        if self.cls is None or not hasattr(self.cls, method_name):
            raise ValueError(f"Method not found: {method_name}")

        method = getattr(self.cls, method_name)
        source = textwrap.dedent(inspect.getsource(method))
        method_def = ast.parse(source).body[0]
        if not isinstance(method_def, ast.FunctionDef):
            raise ValueError(f"Expected function definition for {method_name}")

        return_stmt = next(
            (stmt for stmt in method_def.body if isinstance(stmt, ast.Return)), None
        )
        if return_stmt is None or return_stmt.value is None:
            raise ValueError(f"Method {method_name} has no return value")

        return self._convert_expr(return_stmt.value)
