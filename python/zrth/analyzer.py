"""
Abstract Interpreter for Python functions.
- No loops, no recursion.
- Path-sensitive: explores every branch.
- Tracks reads, writes, types, and values of variables and attributes.
- Unknown function calls produce symbolic return values.
"""

from __future__ import annotations

import ast
import textwrap
import inspect
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Set, Tuple, Union, Callable


# ---------------------------------------------------------------------------
# Abstract Values
# ---------------------------------------------------------------------------


class ValueKind(Enum):
    CONST = auto()  # Known constant value
    TYPED = auto()  # Known type, unknown value
    CALL_RESULT = auto()  # Return value of an unknown function call
    TOP = auto()  # Completely unknown
    BOTTOM = auto()  # Unreachable / no value


@dataclass(frozen=True)
class AbstractValue:
    kind: ValueKind
    value: Any = None  # For CONST: the Python value
    type_: type | None = None  # For CONST or TYPED: the Python type
    call_repr: str | None = None  # For CALL_RESULT: e.g. "foo(x, 3)"

    @staticmethod
    def const(v: Any) -> AbstractValue:
        return AbstractValue(kind=ValueKind.CONST, value=v, type_=type(v))

    @staticmethod
    def typed(t: type) -> AbstractValue:
        return AbstractValue(kind=ValueKind.TYPED, type_=t)

    @staticmethod
    def call_result(call_repr: str, result_type: type | None = None) -> AbstractValue:
        return AbstractValue(
            kind=ValueKind.CALL_RESULT, call_repr=call_repr, type_=result_type
        )

    @staticmethod
    def top() -> AbstractValue:
        return AbstractValue(kind=ValueKind.TOP)

    @staticmethod
    def bottom() -> AbstractValue:
        return AbstractValue(kind=ValueKind.BOTTOM)

    def is_const(self) -> bool:
        return self.kind == ValueKind.CONST

    def join(self, other: AbstractValue) -> AbstractValue:
        """Lattice join (merge two values from different paths)."""
        if self.kind == ValueKind.BOTTOM:
            return other
        if other.kind == ValueKind.BOTTOM:
            return self
        if self == other:
            return self
        # Both CONST with same type but different values -> TYPED
        if (
            self.kind == ValueKind.CONST
            and other.kind == ValueKind.CONST
            and self.type_ == other.type_
        ):
            assert self.type_ is not None
            return AbstractValue.typed(self.type_)
        # Same type from CONST/TYPED mix
        if self.type_ is not None and self.type_ == other.type_:
            return AbstractValue.typed(self.type_)
        # Otherwise -> TOP
        return AbstractValue.top()

    def __repr__(self) -> str:
        match self.kind:
            case ValueKind.CONST:
                assert self.type_ is not None
                return f"Const({self.value!r}: {self.type_.__name__})"
            case ValueKind.TYPED:
                assert self.type_ is not None
                return f"Typed({self.type_.__name__})"
            case ValueKind.CALL_RESULT:
                t = f": {self.type_.__name__}" if self.type_ else ""
                return f"CallResult({self.call_repr}{t})"
            case ValueKind.TOP:
                return "Top"
            case ValueKind.BOTTOM:
                return "Bottom"


# ---------------------------------------------------------------------------
# Access Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccessRecord:
    name: str  # Variable or "obj.attr"
    lineno: int
    col_offset: int


@dataclass
class AbstractState:
    """State at a single program point on a single path."""

    env: Dict[str, AbstractValue] = field(default_factory=dict)
    # obj_name -> attr_name -> AbstractValue
    attrs: Dict[str, Dict[str, AbstractValue]] = field(default_factory=dict)
    reads: List[AccessRecord] = field(default_factory=list)
    writes: List[AccessRecord] = field(default_factory=list)
    returned: AbstractValue | None = None  # None means not yet returned

    def copy(self) -> AbstractState:
        return AbstractState(
            env=dict(self.env),
            attrs={k: dict(v) for k, v in self.attrs.items()},
            reads=list(self.reads),
            writes=list(self.writes),
            returned=self.returned,
        )

    def write_var(self, name: str, val: AbstractValue, node: ast.AST) -> None:
        self.env[name] = val
        self.writes.append(AccessRecord(name, node.lineno, node.col_offset))

    def read_var(self, name: str, node: ast.AST) -> AbstractValue:
        self.reads.append(AccessRecord(name, node.lineno, node.col_offset))
        return self.env.get(name, AbstractValue.top())

    def write_attr(
        self, obj_name: str, attr: str, val: AbstractValue, node: ast.AST
    ) -> None:
        self.attrs.setdefault(obj_name, {})[attr] = val
        full = f"{obj_name}.{attr}"
        self.writes.append(AccessRecord(full, node.lineno, node.col_offset))

    def read_attr(self, obj_name: str, attr: str, node: ast.AST) -> AbstractValue:
        full = f"{obj_name}.{attr}"
        self.reads.append(AccessRecord(full, node.lineno, node.col_offset))
        obj_attrs = self.attrs.get(obj_name, {})
        return obj_attrs.get(attr, AbstractValue.top())


def join_states(states: List[AbstractState]) -> AbstractState:
    """Join multiple path states into one (merge at control-flow join point)."""
    if not states:
        s = AbstractState()
        s.returned = AbstractValue.bottom()
        return s
    if len(states) == 1:
        return states[0]

    # Merge environments
    all_keys: Set[str] = set()
    for s in states:
        all_keys |= s.env.keys()
    merged_env: Dict[str, AbstractValue] = {}
    for k in all_keys:
        vals = [s.env.get(k, AbstractValue.bottom()) for s in states]
        merged = vals[0]
        for v in vals[1:]:
            merged = merged.join(v)
        merged_env[k] = merged

    # Merge attrs
    all_objs: Set[str] = set()
    for s in states:
        all_objs |= s.attrs.keys()
    merged_attrs: Dict[str, Dict[str, AbstractValue]] = {}
    for obj in all_objs:
        all_attr_keys: Set[str] = set()
        for s in states:
            if obj in s.attrs:
                all_attr_keys |= s.attrs[obj].keys()
        merged_attrs[obj] = {}
        for ak in all_attr_keys:
            vals = [
                s.attrs.get(obj, {}).get(ak, AbstractValue.bottom()) for s in states
            ]
            merged = vals[0]
            for v in vals[1:]:
                merged = merged.join(v)
            merged_attrs[obj][ak] = merged

    # Collect all reads/writes from all paths
    all_reads = []
    all_writes = []
    seen_reads: Set[Tuple[str, int, int]] = set()
    seen_writes: Set[Tuple[str, int, int]] = set()
    for s in states:
        for r in s.reads:
            key = (r.name, r.lineno, r.col_offset)
            if key not in seen_reads:
                seen_reads.add(key)
                all_reads.append(r)
        for w in s.writes:
            key = (w.name, w.lineno, w.col_offset)
            if key not in seen_writes:
                seen_writes.add(key)
                all_writes.append(w)

    # Merge return values
    ret_vals = [s.returned for s in states]
    if all(r is None for r in ret_vals):
        merged_ret = None
    else:
        concrete = [r if r is not None else AbstractValue.bottom() for r in ret_vals]
        merged_ret = concrete[0]
        for v in concrete[1:]:
            merged_ret = merged_ret.join(v)

    return AbstractState(
        env=merged_env,
        attrs=merged_attrs,
        reads=all_reads,
        writes=all_writes,
        returned=merged_ret,
    )


# ---------------------------------------------------------------------------
# Known function definitions (for inlining / known return types)
# ---------------------------------------------------------------------------

# Maps fully-qualified function names to their known return type.
# Extend this as needed; unknown functions produce CALL_RESULT.
KNOWN_BUILTINS: Dict[str, type] = {
    "len": int,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "abs": int,  # simplified
    "round": int,
    "repr": str,
    "type": type,
    "isinstance": bool,
    "hasattr": bool,
    "getattr": object,
    "print": type(None),
    "range": range,
    "sorted": list,
    "reversed": object,
    "enumerate": object,
    "zip": object,
    "map": object,
    "filter": object,
    "min": object,
    "max": object,
    "sum": int,
    "any": bool,
    "all": bool,
    "ord": int,
    "chr": str,
    "hex": str,
    "oct": str,
    "bin": str,
    "id": int,
    "hash": int,
    "input": str,
    "open": object,
}

ANNOT_SUPPORTED_TYPES = {"int": int, "float": float, "bool": bool}

# ---------------------------------------------------------------------------
# Abstract Interpreter
# ---------------------------------------------------------------------------


class UnsupportedFeatureError(Exception):
    """Raised when an unsupported or unfinished feature is encountered."""

    pass


class AbstractInterpreter:
    """
    Path-sensitive abstract interpreter for a single Python function.
    No loops, no recursion. Explores all branches.
    """

    def __init__(
        self,
        fun: str | Callable,
        known_functions: Dict[str, ast.FunctionDef] | None = None,
    ):
        if isinstance(fun, str):
            func_source = fun
        else:
            func_source = inspect.getsource(fun)
        self.source = textwrap.dedent(func_source)
        self.tree = ast.parse(self.source)
        self.func_def: ast.FunctionDef | None = None
        # User-provided function definitions we can inline
        self.known_functions = known_functions or {}

        # Extract the function definition
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                self.func_def = node
                break
        if self.func_def is None:
            raise ValueError("No function definition found in source")

    def analyze(
        self, arg_values: Dict[str, AbstractValue] | None = None
    ) -> List[AbstractState]:
        """
        Analyze the function. Returns a list of final states, one per
        explored path.

        arg_values: optional mapping from parameter names to abstract values.
        """
        initial = AbstractState()

        # Bind parameters
        assert self.func_def is not None
        args = self.func_def.args
        if args.vararg or args.kwarg or args.kw_defaults:
            raise UnsupportedFeatureError(
                "*args, **kwargs, kw_defaults not yet supported"
            )
        if args.posonlyargs:
            raise UnsupportedFeatureError("positional-only args not yet supported")

        # defaults = args.defaults
        # kw_defaults = args.kw_defaults
        all_params = args.args + args.kwonlyargs

        if arg_values is None:
            arg_values = {}

        for param in all_params:
            name = param.arg
            if name in arg_values:
                initial.env[name] = arg_values[name]
            else:
                annot = param.annotation
                value = AbstractValue.top()
                if annot is not None:
                    if not isinstance(annot, ast.Name):
                        raise NotImplementedError("Unsupported type annotation")
                    ty = ANNOT_SUPPORTED_TYPES.get(annot.id)
                    if ty is not None:
                        value = AbstractValue.typed(ty)
                initial.env[name] = value

        return self._interpret_block(self.func_def.body, initial)

    # -------------------------------------------------------------------
    # Block / Statement interpretation
    # -------------------------------------------------------------------

    def _interpret_block(
        self, stmts: List[ast.stmt], state: AbstractState
    ) -> List[AbstractState]:
        """
        Interpret a block of statements. Returns list of states (one per path).
        """
        # We carry a list of "live" states through each statement.
        live_states = [state]

        for stmt in stmts:
            next_live: List[AbstractState] = []
            for s in live_states:
                if s.returned is not None:
                    # Already returned on this path; skip further stmts
                    next_live.append(s)
                    continue
                result_states = self._interpret_stmt(stmt, s)
                next_live.extend(result_states)
            live_states = next_live
            if not live_states:
                break

        return live_states

    def _interpret_stmt(
        self, stmt: ast.stmt, state: AbstractState
    ) -> List[AbstractState]:
        """Interpret one statement. Returns list of resulting states (path explosion)."""

        if isinstance(stmt, ast.Assign):
            return self._interpret_assign(stmt, state)

        elif isinstance(stmt, ast.AugAssign):
            return self._interpret_aug_assign(stmt, state)

        elif isinstance(stmt, ast.AnnAssign):
            return self._interpret_ann_assign(stmt, state)

        elif isinstance(stmt, ast.Return):
            return self._interpret_return(stmt, state)

        elif isinstance(stmt, ast.If):
            return self._interpret_if(stmt, state)

        elif isinstance(stmt, ast.Expr):
            # Expression statement (e.g., a function call ignoring return)
            val, new_state = self._eval_expr(stmt.value, state)
            return [new_state]

        elif isinstance(stmt, ast.Assert):
            # We explore the path where the assertion holds (no pruning)
            val, new_state = self._eval_expr(stmt.test, state)
            return [new_state]

        elif isinstance(stmt, ast.Pass):
            return [state]

        elif isinstance(stmt, ast.Delete):
            s = state.copy()
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    s.env.pop(target.id, None)
                else:
                    raise UnsupportedFeatureError(
                        f"Delete target {type(target).__name__} not supported"
                    )
            return [s]

        elif isinstance(stmt, ast.Raise):
            return self._interpret_raise(stmt, state)

        elif isinstance(stmt, (ast.For, ast.While, ast.AsyncFor)):
            raise UnsupportedFeatureError("Loops are not supported")

        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise UnsupportedFeatureError("Nested function definitions not supported")

        elif isinstance(stmt, ast.ClassDef):
            raise UnsupportedFeatureError("Class definitions not supported")

        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            raise UnsupportedFeatureError("Import statements not supported")

        elif isinstance(stmt, ast.Try):
            raise UnsupportedFeatureError("Try/except not yet supported")

        elif isinstance(stmt, ast.With):
            raise UnsupportedFeatureError("With statements not yet supported")

        elif isinstance(stmt, ast.Global):
            raise UnsupportedFeatureError("Global statements not supported")

        elif isinstance(stmt, ast.Nonlocal):
            raise UnsupportedFeatureError("Nonlocal statements not supported")

        else:
            raise UnsupportedFeatureError(
                f"Statement type {type(stmt).__name__} not supported"
            )

    # -------------------------------------------------------------------
    # Individual statement handlers
    # -------------------------------------------------------------------

    def _interpret_assign(
        self, stmt: ast.Assign, state: AbstractState
    ) -> List[AbstractState]:
        val, s = self._eval_expr(stmt.value, state)
        for target in stmt.targets:
            s = self._assign_target(target, val, s, stmt)
        return [s]

    def _interpret_aug_assign(
        self, stmt: ast.AugAssign, state: AbstractState
    ) -> List[AbstractState]:
        left_val, s = self._eval_expr(stmt.target, state)
        right_val, s = self._eval_expr(stmt.value, s)
        result = self._eval_binop_values(stmt.op, left_val, right_val)
        s = self._assign_target(stmt.target, result, s, stmt)
        return [s]

    def _interpret_ann_assign(
        self, stmt: ast.AnnAssign, state: AbstractState
    ) -> List[AbstractState]:
        if stmt.value is not None:
            val, s = self._eval_expr(stmt.value, state)
        else:
            # Annotation without value: variable declared but not assigned
            s = state.copy()
            val = AbstractValue.bottom()
        if stmt.target is not None and stmt.value is not None:
            s = self._assign_target(stmt.target, val, s, stmt)
        return [s]

    def _interpret_return(
        self, stmt: ast.Return, state: AbstractState
    ) -> List[AbstractState]:
        s = state.copy()
        if stmt.value is not None:
            val, s = self._eval_expr(stmt.value, s)
        else:
            val = AbstractValue.const(None)
        s.returned = val
        return [s]

    def _interpret_if(self, stmt: ast.If, state: AbstractState) -> List[AbstractState]:
        # Evaluate the condition (for side-effects / reads)
        cond_val, cond_state = self._eval_expr(stmt.test, state)

        # Explore BOTH branches unconditionally (path-sensitive)
        then_states = self._interpret_block(stmt.body, cond_state.copy())
        else_states = self._interpret_block(
            stmt.orelse if stmt.orelse else [], cond_state.copy()
        )

        return then_states + else_states

    def _interpret_raise(
        self, stmt: ast.Raise, state: AbstractState
    ) -> List[AbstractState]:
        s = state.copy()
        if stmt.exc is not None:
            val, s = self._eval_expr(stmt.exc, s)
        # Mark path as terminated (we treat raise like return for simplicity)
        s.returned = AbstractValue.typed(BaseException)
        return [s]

    # -------------------------------------------------------------------
    # Target assignment (handles Name, Attribute, Tuple/List unpacking)
    # -------------------------------------------------------------------

    def _assign_target(
        self,
        target: ast.expr,
        val: AbstractValue,
        state: AbstractState,
        source_node: ast.AST,
    ) -> AbstractState:
        s = state.copy()
        if isinstance(target, ast.Name):
            s.write_var(target.id, val, target)

        elif isinstance(target, ast.Attribute):
            obj_name = self._expr_to_name(target.value)
            if obj_name is None:
                raise UnsupportedFeatureError(
                    "Attribute assignment on complex expression not supported"
                )
            # Also record a read of the object itself
            s.read_var(obj_name, target.value) if isinstance(
                target.value, ast.Name
            ) else None
            s.write_attr(obj_name, target.attr, val, target)

        elif isinstance(target, ast.Subscript):
            raise UnsupportedFeatureError("Subscript assignment not yet supported")

        elif isinstance(target, (ast.Tuple, ast.List)):
            # Unpacking: if val is a known tuple/list constant, unpack it
            if val.is_const() and isinstance(val.value, (tuple, list)):
                if len(val.value) != len(target.elts):
                    # Mismatch; still assign TOP to each
                    for elt in target.elts:
                        s = self._assign_target(
                            elt, AbstractValue.top(), s, source_node
                        )
                else:
                    for elt, v in zip(target.elts, val.value):
                        s = self._assign_target(
                            elt, AbstractValue.const(v), s, source_node
                        )
            else:
                for elt in target.elts:
                    if isinstance(elt, ast.Starred):
                        s = self._assign_target(
                            elt.value, AbstractValue.typed(list), s, source_node
                        )
                    else:
                        s = self._assign_target(
                            elt, AbstractValue.top(), s, source_node
                        )

        elif isinstance(target, ast.Starred):
            s = self._assign_target(
                target.value, AbstractValue.typed(list), s, source_node
            )

        else:
            raise UnsupportedFeatureError(
                f"Assignment target {type(target).__name__} not supported"
            )
        return s

    # -------------------------------------------------------------------
    # Expression evaluation
    # -------------------------------------------------------------------

    def _eval_expr(
        self, expr: ast.expr, state: AbstractState
    ) -> Tuple[AbstractValue, AbstractState]:
        """
        Evaluate an expression. Returns (abstract value, updated state).
        State may change due to reads being recorded or function call side-effects.
        """
        s = state.copy()

        if isinstance(expr, ast.Constant):
            return AbstractValue.const(expr.value), s

        elif isinstance(expr, ast.Name):
            val = s.read_var(expr.id, expr)
            return val, s

        elif isinstance(expr, ast.Attribute):
            obj_val, s = self._eval_expr(expr.value, s)
            obj_name = self._expr_to_name(expr.value)
            if obj_name is not None:
                val = s.read_attr(obj_name, expr.attr, expr)
                return val, s
            else:
                return AbstractValue.top(), s

        elif isinstance(expr, ast.BinOp):
            left, s = self._eval_expr(expr.left, s)
            right, s = self._eval_expr(expr.right, s)
            result = self._eval_binop_values(expr.op, left, right)
            return result, s

        elif isinstance(expr, ast.UnaryOp):
            operand, s = self._eval_expr(expr.operand, s)
            result = self._eval_unaryop_value(expr.op, operand)
            return result, s

        elif isinstance(expr, ast.BoolOp):
            return self._eval_boolop(expr, s)

        elif isinstance(expr, ast.Compare):
            return self._eval_compare(expr, s)

        elif isinstance(expr, ast.Call):
            return self._eval_call(expr, s)

        elif isinstance(expr, ast.IfExp):
            return self._eval_ifexp(expr, s)

        elif isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
            return self._eval_collection(expr, s)

        elif isinstance(expr, ast.Dict):
            return self._eval_dict(expr, s)

        elif isinstance(expr, ast.Subscript):
            return self._eval_subscript(expr, s)

        elif isinstance(expr, ast.Starred):
            raise UnsupportedFeatureError(
                "Starred expression in non-assignment context"
            )

        elif isinstance(expr, ast.JoinedStr):
            # f-string: evaluate all values
            for val_node in expr.values:
                if isinstance(val_node, ast.FormattedValue):
                    _, s = self._eval_expr(val_node.value, s)
                elif isinstance(val_node, ast.Constant):
                    pass
            return AbstractValue.typed(str), s

        elif isinstance(expr, ast.FormattedValue):
            val, s = self._eval_expr(expr.value, s)
            return AbstractValue.typed(str), s

        elif isinstance(expr, ast.NamedExpr):
            val, s = self._eval_expr(expr.value, s)
            s.write_var(expr.target.id, val, expr.target)
            return val, s

        elif isinstance(
            expr, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            raise UnsupportedFeatureError("Comprehensions not yet supported")

        elif isinstance(expr, ast.Lambda):
            raise UnsupportedFeatureError("Lambda expressions not yet supported")

        elif isinstance(expr, ast.Await):
            raise UnsupportedFeatureError("Await expressions not supported")

        elif isinstance(expr, ast.Yield):
            raise UnsupportedFeatureError("Yield expressions not supported")

        elif isinstance(expr, ast.Slice):
            # Evaluate bounds for tracking
            if expr.lower:
                _, s = self._eval_expr(expr.lower, s)
            if expr.upper:
                _, s = self._eval_expr(expr.upper, s)
            if expr.step:
                _, s = self._eval_expr(expr.step, s)
            return AbstractValue.typed(slice), s

        else:
            raise UnsupportedFeatureError(
                f"Expression type {type(expr).__name__} not supported"
            )

    # -------------------------------------------------------------------
    # Specific expression evaluators
    # -------------------------------------------------------------------

    def _eval_binop_values(
        self, op: ast.operator, left: AbstractValue, right: AbstractValue
    ) -> AbstractValue:
        """Try to compute binop on abstract values."""
        if left.is_const() and right.is_const():
            try:
                result = self._apply_binop(op, left.value, right.value)
                return AbstractValue.const(result)
            except Exception:
                return AbstractValue.top()
        # Infer type from operand types
        result_type = self._binop_result_type(op, left.type_, right.type_)
        if result_type is not None:
            return AbstractValue.typed(result_type)
        return AbstractValue.top()

    def _apply_binop(self, op: ast.operator, a: Any, b: Any) -> Any:
        match op:
            case ast.Add():
                return a + b
            case ast.Sub():
                return a - b
            case ast.Mult():
                return a * b
            case ast.Div():
                return a / b
            case ast.FloorDiv():
                return a // b
            case ast.Mod():
                return a % b
            case ast.Pow():
                return a**b
            case ast.LShift():
                return a << b
            case ast.RShift():
                return a >> b
            case ast.BitOr():
                return a | b
            case ast.BitXor():
                return a ^ b
            case ast.BitAnd():
                return a & b
            case ast.MatMult():
                raise UnsupportedFeatureError("Matrix multiply not supported")
            case _:
                raise UnsupportedFeatureError(
                    f"BinOp {type(op).__name__} not supported"
                )

    def _binop_result_type(
        self, op: ast.operator, lt: type | None, rt: type | None
    ) -> type | None:
        if lt is None or rt is None:
            return None
        # Numeric promotion
        numeric = {int, float, complex}
        if lt in numeric and rt in numeric:
            if isinstance(op, ast.Div):
                return float
            if complex in (lt, rt):
                return complex
            if float in (lt, rt):
                return float
            return int
        # String repetition
        if lt == str and rt == int and isinstance(op, ast.Mult):
            return str
        if lt == str and rt == str and isinstance(op, ast.Add):
            return str
        if lt == list and rt == list and isinstance(op, ast.Add):
            return list
        return None

    def _eval_unaryop_value(
        self, op: ast.unaryop, operand: AbstractValue
    ) -> AbstractValue:
        if operand.is_const():
            try:
                match op:
                    case ast.UAdd():
                        result = +operand.value
                    case ast.USub():
                        result = -operand.value
                    case ast.Invert():
                        result = ~operand.value
                    case ast.Not():
                        result = not operand.value
                    case _:
                        return AbstractValue.top()
                return AbstractValue.const(result)
            except Exception:
                return AbstractValue.top()
        if isinstance(op, ast.Not):
            return AbstractValue.typed(bool)
        if operand.type_ is not None:
            return AbstractValue.typed(operand.type_)
        return AbstractValue.top()

    def _eval_boolop(
        self, expr: ast.BoolOp, state: AbstractState
    ) -> Tuple[AbstractValue, AbstractState]:
        s = state
        result = AbstractValue.bottom()
        for val_node in expr.values:
            v, s = self._eval_expr(val_node, s)
            result = result.join(v)
        # BoolOps can return any of the operand values; type is their join
        return result, s

    def _eval_compare(
        self, expr: ast.Compare, state: AbstractState
    ) -> Tuple[AbstractValue, AbstractState]:
        s = state
        left, s = self._eval_expr(expr.left, s)
        all_const = left.is_const()
        values = [left]
        for comp in expr.comparators:
            v, s = self._eval_expr(comp, s)
            values.append(v)
            if not v.is_const():
                all_const = False

        if all_const and len(expr.ops) == 1:
            try:
                result = self._apply_compare(
                    expr.ops[0], values[0].value, values[1].value
                )
                return AbstractValue.const(result), s
            except Exception:
                pass

        return AbstractValue.typed(bool), s

    def _apply_compare(self, op: ast.cmpop, a: Any, b: Any) -> bool:
        match op:
            case ast.Eq():
                return a == b
            case ast.NotEq():
                return a != b
            case ast.Lt():
                return a < b
            case ast.LtE():
                return a <= b
            case ast.Gt():
                return a > b
            case ast.GtE():
                return a >= b
            case ast.Is():
                return a is b
            case ast.IsNot():
                return a is not b
            case ast.In():
                return a in b
            case ast.NotIn():
                return a not in b
            case _:
                raise UnsupportedFeatureError(
                    f"Compare op {type(op).__name__} not supported"
                )

    def _eval_call(
        self, expr: ast.Call, state: AbstractState
    ) -> Tuple[AbstractValue, AbstractState]:
        s = state

        # Evaluate the callable
        func_val, s = self._eval_expr(expr.func, s)
        func_name = self._expr_to_name(expr.func)

        # Evaluate arguments (for reads / side-effects)
        arg_vals = []
        for arg in expr.args:
            if isinstance(arg, ast.Starred):
                raise UnsupportedFeatureError(
                    "Starred call arguments not yet supported"
                )
            v, s = self._eval_expr(arg, s)
            arg_vals.append(v)

        kw_vals = []
        for kw in expr.keywords:
            v, s = self._eval_expr(kw.value, s)
            kw_vals.append((kw.arg, v))

        # Try to compute result for known builtins with const args
        if func_name and func_name in KNOWN_BUILTINS:
            ret_type = KNOWN_BUILTINS[func_name]
            # Try actual computation for type constructors
            if (
                func_name in ("int", "float", "str", "bool")
                and len(arg_vals) <= 1
                and all(a.is_const() for a in arg_vals)
            ):
                try:
                    builtin_fn = (
                        __builtins__[func_name]
                        if isinstance(__builtins__, dict)
                        else getattr(__builtins__, func_name)
                    )
                    if arg_vals:
                        result = builtin_fn(arg_vals[0].value)
                    else:
                        result = builtin_fn()
                    return AbstractValue.const(result), s
                except Exception:
                    pass
            return AbstractValue.typed(ret_type), s

        # Unknown function: produce symbolic CALL_RESULT
        if func_name:
            arg_reprs = []
            for av in arg_vals:
                if av.is_const():
                    arg_reprs.append(repr(av.value))
                else:
                    arg_reprs.append("?")
            for kw_name, kw_val in kw_vals:
                if kw_val.is_const():
                    arg_reprs.append(f"{kw_name}={repr(kw_val.value)}")
                else:
                    arg_reprs.append(f"{kw_name}=?")
            call_repr = f"{func_name}({', '.join(arg_reprs)})"
            return AbstractValue.call_result(call_repr), s

        # Totally unknown callable
        return AbstractValue.top(), s

    def _eval_ifexp(
        self, expr: ast.IfExp, state: AbstractState
    ) -> Tuple[AbstractValue, AbstractState]:
        cond, s = self._eval_expr(expr.test, state)

        # Explore both branches
        then_val, then_s = self._eval_expr(expr.body, s.copy())
        else_val, else_s = self._eval_expr(expr.orelse, s.copy())

        merged = join_states([then_s, else_s])
        result = then_val.join(else_val)
        return result, merged

    def _eval_collection(
        self, expr: Union[ast.List, ast.Tuple, ast.Set], state: AbstractState
    ) -> Tuple[AbstractValue, AbstractState]:
        s = state
        elts = []
        all_const = True
        for e in expr.elts:
            if isinstance(e, ast.Starred):
                raise UnsupportedFeatureError(
                    "Starred in collection literal not supported"
                )
            v, s = self._eval_expr(e, s)
            elts.append(v)
            if not v.is_const():
                all_const = False

        if all_const:
            raw = [e.value for e in elts]
            if isinstance(expr, ast.List):
                return AbstractValue.const(raw), s
            elif isinstance(expr, ast.Tuple):
                return AbstractValue.const(tuple(raw)), s
            elif isinstance(expr, ast.Set):
                return AbstractValue.const(set(raw)), s

        if isinstance(expr, ast.List):
            return AbstractValue.typed(list), s
        elif isinstance(expr, ast.Tuple):
            return AbstractValue.typed(tuple), s
        elif isinstance(expr, ast.Set):
            return AbstractValue.typed(set), s
        return AbstractValue.top(), s

    def _eval_dict(
        self, expr: ast.Dict, state: AbstractState
    ) -> Tuple[AbstractValue, AbstractState]:
        s = state
        all_const = True
        keys = []
        vals = []
        for k, v in zip(expr.keys, expr.values):
            if k is None:
                raise UnsupportedFeatureError("Dict unpacking (**) not supported")
            kv, s = self._eval_expr(k, s)
            vv, s = self._eval_expr(v, s)
            keys.append(kv)
            vals.append(vv)
            if not kv.is_const() or not vv.is_const():
                all_const = False

        if all_const:
            d = {kv.value: vv.value for kv, vv in zip(keys, vals)}
            return AbstractValue.const(d), s

        return AbstractValue.typed(dict), s

    def _eval_subscript(
        self, expr: ast.Subscript, state: AbstractState
    ) -> Tuple[AbstractValue, AbstractState]:
        obj_val, s = self._eval_expr(expr.value, state)
        idx_val, s = self._eval_expr(expr.slice, s)

        if obj_val.is_const() and idx_val.is_const():
            try:
                result = obj_val.value[idx_val.value]
                return AbstractValue.const(result), s
            except Exception:
                pass

        return AbstractValue.top(), s

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _expr_to_name(self, expr: ast.expr) -> str | None:
        """Try to extract a dotted name from an expression."""
        if isinstance(expr, ast.Name):
            return expr.id
        elif isinstance(expr, ast.Attribute):
            base = self._expr_to_name(expr.value)
            if base:
                return f"{base}.{expr.attr}"
        return None


# ---------------------------------------------------------------------------
# Analysis Result Printer
# ---------------------------------------------------------------------------


def format_results(states: List[AbstractState]) -> str:
    """Pretty-print the analysis results."""
    lines = []
    lines.append(f"=== Analysis Results: {len(states)} path(s) explored ===\n")

    for i, st in enumerate(states):
        lines.append(f"--- Path {i + 1} ---")
        lines.append("  Variables:")
        for name, val in sorted(st.env.items()):
            lines.append(f"    {name} = {val}")
        if st.attrs:
            lines.append("  Attributes:")
            for obj, attrs in sorted(st.attrs.items()):
                for attr, val in sorted(attrs.items()):
                    lines.append(f"    {obj}.{attr} = {val}")
        lines.append(f"  Return: {st.returned}")
        lines.append(f"  Reads:  {[f'{r.name} ({r.lineno})' for r in st.reads]}")
        lines.append(f"  Writes: {[f'{w.name} ({w.lineno})' for w in st.writes]}")
        lines.append("")

    # Merged summary
    merged = join_states(states)
    lines.append("--- Merged (all paths) ---")
    lines.append("  Variables:")
    for name, val in sorted(merged.env.items()):
        lines.append(f"    {name} = {val}")
    if merged.attrs:
        lines.append("  Attributes:")
        for obj, attrs in sorted(merged.attrs.items()):
            for attr, val in sorted(attrs.items()):
                lines.append(f"    {obj}.{attr} = {val}")
    lines.append(f"  Return: {merged.returned}")
    lines.append(f"  All reads:  {[r.name for r in merged.reads]}")
    lines.append(f"  All writes: {[w.name for w in merged.writes]}")

    return "\n".join(lines)
