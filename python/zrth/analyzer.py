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
import torch
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Set, Tuple, Union, Callable

from zrth import Wire, Term, Sort
from .builder import (
    TermBuilder,
    LRATermBuilder,
    LIATermBuilder,
    BVTermBuilder,
    _normalize_shape,
)


# ---------------------------------------------------------------------------
# Compile-time static values (str/None attributes used in comparisons)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StaticValue:
    """Compile-time primitive value (str, None, etc.) used for static evaluation of
    comparisons and if-conditions involving non-wire Python attributes."""

    value: Any


def _eval_static_compare(a, op, b):
    """Evaluate a Python-level comparison on two primitive values at analysis time."""
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
        case _:
            raise ValueError(f"Unsupported static compare op: {type(op).__name__}")


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


def _merge_dicts(states, get_dict):
    """Merge dicts from multiple states using AbstractValue.join."""
    all_keys: Set[str] = set()
    for s in states:
        all_keys |= get_dict(s).keys()
    merged: Dict[str, AbstractValue] = {}
    for k in all_keys:
        vals = [get_dict(s).get(k, AbstractValue.bottom()) for s in states]
        result = vals[0]
        for v in vals[1:]:
            result = result.join(v)
        merged[k] = result
    return merged


def _dedup_records(states, get_list):
    """Collect AccessRecords from all states, deduplicating by (name, line, col)."""
    seen: Set[Tuple[str, int, int]] = set()
    out = []
    for s in states:
        for r in get_list(s):
            key = (r.name, r.lineno, r.col_offset)
            if key not in seen:
                seen.add(key)
                out.append(r)
    return out


def join_states(states: List[AbstractState]) -> AbstractState:
    """Join multiple path states into one (merge at control-flow join point)."""
    if not states:
        s = AbstractState()
        s.returned = AbstractValue.bottom()
        return s
    if len(states) == 1:
        return states[0]

    merged_env = _merge_dicts(states, lambda s: s.env)

    # Merge attrs (nested: object -> attr -> value)
    all_objs: Set[str] = set()
    for s in states:
        all_objs |= s.attrs.keys()
    merged_attrs = {
        obj: _merge_dicts(states, lambda s, o=obj: s.attrs.get(o, {}))
        for obj in all_objs
    }

    all_reads = _dedup_records(states, lambda s: s.reads)
    all_writes = _dedup_records(states, lambda s: s.writes)

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

        else:
            _UNSUPPORTED_STMTS = {
                ast.For: "Loops",
                ast.While: "Loops",
                ast.AsyncFor: "Loops",
                ast.FunctionDef: "Nested function definitions",
                ast.AsyncFunctionDef: "Nested function definitions",
                ast.ClassDef: "Class definitions",
                ast.Import: "Import statements",
                ast.ImportFrom: "Import statements",
                ast.Try: "Try/except",
                ast.With: "With statements",
                ast.Global: "Global statements",
                ast.Nonlocal: "Nonlocal statements",
            }
            msg = _UNSUPPORTED_STMTS.get(type(stmt))
            if msg:
                raise UnsupportedFeatureError(f"{msg} not supported")
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
        if lt is str and rt is int and isinstance(op, ast.Mult):
            return str
        if isinstance(op, ast.Add):
            if lt is str and rt is str:
                return str
            if lt is list and rt is list:
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

        py_type = {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(expr)]
        if all_const:
            return AbstractValue.const(py_type(e.value for e in elts)), s
        return AbstractValue.typed(py_type), s

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


# ---------------------------------------------------------------------------
# Module helpers (used by Env / NN wrapping logic)
# ---------------------------------------------------------------------------


def wire_pair(dtype):
    """Create a [latched, next] wire pair for the given dtype."""
    return [Wire(dtype), Wire(dtype)]


def resolve_wire(name, dtype, user_val=None):
    """Return a validated [latched, next] wire pair for an observable signal.

    If user_val is None, creates a fresh pair from dtype.
    If user_val is a wire pair, validates its dtype and returns it.
    """
    if user_val is None:
        return wire_pair(dtype)
    is_pair = (
        isinstance(user_val, (list, tuple))
        and len(user_val) == 2
        and all(isinstance(w, Wire) for w in user_val)
    )
    if is_pair:
        for w in user_val:
            if w.dtype != dtype:
                raise ValueError(
                    f"DType mismatch for '{name}': expected {dtype}, got {w.dtype}"
                )
        return list(user_val)
    raise ValueError(
        f"Invalid wire format for '{name}': expected [Wire, Wire], got {type(user_val).__name__}"
    )


def _infer_shape_and_elem_type(value):
    """Recursively derive tensor shape and element type from a Python value."""
    if isinstance(value, bool):  # before int -- bool subclasses int
        return [], bool
    if isinstance(value, (int, float)):
        return [], type(value)
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            elem = bool
        elif value.dtype.is_floating_point:
            elem = float
        else:
            elem = int
        return list(value.shape), elem
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Cannot infer shape from empty collection")
        inner_shape, elem_type = _infer_shape_and_elem_type(value[0])
        return [len(value)] + inner_shape, elem_type
    raise ValueError(f"Unsupported element type: {type(value).__name__}")


def infer_dtype(name, abstract_value, builder):
    """Infer a DType from an AbstractValue using the builder's theory."""
    if abstract_value is None:
        raise ValueError(f"Cannot infer DType for '{name}': analyzer returned None")

    if abstract_value.is_const():
        shape, elem_type = _infer_shape_and_elem_type(abstract_value.value)
        return builder.python_type_to_dtype(elem_type, _normalize_shape(shape or [1]))

    # np.array(<literal>) CallResult (e.g. from _instance_to_init_attrs):
    # recover shape and element type from the literal data.
    repr_ = abstract_value.call_repr
    if repr_ and (repr_.startswith("np.array(") or repr_.startswith("numpy.array(")):
        data = ast.literal_eval(repr_[repr_.index("(") + 1 : repr_.rindex(")")])
        shape, elem_type = _infer_shape_and_elem_type(data)
        return builder.python_type_to_dtype(elem_type, _normalize_shape(shape or [1]))

    if abstract_value.type_ is None:
        raise ValueError(
            f"Cannot infer Sort for '{name}': analyzer returned {abstract_value}"
        )
    return builder.python_type_to_dtype(abstract_value.type_, _normalize_shape([1]))


def classify_attrs(cls, roots, init_attrs=None, base_cls=None):
    """Classify self.* attributes used in root methods (and their callees).

    Walks cls.__mro__ up to (but not including) base_cls, so only user-defined
    methods are analyzed -- not framework methods from Env, Module, gym.Env, etc.

    Args:
        cls:        The class to analyze.
        roots:      Method names to start from (e.g. ['reset', 'step']).
        init_attrs: Optional dict[str, AbstractValue] from analyze_init, used
                    as a fallback for attrs whose values are unknown in roots.
        base_cls:   Stop walking the MRO at this class (exclusive). Pass Env or
                    NN to exclude framework-level methods from analysis.

    Returns:
        prvt:      set -- attributes both read and written (private mutable state)
        params:    set -- attributes only read (constants set in __init__)
        attr_vals: dict[str, AbstractValue] -- best-known value for each attr

    Raises:
        ValueError: if any attribute is written but never read back.
    """
    # Collect user-defined methods by walking the MRO up to base_cls.
    # Iterate most-derived-first; setdefault keeps the most-derived definition.
    methods = {}
    for klass in cls.__mro__:
        if klass is base_cls or klass is object:
            break
        for name, val in klass.__dict__.items():
            if callable(val) and not isinstance(val, (staticmethod, classmethod)):
                methods.setdefault(name, val)

    # Analyze each method individually
    summaries = {}
    for name, method in methods.items():
        try:
            merged = join_states(AbstractInterpreter(method).analyze())
        except (UnsupportedFeatureError, NotImplementedError, OSError):
            continue
        read_attrs = {r.name[5:] for r in merged.reads if r.name.startswith("self.")}
        written_attrs = {
            w.name[5:] for w in merged.writes if w.name.startswith("self.")
        }
        # self.foo reads where foo is a known method -> calls, not data reads
        calls = read_attrs & set(methods.keys())
        read_attrs -= calls
        summaries[name] = (
            read_attrs,
            written_attrs,
            calls,
            merged.attrs.get("self", {}),
        )

    # BFS from roots, following intra-class calls
    visited, queue = set(), list(roots)
    while queue:
        name = queue.pop()
        if name in visited or name not in summaries:
            continue
        visited.add(name)
        queue.extend(summaries[name][2])  # calls

    read_self, written_self, attr_vals = set(), set(), {}
    for name in visited:
        ra, wa, _, av = summaries[name]
        read_self |= ra
        written_self |= wa
        for attr, val in av.items():
            existing = attr_vals.get(attr)
            if existing is None or (val.is_const() and not existing.is_const()):
                attr_vals[attr] = val

    prvt_names = written_self & read_self
    param_names = read_self - written_self
    write_only_names = written_self - read_self

    # Order by declaration in __init__ for deterministic iteration,
    # with any names not in __init__ appended in sorted order
    decl_order = list(init_attrs.keys()) if init_attrs else []

    def ordered(names):
        return [n for n in decl_order if n in names] + sorted(names - set(decl_order))

    prvt = ordered(prvt_names)
    params = ordered(param_names)
    write_only = ordered(write_only_names)

    # Use init_attrs as a fallback for attrs with missing or non-const values
    if init_attrs:
        for attr in prvt + params:
            val = attr_vals.get(attr)
            init_val = init_attrs.get(attr)
            if init_val is not None and (val is None or not val.is_const()):
                attr_vals[attr] = init_val

    if write_only:
        raise ValueError(
            f"Attributes written in {roots} but never read back: {write_only}. "
            f"These must be made observable."
        )

    return prvt, params, attr_vals


# ---------------------------------------------------------------------------
# AST-to-Terms Converter
# ---------------------------------------------------------------------------


def _torch_dtype(target_dtype):
    """Map a Sort to the corresponding torch dtype. Defaults to float32."""
    match target_dtype:
        case Sort.Int(_) | Sort.BitVec(_, _):
            return torch.long
        case Sort.Bool(_):
            return torch.bool
        case _:
            return torch.float32


def _normalize_early_returns(stmts: list) -> list:
    """Convert early-return patterns to single-exit form.

    Transforms:
        if cond:
            ...
            return val      <- replaced with _ret_i = val assignments
        rest...
        return final
    Into:
        if cond:
            ...
            _ret_0 = val0
            _ret_1 = val1
        else:
            rest...
            return final    <- visit_Return stores _ret_i in temp_vars

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
                ret.value.elts
                if isinstance(ret.value, ast.Tuple)
                else ([ret.value] if ret.value else [])
            )
            # Replace return with _ret_i = value assignments so SSA sees both branches
            ret_assigns = [
                ast.fix_missing_locations(
                    ast.copy_location(
                        ast.Assign(
                            targets=[ast.Name(id=f"_ret_{i}", ctx=ast.Store())],
                            value=val,
                            type_comment=None,
                        ),
                        ret,
                    )
                )
                for i, val in enumerate(values)
            ]
            body = _normalize_early_returns(stmt.body[:-1]) + ret_assigns
            rest = _normalize_early_returns(stmts[idx + 1 :])
            result.append(
                ast.copy_location(
                    ast.If(test=stmt.test, body=body or [ast.Pass()], orelse=rest),
                    stmt,
                )
            )
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


class MethodVisitor(ast.NodeVisitor):
    """AST visitor to convert Python methods to reactive Terms

    Intermediate values are stored as Wire objects in temp_vars.
    """

    def __init__(
        self,
        wire_pairs: dict[str, tuple[Wire, Wire]],
        result_wires: list[Wire],
        cls=None,
        layers: dict[str, int] | None = None,
        params: dict[str, Wire] | None = None,
        live_layers: dict | None = None,
        builder: TermBuilder | None = None,
        static_attrs: dict[str, Any] | None = None,
    ):
        self.wire_pairs = wire_pairs
        self.result_wires = result_wires
        self.cls = cls
        self.layers = layers or {}
        self.params = params or {}
        self.live_layers = live_layers or {}
        self.static_attrs = static_attrs or {}
        self.terms = []
        self.temp_vars = {}
        self.scopes = []
        self.written_wires = set()
        self.builder = builder or LRATermBuilder()

    @staticmethod
    def _w(val) -> Wire:
        """Extract wire from Term or Wire."""
        if isinstance(val, Term):
            return val.write[0]
        return val

    @staticmethod
    def _d(val):
        """Extract dtype from Term or Wire."""
        return (val.write[0] if isinstance(val, Term) else val).dtype

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

        # If cond is a compile-time constant, take just the chosen branch
        if isinstance(cond_wire, _StaticValue):
            body = node.body if cond_wire.value else node.orelse
            for stmt in body:
                self.visit(stmt)
            return

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
        # after the return -- only wire_pairs and _ret_* should escape.
        if any(k.startswith("_ret_") for k in if_scope_after):
            if_scope_after = {
                k: v for k, v in if_scope_after.items() if self._should_escape_scope(k)
            }
        if any(k.startswith("_ret_") for k in else_scope_after):
            else_scope_after = {
                k: v
                for k, v in else_scope_after.items()
                if self._should_escape_scope(k)
            }

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
                ite_term = self.builder.ite(cond_wire, if_wire, else_wire)
                self.terms.append(ite_term)
                self.temp_vars[var] = ite_term

                # If this is a state wire and we're at top level, write the merged
                # value to the output wire — replacing any earlier Id term, since
                # this if-branch supersedes whatever was assigned before it.
                if var in self.wire_pairs and len(self.scopes) == 0:
                    output_wire = self.wire_pairs[var][1]
                    if var in self.written_wires:
                        for i in range(len(self.terms) - 1, -1, -1):
                            t = self.terms[i]
                            if len(t.write) == 1 and t.write[0] == output_wire:
                                del self.terms[i]
                                break
                    term = self.builder.id_(ite_term, output_wire=output_wire)
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
                    term = self.builder.id_(if_wire, output_wire=output_wire)
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

        elif isinstance(target, ast.Tuple):
            # Tuple unpacking: x, y = <expr>. Supported only when RHS is a tuple
            # literal OR an untraced call (each target gets its own placeholder).
            if isinstance(node.value, ast.Tuple) and len(node.value.elts) == len(
                target.elts
            ):
                for tgt_elt, val_elt in zip(target.elts, node.value.elts):
                    if not isinstance(tgt_elt, ast.Name):
                        raise ValueError("Tuple unpacking targets must be plain names")
                    self.temp_vars[tgt_elt.id] = self._convert_expr(val_elt)
            elif isinstance(node.value, ast.Call):
                label = ast.unparse(node.value)
                for tgt_elt in target.elts:
                    if not isinstance(tgt_elt, ast.Name):
                        raise ValueError("Tuple unpacking targets must be plain names")
                    self.temp_vars[tgt_elt.id] = self._untraced_call(label)
            else:
                raise ValueError(
                    "Tuple unpacking supports tuple literals or untraced calls"
                )

        elif isinstance(target, ast.Attribute) and target.attr in self.wire_pairs:
            wire_name = target.attr
            target_dtype = self.wire_pairs[wire_name][1].dtype
            result_val = self._convert_expr(node.value, target_dtype=target_dtype)
            self.temp_vars[wire_name] = result_val

            if len(self.scopes) == 0:
                output_wire = self.wire_pairs[wire_name][1]
                if wire_name in self.written_wires:
                    # Re-assignment: drop the previous Id term for this output wire
                    for i in range(len(self.terms) - 1, -1, -1):
                        t = self.terms[i]
                        if len(t.write) == 1 and t.write[0] == output_wire:
                            del self.terms[i]
                            break
                term = self.builder.id_(result_val, output_wire=output_wire)
                self.terms.append(term)
                self.written_wires.add(wire_name)

    def visit_AugAssign(self, node):
        """Handle augmented assignment (+=, -=, *=, /=). Desugar to regular assignment
        so re-assignment deduping in visit_Assign applies uniformly."""
        if isinstance(node.target, ast.Name):
            load_target = ast.Name(id=node.target.id, ctx=ast.Load())
        elif isinstance(node.target, ast.Attribute):
            load_target = ast.Attribute(
                value=node.target.value, attr=node.target.attr, ctx=ast.Load()
            )
        else:
            raise ValueError(
                "Augmented assignment only supported for names or self.attribute wires"
            )
        ast.copy_location(load_target, node.target)
        binop = ast.BinOp(left=load_target, op=node.op, right=node.value)
        ast.copy_location(binop, node)
        assign = ast.Assign(targets=[node.target], value=binop)
        ast.copy_location(assign, node)
        self.visit_Assign(assign)

    def visit_Expr(self, node):
        """Silently skip bare expression statements (e.g. super().reset(seed=seed))"""
        pass

    def visit_Return(self, node):
        """Handle return statement: zip return values positionally with result_wires"""
        if node.value is None:
            return

        value_nodes = (
            node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
        )

        if len(value_nodes) < len(self.result_wires):
            raise ValueError(
                f"Return has {len(value_nodes)} value(s) but "
                f"{len(self.result_wires)} result wire(s) were declared"
            )
        # Truncate extra return values (e.g. gym's trailing info dict)
        value_nodes = value_nodes[: len(self.result_wires)]

        for i, (result_wire, value_node) in enumerate(
            zip(self.result_wires, value_nodes)
        ):
            src_wire = self._convert_expr(value_node, target_dtype=result_wire.dtype)
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
        elif isinstance(expr, ast.Subscript):
            return self._convert_subscript(expr, target_dtype)
        else:
            raise ValueError(f"Unsupported expression type: {type(expr).__name__}")

    def _convert_call(self, call, target_dtype=None):
        """Convert method/function call"""
        if isinstance(call.func, ast.Attribute):
            method = call.func.attr
            obj = call.func.value

            # Handle module/name-based calls first (np.*, self.*)
            if isinstance(obj, ast.Name):
                name = obj.id
                if name in ("np", "numpy"):
                    return self._convert_numpy_creation(
                        method, call.args, target_dtype=target_dtype
                    )
                elif name == "torch":
                    if method == "relu":
                        input_val = self._convert_expr(
                            call.args[0], target_dtype=target_dtype
                        )
                        term = self.builder.relu(input_val)
                        self.terms.append(term)
                        return term
                    elif method == "tanh":
                        input_val = self._convert_expr(
                            call.args[0], target_dtype=target_dtype
                        )
                        term = self.builder.tanh(input_val)
                        self.terms.append(term)
                        return term
                    elif method in ("sin", "cos"):
                        input_val = self._convert_expr(
                            call.args[0], target_dtype=target_dtype
                        )
                        term = (
                            self.builder.sin(input_val)
                            if method == "sin"
                            else self.builder.cos(input_val)
                        )
                        self.terms.append(term)
                        return term
                    else:
                        raise ValueError(f"Unsupported torch function: torch.{method}")
                elif name == "math":
                    if method in ("sin", "cos"):
                        input_val = self._convert_expr(
                            call.args[0], target_dtype=target_dtype
                        )
                        term = (
                            self.builder.sin(input_val)
                            if method == "sin"
                            else self.builder.cos(input_val)
                        )
                        self.terms.append(term)
                        return term
                    else:
                        raise ValueError(f"Unsupported math function: math.{method}")
                elif name == "self":
                    return self._inline_method(method, call.args)

            if method == "argmax":
                obj_val = self._convert_expr(obj)
                term = self.builder.argmax(obj_val)
                self.terms.append(term)
                return term
            elif method == "item":
                return self._convert_expr(obj)
            else:
                return self._untraced_call(ast.unparse(call), target_dtype)

        elif isinstance(call.func, ast.Name):
            func_name = call.func.id

            if func_name in ("min", "max"):
                return self._convert_minmax(call.args, func_name)
            elif func_name == "bool":
                if len(call.args) != 1:
                    raise ValueError("bool() requires 1 argument")
                return self._convert_expr(call.args[0], target_dtype=target_dtype)
            else:
                return self._untraced_call(ast.unparse(call), target_dtype)
        else:
            return self._untraced_call(ast.unparse(call), target_dtype)

    def _untraced_call(self, label, target_dtype=None):
        """Emit an Uninterpreted placeholder for calls we can't trace symbolically.

        The wire is produced symbolically but never evaluated at runtime in the
        gym.Env case — the real env overwrites the state via delegation.
        """
        # Compact label for display: drop arguments, strip leading "self."
        # e.g. "self.np_random.uniform(low=low, high=high)" -> "np_random.uniform(...)"
        head = label.split("(", 1)[0].lstrip()
        if head.startswith("self."):
            head = head[len("self."):]
        label = f"{head}(...)" if "(" in label else head

        sort = target_dtype if target_dtype is not None else Sort.Real([1, 1])
        term = self.builder.uninterpreted(label, sort)
        self.terms.append(term)
        return term

    def _convert_minmax(self, args, func_name):
        """Convert min/max to conditional: min(a,b) -> Ite(Lt(a,b), a, b)"""
        if len(args) != 2:
            raise ValueError(f"{func_name}() requires 2 arguments")

        a_val = self._convert_expr(args[0])
        b_val = self._convert_expr(args[1])

        cmp_term = (
            self.builder.lt(a_val, b_val)
            if func_name == "min"
            else self.builder.gt(a_val, b_val)
        )
        self.terms.append(cmp_term)
        ite_term = self.builder.ite(cmp_term, a_val, b_val)
        self.terms.append(ite_term)
        return ite_term

    def _make_tensor_wire(self, tensor_data, target_dtype=None):
        """Create a constant Term from tensor_data and append it."""
        shape = list(tensor_data.size())
        if len(shape) == 0:
            shape = [1, 1]
        elif len(shape) == 1:
            shape = [1, shape[0]]
        tensor_data = tensor_data.reshape(shape)
        term = self.builder.const(tensor_data)
        self.terms.append(term)
        return term

    def _convert_numpy_creation(self, func_name, args, target_dtype=None):
        """Convert np.zeros/ones/array/clip."""
        # np.clip(x, low, high) -> min(max(x, low), high)
        if func_name == "clip":
            if len(args) != 3:
                raise ValueError("np.clip() requires 3 arguments (x, low, high)")
            x_val = self._convert_expr(args[0], target_dtype=target_dtype)
            low_val = self._convert_expr(args[1], target_dtype=self._d(x_val))
            high_val = self._convert_expr(args[2], target_dtype=self._d(x_val))
            # max(x, low)
            cmp1 = self.builder.gt(x_val, low_val)
            self.terms.append(cmp1)
            lo = self.builder.ite(cmp1, x_val, low_val)
            self.terms.append(lo)
            # min(lo, high)
            cmp2 = self.builder.lt(lo, high_val)
            self.terms.append(cmp2)
            result = self.builder.ite(cmp2, lo, high_val)
            self.terms.append(result)
            return result

        if len(args) != 1:
            raise ValueError(f"np.{func_name}() requires 1 argument, got {len(args)}")

        tdtype = _torch_dtype(target_dtype)

        if func_name == "zeros":
            tensor_data = torch.zeros(*self._eval_shape(args[0]), dtype=tdtype)
        elif func_name == "ones":
            tensor_data = torch.ones(*self._eval_shape(args[0]), dtype=tdtype)
        elif func_name == "array":
            return self._convert_list_literal(args[0], target_dtype)
        else:
            raise ValueError(f"Unsupported NumPy function: np.{func_name}()")

        return self._make_tensor_wire(tensor_data, target_dtype)

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

    def _build_binop_term(self, op_type, left_val, right_val) -> Term:
        if op_type is ast.Add:
            return self.builder.add(left_val, right_val)
        elif op_type is ast.Sub:
            return self.builder.sub(left_val, right_val)
        elif op_type is ast.Mult:
            return self.builder.mul(left_val, right_val)
        elif op_type is ast.MatMult:
            return self.builder.matmul(left_val, right_val)
        elif op_type is ast.Div:
            raise ValueError("Division is not supported")
        elif op_type is ast.Pow:
            # Non-linear: routed through the builder so it works once a non-linear
            # theory provides it. In LIA/LRA the builder raises NonLinearError.
            return self.builder.pow(left_val, right_val)
        else:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

    def _convert_binop(self, binop, target_dtype=None):
        """Convert binary operation

        Dtype propagation:
        - With target_dtype: both operands inherit it (e.g., self.x = 2 + 3 -> both Int)
        - Without: right operand inherits from left (e.g., self.x + 3 -> 3 matches x's dtype)
        """
        if target_dtype:
            left_val = self._convert_expr(binop.left, target_dtype=target_dtype)
            right_val = self._convert_expr(binop.right, target_dtype=target_dtype)
        else:
            left_val = self._convert_expr(binop.left)
            right_val = self._convert_expr(binop.right, target_dtype=self._d(left_val))
        term = self._build_binop_term(type(binop.op), left_val, right_val)
        self.terms.append(term)
        return term

    def _apply_binop_aug(self, op_type, left_val, right_val):
        term = self._build_binop_term(op_type, left_val, right_val)
        self.terms.append(term)
        return term

    def _emit_const_bool(self, value: bool) -> Term:
        t = self.builder.const_bool(value)
        self.terms.append(t)
        return t

    def _should_escape_scope(self, k: str) -> bool:
        return k in self.wire_pairs or k.startswith("_ret_")

    def _convert_unaryop(self, unaryop, target_dtype=None):
        """Convert unary operation (not, -, +)

        - not x: Always returns Bool
        - -x: Propagates target_dtype to operand and zero constant
        - +x: No-op, returns operand with target_dtype propagated
        """
        op_type = type(unaryop.op)
        if op_type == ast.Not:
            # not x -> Ite(x, False, True) - always Bool
            operand_val = self._convert_expr(unaryop.operand)
            false_term = self._emit_const_bool(False)
            true_term = self._emit_const_bool(True)
            ite_term = self.builder.ite(operand_val, false_term, true_term)
            self.terms.append(ite_term)
            return ite_term
        elif op_type == ast.USub:
            # -x -> 0 - x
            if target_dtype:
                operand_val = self._convert_expr(
                    unaryop.operand, target_dtype=target_dtype
                )
                zero_term = self.builder.const(
                    torch.tensor([0], dtype=_torch_dtype(target_dtype))
                )
            else:
                operand_val = self._convert_expr(unaryop.operand)
                zero_term = self.builder.const(
                    torch.tensor([0], dtype=_torch_dtype(self._d(operand_val)))
                )
            self.terms.append(zero_term)
            sub_term = self.builder.sub(zero_term, operand_val)
            self.terms.append(sub_term)
            return sub_term
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

        vals = [self._convert_expr(val) for val in boolop.values]
        is_and = isinstance(boolop.op, ast.And)

        if not is_and and not isinstance(boolop.op, ast.Or):
            raise ValueError(
                f"Unsupported boolean operator: {type(boolop.op).__name__}"
            )

        # Build nested Ite from right to left
        result = vals[-1]
        for val in reversed(vals[:-1]):
            false_term = self._emit_const_bool(False)
            true_term = self._emit_const_bool(True)

            if is_and:
                # a and b -> Ite(a, b, False)
                merged = self.builder.ite(val, result, false_term)
            else:
                # a or b -> Ite(a, True, b)
                merged = self.builder.ite(val, true_term, result)
            self.terms.append(merged)
            result = merged
        return result

    def _convert_compare(self, compare):
        """Convert comparison operation

        Handles both simple comparisons (a < b) and chains (a < b < c).
        Chains are expanded: a < b < c becomes (a < b) and (b < c)
        """
        COMPARE_OPS = {
            ast.Eq: self.builder.eq,
            ast.NotEq: self.builder.ne,
            ast.Lt: self.builder.lt,
            ast.LtE: self.builder.le,
            ast.Gt: self.builder.gt,
            ast.GtE: self.builder.ge,
        }
        comparison_vals = []
        left = compare.left

        for op, comparator in zip(compare.ops, compare.comparators):
            left_val = self._convert_expr(left)
            target = None if isinstance(left_val, _StaticValue) else self._d(left_val)
            right_val = self._convert_expr(comparator, target_dtype=target)

            # Static evaluation when both sides are compile-time values (str, None, etc.)
            if isinstance(left_val, _StaticValue) and isinstance(
                right_val, _StaticValue
            ):
                return _StaticValue(_eval_static_compare(left_val.value, op, right_val.value))

            op_type = type(op)
            if op_type not in COMPARE_OPS:
                raise ValueError(f"Unsupported comparison operator: {op_type.__name__}")

            cmp_term = COMPARE_OPS[op_type](left_val, right_val)
            self.terms.append(cmp_term)
            comparison_vals.append(cmp_term)
            left = comparator

        # Combine with AND (single comparison returns directly)
        result = comparison_vals[0]
        for cmp_val in comparison_vals[1:]:
            false_term = self.builder.const_bool(False)
            self.terms.append(false_term)
            merged = self.builder.ite(result, cmp_val, false_term)
            self.terms.append(merged)
            result = merged

        return result

    def _convert_ifexp(self, ifexp, target_dtype=None):
        """Convert ternary conditional to Ite

        Propagates target_dtype to both branches (e.g., self.x = 5 if c else 10 -> both Int)
        """
        cond_val = self._convert_expr(ifexp.test)
        true_val = self._convert_expr(ifexp.body, target_dtype=target_dtype)
        false_val = self._convert_expr(ifexp.orelse, target_dtype=target_dtype)

        ite_term = self.builder.ite(cond_val, true_val, false_val)
        self.terms.append(ite_term)
        return ite_term

    def _convert_name(self, name):
        """Convert variable reference"""
        var_name = name.id
        if var_name in self.temp_vars:
            return self.temp_vars[var_name]
        else:
            raise ValueError(f"Unknown variable: {var_name}")

    def _convert_constant(self, constant, target_dtype=None):
        """Convert scalar constant (bool, int, float) to a constant Term."""
        value = constant.value

        if isinstance(value, bool):
            term = self.builder.const_bool(value)
            self.terms.append(term)
            return term
        elif isinstance(value, (int, float)):
            if target_dtype is None:
                target_dtype = Sort.Real([1, 1])
            tensor_data = torch.tensor([value], dtype=_torch_dtype(target_dtype))
            term = self.builder.const(tensor_data)
            self.terms.append(term)
            return term
        elif value is None or isinstance(value, str):
            return _StaticValue(value)
        else:
            raise ValueError(f"Unsupported constant type: {type(value)}")

    def _convert_list_literal(self, node, target_dtype=None):
        """Convert a list/tuple literal to a wire.

        Constant literals fold to a Tensor. Dynamic lists (elements that are
        wires/subscripts) are routed through builder.stack so they work once a
        theory provides Stack."""
        try:
            data = self._eval_literal(node)
        except ValueError:
            if isinstance(node, (ast.List, ast.Tuple)):
                element_wires = [self._convert_expr(e) for e in node.elts]
                return self.builder.stack(element_wires)
            # np.array(<expr>) on a non-literal (e.g. np.array(self.state)): the arg
            # already denotes the array value, so this is identity.
            return self._convert_expr(node)
        tensor_data = torch.tensor(data, dtype=_torch_dtype(target_dtype))
        return self._make_tensor_wire(tensor_data, target_dtype)

    def _convert_subscript(self, expr, target_dtype=None):
        """x[i] -> TensorGet, routed through the builder so it works once a
        theory provides TensorGet."""
        base_val = self._convert_expr(expr.value)
        index_val = self._convert_expr(expr.slice)
        return self.builder.tensor_get(base_val, index_val)

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

            if wire_name in self.static_attrs:
                return _StaticValue(self.static_attrs[wire_name])

            raise ValueError(f"Unknown wire: {wire_name}")
        else:
            raise ValueError(f"Unsupported attribute access: {ast.unparse(attr)}")

    def _inline_method(self, method_name, args):
        """Inline simple method or dispatch to layer translator."""
        if method_name in self.layers:
            input_val = self._convert_expr(args[0])
            out_features = self.layers[method_name]
            layer = self.live_layers.get(method_name)
            if layer is not None:
                bias = (
                    layer.bias.unsqueeze(1)
                    if layer.bias is not None
                    else torch.zeros(out_features, 0)
                )
                ts = self.builder.linear(input_val, layer.weight, bias)
                self.terms.extend(ts)
                return ts[-1]  # last term's write[0] is the output
            else:
                term, _, _ = self.builder.linear_symbolic(input_val, out_features)
                self.terms.append(term)
                return term

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


def convert_method(
    method,
    wires: dict[str, tuple[Wire, Wire]],
    result: list[Wire],
    cls=None,
    layers: dict[str, int] | None = None,
    params: dict[str, Wire] | None = None,
    live_layers: dict | None = None,
    builder: TermBuilder | None = None,
    theory=None,
    static_attrs: dict[str, Any] | None = None,
) -> list[Term]:
    """Convert a Python method to a list of Terms.

    Args:
        method: Unbound method to convert (e.g. cls.reset, cls.step, cls.forward)
        wires: All named wire pairs available to the method - both method
               parameters (by parameter name) and self.* attributes (by attr name)
        result: Ordered list of next-wires matched positionally to the return tuple
        cls: Class owning the method, needed only for inlining self.helper() calls
        params: Read-only parameter wires (single wires, not pairs) for self.* constants
        live_layers: Optional dict of {layer_name: nn.Linear} for live tensor references
        builder: Optional TermBuilder instance; if provided takes precedence over theory
        theory: IType.LRA (default), IType.LIA, or IType.BV — selects the term builder
                (ignored if builder is provided)

    Returns:
        List of Terms representing the method as a reactive diagram
    """
    from .builder import builder_for

    if builder is None:
        builder = builder_for(theory)

    source = textwrap.dedent(inspect.getsource(method))
    func_def = ast.parse(source).body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError(f"Expected function definition, got {type(func_def).__name__}")

    # Normalize early returns to single-exit form before visiting
    func_def.body = _normalize_early_returns(func_def.body)

    param_names = [arg.arg for arg in func_def.args.args if arg.arg != "self"]
    visitor = MethodVisitor(
        wires,
        result,
        cls=cls,
        layers=layers,
        params=params,
        live_layers=live_layers,
        builder=builder,
        static_attrs=static_attrs,
    )
    visitor.temp_vars.update(
        {name: wires[name][0] for name in param_names if name in wires}
    )

    for stmt in func_def.body:
        visitor.visit(stmt)

    for i, result_wire in enumerate(result):
        src = visitor.temp_vars.get(f"_ret_{i}")
        if src is None:
            raise ValueError(f"Method has no return value for result {i}")
        visitor.terms.append(visitor.builder.id_(src, output_wire=result_wire))

    return visitor.terms
