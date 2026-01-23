from random import randrange

from typing import Any, TypeAlias
import torch

from .zrth import DType, IType, Term, Wire
from .context import get_ctx


# types that we can convert to [Expr]
type ToExpr = Expr | int | bool | float | torch.Tensor


class Expr:
    """
    A minimalistic representation of an expression.

    We use this class during translating Python code into reactive modules terms.
    It is nothing else than a symbolic representation of a computation of a value:
    an `Expr` is basically a node in an abstract-syntax tree. As such, it has
    some attributes and *children*.

    We deliberately use strings for operations so that this class
    is as flexible as possible, it is temporary until we have a global IType.

    :param op: name of the operation of the expression
    :param ty: a string representing the type of the expression
    :param args: a list of expression arguments (children)
    """

    __cnt = 0

    def __init__(self, op: str, *args):
        assert isinstance(op, str), type(op)
        self.op = op
        # FIXME: this special handling of "sym"
        self.args = list(args) if op != "sym" else []

        Expr.__cnt += 1
        self.id = Expr.__cnt
        ctx = get_ctx()

        assert all(not isinstance(a, Expr) or a.ctx() is ctx for a in args), (
            "Expr must be created in the same context as its arguments"
        )

        if op == "sym":
            name, dtype = args
            assert isinstance(name, str), name
            assert isinstance(dtype, DType), (type(dtype), dtype)
            self._out_wire: Wire = ctx.wire(name, dtype)
            self._term = None
            self._dtype: DType = dtype
        elif op == "const":
            assert len(args) == 1
            val = args[0]
            itype, dtype = const_to_itype_dtype(val)
            self._out_wire: Wire = ctx.tmp_wire(dtype)
            self._term: Term = Term(itype, [self._out_wire], [])
            self._dtype: DType = dtype
        elif op == "assign":  # FIXME: do not have this as a special case
            assert len(args) == 2
            dtype = args[0].dtype()
            assert dtype == args[1].dtype()
            self._out_wire: Wire = args[1].wire()
            self._term: Term = Term(IType.Id(), [self._out_wire], [args[0].wire()])
            self._dtype: DType = dtype
        else:
            itype, dtype = op_to_itype_dtype(op, args)
            self._out_wire: Wire = ctx.tmp_wire(dtype)
            self._term: Term = Term(itype, [self._out_wire], [a.wire() for a in args])
            self._dtype: DType = dtype

        # add the new term to the context
        if self._term:
            ctx.add_term(self._term)

        # Store the context this expression was created in.
        # At the moment, it is rather for checking consistency
        # (expressions used together should be created together)
        self._ctx = ctx

    def ctx(self):
        return self._ctx

    def term(self):
        return self._term

    def dtype(self):
        return self._dtype

    def wire(self):
        """
        Get the output wire of this expression's term.
        """
        return self._out_wire

    def __eq__(self, oth):
        return (
            isinstance(self, Expr)
            and self.op == oth.op
            and self.get_children() == oth.get_children()
        )

    ###
    # Operations on this expression
    ###
    def matmul(self, rhs: ToExpr) -> "Expr":
        return MatMul(self, rhs)

    def add(self, rhs: ToExpr) -> "Expr":
        return Add(self, rhs)

    def mul(self, rhs: ToExpr) -> "Expr":
        return Mul(self, rhs)

    def sub(self, rhs: ToExpr) -> "Expr":
        return Sub(self, rhs)

    def div(self, rhs: ToExpr) -> "Expr":
        return Div(self, rhs)

    def eq(self, rhs: ToExpr) -> "Expr":
        return Eq(self, rhs)

    def neq(self, rhs: ToExpr) -> "Expr":
        return Neq(self, rhs)

    def lt(self, rhs: ToExpr) -> "Expr":
        return Lt(self, rhs)

    def gt(self, rhs: ToExpr) -> "Expr":
        return Gt(self, rhs)

    def le(self, rhs: ToExpr) -> "Expr":
        return Le(self, rhs)

    def ge(self, rhs: ToExpr) -> "Expr":
        return Ge(self, rhs)

    def land(self, rhs: ToExpr) -> "Expr":
        return And(self, rhs)

    def lor(self, rhs: ToExpr) -> "Expr":
        return Or(self, rhs)

    def lnot(self) -> "Expr":
        return Not(self)

    def ite(self, iftrue: ToExpr, iffalse: ToExpr) -> "Expr":
        return Ite(self, iftrue, iffalse)

    ###
    # Operators
    ###
    def __rmatmul__(self, lhs: ToExpr) -> "Expr":
        return MatMul(lhs, self)

    def __matmul__(self, rhs: ToExpr) -> "Expr":
        return self.matmul(rhs)

    def __add__(self, rhs: ToExpr) -> "Expr":
        return self.add(rhs)

    def __mul__(self, rhs: ToExpr) -> "Expr":
        return self.mul(rhs)

    def __lt__(self, rhs: ToExpr) -> "Expr":
        return Lt(self, rhs)

    def __gt__(self, rhs: ToExpr) -> "Expr":
        return Gt(self, rhs)

    def __or__(self, rhs: ToExpr) -> "Expr":
        return Or(self, rhs)

    def __invert__(self) -> "Expr":
        return Not(self)

    def __str__(self) -> str:
        return f"<{self.id}> {self.op}({', '.join(map(str, self.args))})"

    def get_children(self) -> list:
        return self.args


def matmul_dtype(dt1: DType, dt2: DType) -> DType:
    assert dt1.eq_dtype(dt2), f"DTypes have different element types: {dt1}, {dt2}"

    dim1 = dt1.dims()
    dim2 = dt2.dims()

    if len(dim2) == 1:
        # tensor @ vector
        if dim1[-1] == dim2[0]:
            return type(dt1)(dim1[:-1])
        else:
            raise RuntimeError("Unsupported tensor @ vector operation")
    elif len(dim1) == len(dim2) == 2:
        # matrix @ matrix
        if dim1[-1] == dim2[0]:
            return type(dt1)([dim1[0], dim2[1]])
        else:
            raise RuntimeError(
                f"Unsupported matrix multiplication, dimensions do not match: {dim1} x {dim2}"
            )

    # TODO: allow broadcasting

    raise RuntimeError(
        f"Unsupported/unimplemented tensor matmul operation: '{dim1} @ {dim2}'"
    )


def arith_op_to_itype_dtype(op: str, args: tuple[ToExpr]) -> tuple[IType, DType]:
    assert len(args) == 2

    if op == "arith.matmul":
        dtype = matmul_dtype(args[0].dtype(), args[1].dtype())
        return IType.MatMul(), dtype

    if op == "arith.add":
        assert args[0].dtype() == args[1].dtype(), (args[0].dtype(), args[1].dtype())
        return IType.Add(), args[0].dtype()

    if op == "arith.mul":
        assert args[0].dtype() == args[1].dtype(), (args[0].dtype(), args[1].dtype())
        return IType.Mul(), args[0].dtype()

    if op == "arith.sub":
        assert args[0].dtype() == args[1].dtype(), (args[0].dtype(), args[1].dtype())
        return IType.Sub(), args[0].dtype()

    if op == "arith.div":
        assert args[0].dtype() == args[1].dtype(), (args[0].dtype(), args[1].dtype())
        return IType.Div(), args[0].dtype()

    raise NotImplementedError(f"Unsupported arith op: {op}")


def cmp_op_to_itype_dtype(op: str, args: tuple[ToExpr]) -> tuple[IType, DType]:
    assert len(args) == 2
    assert args[0].dtype() == args[1].dtype()

    if op == "cmp.lt":
        return IType.Lt(), DType.Bool

    if op == "cmp.gt":
        return IType.Gt(), DType.Bool

    if op == "cmp.le":
        return IType.Le(), DType.Bool

    if op == "cmp.ge":
        return IType.Ge(), DType.Bool

    if op == "cmp.eq":
        return IType.Eq(), DType.Bool

    if op == "cmp.neq":
        return IType.Neq(), DType.Bool

    raise NotImplementedError(f"Unsupported cmp op: {op}")


def logic_op_to_itype_dtype(op: str, args: tuple[ToExpr]) -> tuple[IType, DType]:
    if op == "logic.not":
        assert len(args) == 1
        assert args[0].dtype() == DType.Bool
        return IType.Not(), DType.Bool

    assert len(args) == 2
    assert args[0].dtype() == args[1].dtype() == DType.Bool

    if op == "logic.or":
        return IType.Or(), DType.Bool

    if op == "logic.and":
        return IType.And(), DType.Bool

    raise NotImplementedError(f"Unsupported logic op: {op}")


# until we have global itypes with hierarchy (for visitors)
# and terms for variables (if ever), we use strings for representing
# operations, so we have to translate them to IType
# TODO: we do typechecking here, this should be probably somewhere else
# (I mean, we can keep the assertions, but we should do proper typechecking
# somewhere else...)
def op_to_itype_dtype(op: str, args: tuple[ToExpr]) -> tuple[IType, DType]:
    """
    Translate operation with arguments into itype and the return dtype.
    """
    assert all(isinstance(a, Expr) for a in args), args

    if op.startswith("arith."):
        return arith_op_to_itype_dtype(op, args)

    if op.startswith("cmp."):
        return cmp_op_to_itype_dtype(op, args)

    if op.startswith("logic."):
        return logic_op_to_itype_dtype(op, args)

    if op == "ite":
        assert len(args) == 3
        assert args[0].dtype() == DType.Bool
        assert args[1].dtype() == args[2].dtype()
        return IType.Ite(), args[1].dtype()

    if op == "id":
        assert len(args) == 1
        return IType.Id(), args[0].dtype()

    raise NotImplementedError(f"Translation not implemented for {op}")


def const_to_itype_dtype(val) -> tuple[IType, DType]:
    """
    Translate operation with arguments into itype and the return dtype.
    """
    if isinstance(val, bool):
        return IType.Tensor(torch.Tensor([val])), DType.Bool
    if isinstance(val, float):
        return IType.Tensor(torch.Tensor([val])), DType.Float
    if isinstance(val, int):
        return IType.Tensor(torch.Tensor([val])), DType.Int

    if isinstance(val, torch.Tensor):
        dtype = val.dtype

        if dtype == torch.bool:
            return IType.Tensor(val), DType.TensorBool(val.size())

        if dtype in (
            torch.int,
            torch.long,
            torch.uint64,
            torch.uint32,
            torch.uint8,
            torch.uint16,
            torch.short,
        ):
            return IType.Tensor(val), DType.TensorInt(val.size())

        if dtype in (torch.float, torch.float32, torch.float64):
            return IType.Tensor(val), DType.TensorFloat(val.size())

        raise NotImplementedError(f"Unsupported tensor element type: {dtype}")

    raise NotImplementedError(f"Unimplemented constant: {val} ({type(val)})")


class Sym(Expr):
    """
    An expression representing a Symbol
    """

    def __init__(self, name, dtype, create_pair=True):
        if isinstance(dtype, str):
            dtype = DType.from_str(dtype)
        super().__init__("sym", name, dtype)
        self._name = name
        if create_pair:
            self._nxt = Sym(f"{name}'", dtype, create_pair=False)
        else:
            self._nxt = None

    def nxt(self):
        if self._nxt is None:
            raise RuntimeError(f"Symbol {self._name} has no next symbol associated")

        return self._nxt

    def __getitem__(self, item: int) -> "Sym":
        if item == 0:
            return self
        if item == 1:
            return self.nxt()
        else:
            raise RuntimeError("Symbol has only two items: 0 (latched) and 1 (next)")

    def __hash__(self) -> int:
        return hash(self._name)

    @property
    def name(self):
        return self._name

    def fresh(self, name: str) -> "Sym":
        """
        Create a symbol of the same type with a new name
        """
        return Sym(name, self.dtype(), create_pair=(self._nxt is not None))

    def __str__(self):
        return f"Sym({self._name} : {self.dtype()})"


def to_expr(val: ToExpr) -> Expr:
    if isinstance(val, Expr):
        return val
    if isinstance(val, (int, bool, float, torch.Tensor)):
        return Expr("const", val)
    raise NotImplementedError(f"Cannot covert object to Expr: {val} ({type(val)})")


def nxt(var: Sym) -> Sym:
    """
    Get next variable for `var`.
    """
    assert isinstance(var, Sym), var
    return var.nxt()


def sym(name: str, ty: DType, create_pair=True) -> Sym:
    return Sym(name, ty, create_pair=create_pair)
    # s = Sym(name, ty, create_pair=create_pair)
    # if create_pair:
    #     return (s, s.nxt())
    # return s


# class IfThen:
#     """
#     An object representing `IfThen` term.
#     """
#
#     def __init__(self, cond: Expr | bool, expr: Any):
#         pass
#
#     def cond(self):
#         pass
#
#     def expr(self):
#         pass
#
#     def is_concrete(self) -> bool:
#         pass
#
#
# class IfThenExpr(IfThen, Expr):
#     """
#     An expression representing `IfThen` term.
#
#     This expression is used inside `Choose` terms.
#     """
#
#     def __init__(self, cond: Expr | bool, expr: Any):
#         Expr.__init__(self, "ifthen", cond, expr)
#
#     def cond(self):
#         return self.get_children()[0]
#
#     def expr(self):
#         return self.get_children()[1]
#
#     def is_concrete(self) -> bool:
#         return False
#
#
# class IfThenConcrete(IfThen):
#     """
#     An expression representing `IfThen` term.
#
#     This expression is used inside `Choose` terms.
#     """
#
#     def __init__(self, cond: bool, expr: Any):
#         self._cond = cond
#         self._expr = expr
#
#     def cond(self):
#         return self._cond
#
#     def expr(self):
#         return self._expr
#
#     def is_concrete(self) -> bool:
#         return True
#
#
# def ifthen(cond, act):
#     if isinstance(cond, Expr) or isinstance(act, Expr):
#         return IfThenExpr(cond, act)
#     return IfThenConcrete(cond, act)
#

# def _choose(alist):
#     assert all(isinstance(a, IfThen) for a in alist), alist
#     if all(a.is_concrete() for a in alist):
#         # execute choose concretely
#         sat_args = [arg for arg in alist if arg.cond()]
#         if sat_args:
#             return sat_args[randrange(len(sat_args))].expr()
#
#         # return None
#         raise RuntimeError("No satisfiable branch in a choose statement")
#
#     return Expr("choose", *alist)
#
#
# def _choose_or(alist):
#     choices = alist[:-1]
#     last = alist[-1]
#
#     # the last argument may not be `IfThen`, in which case
#     # it is the default (unconditional) argument
#     assert not isinstance(last, IfThen), alist
#     assert all(isinstance(a, IfThen) for a in choices), alist
#
#     if not isinstance(last, Expr) and all(a.is_concrete() for a in choices):
#         # execute choose_or concretely
#         sat_args = [arg for arg in choices if arg.cond()]
#         if sat_args:
#             return sat_args[randrange(len(sat_args))].expr()
#         return last
#
#     return Expr("choose_or", *alist)
#
#
# def choose(*args):
#     """
#     Implementation of choose construct.
#     """
#     alist = list(args)
#     if isinstance(alist[-1], IfThen):
#         return _choose(alist)
#     else:
#         return _choose_or(alist)
#


###
# Transform an expression using a visitor pattern
class Transform:
    def transform(self, formula):
        return self._visit(formula)

    def default(self, expr: Expr, args: list):
        return [expr]

    def _visit(self, expr: Expr, depth=0):
        # print(" " * depth, "Visiting", expr)

        before_all = getattr(self, "before_all", None)

        translated_args = []
        if isinstance(expr, Expr):
            for child in expr.get_children():
                translated_args.extend(self._visit(child, depth + 1))

        if before_all is not None:
            before_all(expr, translated_args)

        # this is an expression, recur into the children
        if isinstance(expr, Expr):
            # operations are named 'group.group.op' (with arbitrary many groups).
            # Find the most specific handler by checking methods
            #  visit_group_group_op
            #  visit_group_group
            #  visit_group
            names = expr.op.split(".")
            op = []
            # find the most generic method the transformer has
            while names:
                method = getattr(self, f"visit_{'_'.join(names)}", None)
                if method:
                    break
                op.append(names[-1])
                names = names[:-1]

            if method:
                if op:
                    # partially qualified name, add the "op" parameter
                    op = reversed(op)
                    return method(expr, translated_args, ".".join(op))
                # fully qualified name, do not add the "op" parameter
                return method(expr, translated_args)

        else:
            method = getattr(self, f"visit_type_{type(expr)}", None)
            if method is None:
                # try a generic type visitor
                method = getattr(self, "visit_type", None)
            if method:
                return method(expr, translated_args)

        if method is None:
            return self.default(expr, translated_args)


#####
# Expr classes for nicer expression construction
#####


class Ite(Expr):
    """
    An expression representing `Ite` term.
    """

    def __init__(self, cond: Expr | bool, iftrue: Any, iffalse: Any):
        Expr.__init__(self, "ite", cond, iftrue, iffalse)

    def cond(self):
        return self.get_children()[0]

    def if_true(self):
        return self.get_children()[1]

    def if_false(self):
        return self.get_children()[2]


class BinaryExpr(Expr):
    def __init__(self, op, lhs: ToExpr, rhs: ToExpr):
        super().__init__(op, to_expr(lhs), to_expr(rhs))

    @property
    def lhs(self):
        return self.get_children()[0]

    @property
    def rhs(self):
        return self.get_children()[1]


class MatMul(BinaryExpr):
    def __init__(self, lhs: ToExpr, rhs: ToExpr):
        super().__init__("arith.matmul", lhs, rhs)


class Add(BinaryExpr):
    def __init__(self, lhs: ToExpr, rhs: ToExpr):
        super().__init__("arith.add", lhs, rhs)


class Sub(BinaryExpr):
    def __init__(self, lhs: ToExpr, rhs: ToExpr):
        super().__init__("arith.sub", lhs, rhs)


class Mul(BinaryExpr):
    def __init__(self, lhs: ToExpr, rhs: ToExpr):
        super().__init__("arith.mul", lhs, rhs)


class Div(BinaryExpr):
    def __init__(self, lhs: ToExpr, rhs: ToExpr):
        super().__init__("arith.div", lhs, rhs)


class Lt(BinaryExpr):
    def __init__(self, lhs: ToExpr, rhs: ToExpr):
        super().__init__("cmp.lt", lhs, rhs)


class Gt(BinaryExpr):
    def __init__(self, lhs: ToExpr, rhs: ToExpr):
        super().__init__("cmp.gt", lhs, rhs)


class Le(BinaryExpr):
    def __init__(self, lhs: ToExpr, rhs: ToExpr):
        super().__init__("cmp.le", lhs, rhs)


class Ge(BinaryExpr):
    def __init__(self, lhs: ToExpr, rhs: ToExpr):
        super().__init__("cmp.ge", lhs, rhs)


class Eq(BinaryExpr):
    def __init__(self, lhs: ToExpr, rhs: ToExpr):
        super().__init__("cmp.eq", lhs, rhs)


class Neq(BinaryExpr):
    def __init__(self, lhs: ToExpr, rhs: ToExpr):
        super().__init__("cmp.neq", lhs, rhs)


class Or(BinaryExpr):
    def __init__(self, lhs: ToExpr, rhs: ToExpr):
        super().__init__("logic.or", lhs, rhs)


class And(BinaryExpr):
    def __init__(self, lhs: ToExpr, rhs: ToExpr):
        super().__init__("logic.and", lhs, rhs)


class Not(Expr):
    def __init__(self, expr: ToExpr):
        super().__init__("logic.not", to_expr(expr))


class Id(Expr):
    def __init__(self, e: ToExpr):
        super().__init__("id", e)


#####
# Helper functions to create expressions or evaluate the values concretely
#####
def ite(cond, iftrue, iffalse):
    """
    Implement if-then-else construct for straght-line code:
    if cond is concrete Python bool or int, `ite` is evaluated
    as Python's if-else block. Otherwise an expression is created.
    """
    if any(isinstance(val, Expr) for val in (cond, iftrue, iffalse)):
        return Ite(cond, iftrue, iffalse)

    return iftrue if cond else iffalse


def matmul(lhs: ToExpr, rhs: ToExpr) -> MatMul | torch.Tensor:
    if isinstance(lhs, Expr) or isinstance(rhs, Expr):
        return MatMul(lhs, rhs)
    return lhs @ rhs


def add(lhs, rhs) -> Add | int | float:
    if isinstance(lhs, Expr) or isinstance(rhs, Expr):
        return Add(lhs, rhs)
    return lhs + rhs


def mul(lhs: ToExpr, rhs: ToExpr) -> Mul | torch.Tensor | float | int:
    if isinstance(lhs, Expr) or isinstance(rhs, Expr):
        return Mul(lhs, rhs)
    return lhs * rhs


def div(lhs: ToExpr, rhs: ToExpr) -> Div | torch.Tensor | float | int:
    if isinstance(lhs, Expr) or isinstance(rhs, Expr):
        return Div(lhs, rhs)
    return lhs / rhs


def sub(lhs: ToExpr, rhs: ToExpr) -> Sub | torch.Tensor | float | int:
    if isinstance(lhs, Expr) or isinstance(rhs, Expr):
        return Sub(lhs, rhs)
    return lhs - rhs


def lt(lhs, rhs) -> Lt | bool:
    if isinstance(lhs, Expr) or isinstance(rhs, Expr):
        return Lt(lhs, rhs)
    return lhs < rhs


def gt(lhs, rhs) -> Gt | bool:
    if isinstance(lhs, Expr) or isinstance(rhs, Expr):
        return Gt(lhs, rhs)
    return lhs > rhs


def ge(lhs, rhs) -> Ge | bool:
    if isinstance(lhs, Expr) or isinstance(rhs, Expr):
        return Ge(lhs, rhs)
    return lhs >= rhs


def le(lhs, rhs) -> Le | bool:
    if isinstance(lhs, Expr) or isinstance(rhs, Expr):
        return Le(lhs, rhs)
    return lhs <= rhs


def eq(lhs, rhs) -> Eq | bool:
    if isinstance(lhs, Expr) or isinstance(rhs, Expr):
        return Eq(lhs, rhs)
    return lhs == rhs


def neq(lhs, rhs) -> Neq | bool:
    if isinstance(lhs, Expr) or isinstance(rhs, Expr):
        return Neq(lhs, rhs)
    return lhs != rhs


# Logical or (we have to avoid clash with Python keyword 'or')
def lor(lhs, rhs) -> Or | bool:
    if isinstance(lhs, Expr) or isinstance(rhs, Expr):
        return Or(lhs, rhs)
    return lhs or rhs


# Logical and
def land(lhs, rhs) -> And | bool:
    if isinstance(lhs, Expr) or isinstance(rhs, Expr):
        return And(lhs, rhs)
    return lhs and rhs


# Logical not
def lnot(e) -> And | bool:
    if isinstance(e, Expr):
        return Not(e)
    return not e


def const(x: int | bool | float | torch.Tensor) -> Expr:
    return to_expr(x)
