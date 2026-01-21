from random import randrange

from typing import Any
from torch import Tensor

from .zrth import DType, IType, Term, Wire
from .context import get_ctx


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
        self.op = op
        # FIXME: this special handling of "sym"
        self.args = list(args) if op != "sym" else []

        Expr.__cnt += 1
        self.id = Expr.__cnt
        ctx = get_ctx()

        if op == "sym":
            name, dtype = args
            assert isinstance(name, str), name
            assert isinstance(dtype, DType), (type(dtype), dtype)
            self._out_wire: Wire = ctx.unwrap().wire(name, dtype)
            self._term = None
            self._dtype: DType = dtype
        elif op == "const":
            assert len(args) == 1
            val = args[0]
            itype, dtype = const_to_itype_dtype(val)
            self._out_wire: Wire = ctx.unwrap().tmp_wire(dtype)
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
            self._out_wire: Wire = ctx.unwrap().tmp_wire(dtype)
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

    def __rmatmul__(self, lhs):
        return Expr("arith.matmul", to_expr(lhs), self)

    def __matmul__(self, rhs):
        return Expr("arith.matmul", self, to_expr(rhs))

    def __add__(self, rhs):
        return Expr("arith.add", self, to_expr(rhs))

    def __lt__(self, rhs):
        return Expr("cmp.lt", self, to_expr(rhs))

    def __gt__(self, rhs):
        return Expr("cmp.gt", self, to_expr(rhs))

    def __or__(self, rhs):
        return Expr("logic.or", self, to_expr(rhs))

    def __invert__(self):
        return Expr("logic.not", self)

    def __str__(self) -> str:
        return f"<{self.id}> {self.op}({', '.join(map(str, self.args))})"

    def get_children(self) -> list:
        return self.args


def matmul_dtype(dt1, dt2):
    assert dt1.is_tensor()
    assert dt2.is_tensor()
    dim1 = dt1.dims()
    dim2 = dt2.dims()

    if len(dim2) == 1:
        # tensor @ vector
        if dim1[-1] == dim2[0]:
            return DType.Tensor(dim1[:-1])
        else:
            raise RuntimeError("Unsupported tensor @ vector operation")
    elif len(dim1) == len(dim2) == 2:
        # matrix @ matrix
        if dim1[-1] == dim2[0]:
            return DType.Tensor([dim1[0], dim2[1]])
        else:
            raise RuntimeError(
                f"Unsupported matrix multiplication, dimensions do not match: {dim1} x {dim2}"
            )

    # TODO: allow broadcasting

    raise RuntimeError(
        f"Unsupported/unimplemented tensor matmul operation: '{dim1} @ {dim2}'"
    )


# until we have global itypes with hierarchy (for visitors)
# and terms for variables (if ever), we use strings for representing
# operations, so we have to translate them to IType
# TODO: we do typechecking here, this should be probably somewhere else
# (I mean, we can keep the assertions, but we should do proper typechecking
# somewhere else...)
def op_to_itype_dtype(op: str, args) -> tuple[IType, DType]:
    """
    Translate operation with arguments into itype and the return dtype.
    """
    assert all(isinstance(a, Expr) for a in args), args

    if op == "arith.add":
        assert len(args) == 2
        assert args[0].dtype() == args[1].dtype()
        return IType.Add(), args[0].dtype()

    if op == "arith.matmul":
        assert len(args) == 2
        dtype = matmul_dtype(args[0].dtype(), args[1].dtype())
        return IType.MatMul(), dtype

    if op == "cmp.lt":
        assert len(args) == 2
        assert args[0].dtype() == args[1].dtype()
        return IType.Lt(), DType.Bool()

    if op == "logic.or":
        assert len(args) == 2
        assert args[0].dtype() == args[1].dtype() == DType.Bool()
        return IType.Or(), args[0].dtype()

    if op == "logic.not":
        assert len(args) == 1
        assert args[0].dtype() == DType.Bool()
        return IType.Not(), args[0].dtype()

    if op == "ite":
        assert len(args) == 3
        assert args[0].dtype() == DType.Bool()
        assert args[1].dtype() == args[2].dtype()
        return IType.Ite(), args[1].dtype()

    # if op == "ifthen":
    #     assert len(args) == 2
    #     assert args[0].dtype() == DType.bool()
    #     return IType.IfThen(), args[1].dtype()
    #
    # if op == "choose":
    #     assert len(args) > 0
    #     assert all(a.dtype() == args[0].dtype() for a in args)
    #     return IType.mk_choose(), args[0].dtype()
    #
    # if op == "choose_or":
    #     assert len(args) > 0
    #     assert all(a.dtype() == args[0].dtype() for a in args)
    #     return IType.mk_choose(), args[0].dtype()

    raise NotImplementedError(f"Translation not implemented for {op}")


def const_to_itype_dtype(val) -> tuple[IType, DType]:
    """
    Translate operation with arguments into itype and the return dtype.
    """
    if isinstance(val, (int, float, bool)):
        return IType.Tensor(Tensor([val])), DType.Tensor([1])

    if isinstance(val, Tensor):
        return IType.Tensor(val), DType.Tensor(val.size())

    raise NotImplementedError(f"Unimplemented constant: {val} ({type(val)})")


class Sym(Expr):
    """
    An expression representing a Symbol
    """

    def __init__(self, name, dtype, create_pair=True):
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


def to_expr(val: Any) -> Expr:
    if isinstance(val, Expr):
        return val
    if isinstance(val, (int, bool, float, Tensor)):
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


def input_sym(name: str, ty: DType) -> Sym:
    s = Sym(name, ty, create_pair=True)
    if create_pair:
        return (s, s.nxt())
    return s


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


def ite(cond, iftrue, iffalse):
    """
    Implement if-then-else construct for straght-line code:
    if cond is concrete Python bool or int, `ite` is evaluated
    as Python's if-else block. Otherwise an expression is created.
    """
    if any(isinstance(val, Expr) for val in (cond, iftrue, iffalse)):
        return Ite(cond, iftrue, iffalse)

    return iftrue if cond else iffalse


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
