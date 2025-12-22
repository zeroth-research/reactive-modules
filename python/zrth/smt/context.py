from pysmt.shortcuts import Symbol, Int, Real, Bool, Plus
from pysmt.typing import INT, REAL, BOOL
from pysmt.fnode import FNode as Expr
import pysmt.operators as op

from zrth import _zrth
from zrth.context import Context as ContextBase

from typing import Callable, Any

import inspect

from itertools import chain

PyVal = _zrth.PyVal
WrappedModule = _zrth.smt.WrappedModule
WrappedTerm = _zrth.smt.WrappedTerm


def handle_return_value(r):
    """
    Transform the return value into a list
    """

    if r is None:
        r = []
    elif isinstance(r, tuple):
        r = [x for x in r]
    elif not isinstance(r, list):
        r = [r]

    # return [x.var if isinstance(x, Assignment) else x for x in r]
    return r


def to_pysmt_type(ty: str):
    if ty == "Int":
        return INT
    if ty == "Real":
        return REAL
    if ty == "Bool":
        return BOOL
    raise NotImplementedError(f"Unknown type: {ty}")


def from_pysmt_type(ty) -> str:
    if ty == INT:
        return "Int"
    if ty == REAL:
        return "Real"
    if ty == BOOL:
        return "Bool"
    raise NotImplementedError(f"Unknown type: {ty}")


def nxt(var: Symbol) -> Symbol:
    """
    Get next variable for `var`.
    """
    assert isinstance(var, Expr), var
    return Symbol(f"nxt({var.symbol_name()})", var.symbol_type())


class PySMTContext(ContextBase):
    def __init__(self, ctx_impl):
        super().__init__(ctx_impl)

    def var(self, name: str, ty: str) -> Symbol:
        return Symbol(name, to_pysmt_type(ty))

    @staticmethod
    def next_var(var: Symbol) -> Symbol:
        """
        Get next variable for `var`.
        """
        return nxt(var)

    def _parse_variables(self, ctrl, extl):
        if isinstance(ctrl, str):
            V = ctrl.split(",")
            if not all(":" in v for v in V):
                raise RuntimeError("A variable is missing a type annotation")
            ctrl = tuple(
                self.var(v[0].strip(), v[1].strip())
                for v in map(lambda s: s.split(":"), V)
            )
            cur_vars = [*ctrl]
        elif isinstance(ctrl, (tuple, list)) and len(ctrl) > 0:
            if isinstance(ctrl[0], str):
                if not all(":" in v for v in ctrl):
                    raise RuntimeError(
                        "A variable is missing a type annotation")
                ctrl = tuple(
                    self.var(v[0].strip(), v[1].strip())
                    for v in map(lambda s: s.split(":"), ctrl)
                )
            elif isinstance(ctrl[0], Expr):
                if not all(isinstance(c, Expr) and c.is_symbol() for c in ctrl):
                    raise RuntimeError(
                        f"Expected variables to be all PySMT variables, got: {
                            ctrl}"
                    )
            else:
                raise RuntimeError(
                    f"Expected variables to be a tuple of strings or PySMT variables, got: {
                        ctrl}"
                )
            cur_vars = [*ctrl]
        else:
            raise RuntimeError(
                f"Expect variables to be a non-empty string, tuple or list, got: {
                    ctrl}"
            )

        if isinstance(extl, str):
            V = extl.split(",")
            if not all(":" in v for v in V):
                raise RuntimeError("A variable is missing a type annotation")
            extl = tuple(
                self.var(v[0].strip(), v[1].strip())
                for v in map(lambda s: s.split(":"), V)
            )
        elif isinstance(extl, (tuple, list)) and len(extl) > 0:
            if isinstance(extl[0], str):
                if not all(":" in v for v in extl):
                    raise RuntimeError(
                        "A variable is missing a type annotation")
                extl = tuple(
                    self.var(v[0].strip(), v[1].strip())
                    for v in map(lambda s: s.split(":"), extl)
                )
            elif isinstance(extl[0], Expr):
                if not all(isinstance(c, Expr) and c.is_symbol() for c in extl):
                    raise RuntimeError(
                        f"Expected variables to be all PySMT variables, got: {
                            extl}"
                    )
            else:
                raise RuntimeError(
                    f"Expected variables to be a tuple of strings or PySMT variables, got: {
                        extl}"
                )
            cur_vars.extend(extl)
        else:
            raise RuntimeError(
                f"Expect variables to be a string, tuple or list, got: {extl}"
            )

        return ctrl, extl, cur_vars

    def trace(self, fun: Callable, *args, **kwargs):
        """
        Execute a given function with binding our names like `next` into it.
        """
        return handle_return_value(fun(*args, **kwargs))

    def get_pyval_sym(self, sym: Expr) -> PyVal:
        assert sym.is_symbol(), sym

        return self._context.get_sym(
            sym.symbol_name(), from_pysmt_type(sym.symbol_type())
        )


class Context(PySMTContext):
    def __init__(self):
        super().__init__(_zrth.smt.WrappedContext())

    def module_from_methods(
        self,
        ctrl: str | tuple[str],
        extl: str | tuple[str],
        init: Callable[[], None],
        update: Callable[[], None],
        name: str | None = None,
    ) -> WrappedModule:
        """
        Create the Rust module (:class:`WrappedModule`) from the `init` and `update` methods.
        """

        ctrl, extl, cur_vars = self._parse_variables(ctrl, extl)

        # if the user uses a single variable, it is more natural in the `init` and `update` to unwrap it
        ctrl_arg = ctrl[0] if len(ctrl) == 1 else ctrl
        extl_arg = extl[0] if len(extl) == 1 else extl

        extl_nxt = [self.next_var(v) for v in extl]
        # extl_nxt_arg = extl_nxt[0] if len(extl_nxt) == 1 else extl_nxt

        # trace the init function
        if init:
            init_ret = self.trace(init, extl_arg)
            assert len(init_ret) == len(ctrl)
        else:
            init_terms = []

        # trace the update function
        update_ret = self.trace(update, ctrl_arg, extl_arg)
        assert len(update_ret) == len(ctrl)

        cur_vars, nxt_vars, init_terms, update_terms = self.to_terms(
            ctrl, extl, init_ret, update_ret
        )

        module = WrappedModule(
            self._context, cur_vars, nxt_vars, init_terms, update_terms
        )
        # if name is not None:
        #    module.set_name(name)
        return module

    def _cond(self, cnd, iftrue, iffalse):
        return Cond(args)

    def to_terms(self, ctrl, extl, init_ret, update_ret) -> (list, list, list, list):
        """
        MODIFIES `self._context`
        """
        walker = ToTerms(self.unwrap())

        init_terms = []
        assert len(init_ret) == len(ctrl)
        for var, expr in zip(ctrl, init_ret):
            tmp, outvar = walker.translate(expr)
            init_terms.extend(tmp)

            # map the output of the expression to the output wire
            assert len(outvar) == 1
            init_terms.append(
                WrappedTerm("Id", outvar, [
                            self.get_pyval_sym(self.next_var(var))])
            )

        update_terms = []
        assert len(update_ret) == len(ctrl)
        for var, expr in zip(ctrl, update_ret):
            tmp, outvar = walker.translate(expr)
            update_terms.extend(tmp)

            # map the output of the expression to the output wire
            assert len(outvar) == 1
            update_terms.append(
                WrappedTerm("Id", outvar, [
                            self.get_pyval_sym(self.next_var(var))])
            )

        cur_vars = [self.get_pyval_sym(v) for v in chain(ctrl, extl)]
        nxt_vars = [self.get_pyval_sym(self.next_var(v))
                    for v in chain(ctrl, extl)]
        return cur_vars, nxt_vars, init_terms, update_terms


# NOTE: we want to handle also Choose objects and therefore we have to
# traverse the expression ourselves, without using pysmt.walkers.DagWalker
class ToTerms:
    def __init__(self, ctx) -> None:
        """
        MODIFIES `ctx`
        """
        self._ctx = ctx
        self.terms = None

    def translate(self, formula):
        """
        Translate a formula into reactive module terms.
        Return the terms and the output variable (wire) for the formula.
        """
        self.terms = []
        r = self.walk(formula)
        return self.terms, r

    def walk(self, formula, depth=0):

        # constants
        # NOTE: check for bool must be before int (bool is an instance of int in Python...)
        if isinstance(formula, bool):
            return [PyVal.bool(formula)]
        elif isinstance(formula, int):
            return [PyVal.int(formula)]

        # print(" " * depth, "Visiting", formula)

        args = []
        for child in formula.args():
            args.extend(self.walk(child, depth + 1))

        terms = self.terms

        assert all(isinstance(a, PyVal) for a in args), args
        opty = formula.node_type()
        if opty == op.SYMBOL:
            return [
                self._ctx.get_sym(
                    formula.symbol_name(), from_pysmt_type(formula.symbol_type())
                )
            ]
        elif opty == op.INT_CONSTANT:
            return [PyVal.int(formula.constant_value())]
        elif opty == op.REAL_CONSTANT:
            return [PyVal.real(formula.constant_value())]
        elif opty == op.BOOL_CONSTANT:
            return [PyVal.bool(formula.constant_value())]
        elif opty == op.PLUS:
            assert len(args) == 2
            assert args[0].dtype() == args[1].dtype(), args
            out = self._ctx.tmp_sym(args[0].dtype())
            terms.append(WrappedTerm("Arith::Add", reads=args, writes=[out]))
            return [out]
        elif opty == op.MINUS:
            assert len(args) == 2
            assert args[0].dtype() == args[1].dtype(), args
            out = self._ctx.tmp_sym(args[0].dtype())
            terms.append(WrappedTerm("Arith::Sub", reads=args, writes=[out]))
            return [out]
        elif opty == op.TIMES:
            assert len(args) == 2
            assert args[0].dtype() == args[1].dtype(), args
            out = self._ctx.tmp_sym(args[0].dtype())
            terms.append(WrappedTerm("Arith::Mul", reads=args, writes=[out]))
            return [out]
        elif opty == op.DIV:
            assert len(args) == 2
            assert args[0].dtype() == args[1].dtype(), args
            out = self._ctx.tmp_sym(args[0].dtype())
            terms.append(WrappedTerm("Arith::Div", reads=args, writes=[out]))
            return [out]
        elif opty == op.NOT:
            assert len(args) == 1
            out = self._ctx.tmp_sym("Bool")
            terms.append(WrappedTerm("Logical::Not", reads=args, writes=[out]))
            return [out]
        elif opty == op.AND:
            assert len(args) == 2
            out = self._ctx.tmp_sym("Bool")
            terms.append(WrappedTerm("Logical::And", reads=args, writes=[out]))
            return [out]
        elif opty == op.OR:
            assert len(args) == 2
            out = self._ctx.tmp_sym("Bool")
            terms.append(WrappedTerm("Logical::Or", reads=args, writes=[out]))
            return [out]
        elif opty == op.EQUALS:
            assert len(args) == 2
            out = self._ctx.tmp_sym("Bool")
            terms.append(WrappedTerm("Cmp::Eq", reads=args, writes=[out]))
            return [out]
        elif opty == op.LT:
            assert len(args) == 2
            out = self._ctx.tmp_sym("Bool")
            terms.append(WrappedTerm("Cmp::Lt", reads=args, writes=[out]))
            return [out]
        elif opty == op.LE:
            assert len(args) == 2
            out = self._ctx.tmp_sym("Bool")
            terms.append(WrappedTerm("Cmp::Le", reads=args, writes=[out]))
            return [out]
        elif opty == op.ITE:
            assert len(args) == 3
            assert args[1].dtype() == args[2].dtype(), args
            out = self._ctx.tmp_sym(args[1].dtype())
            terms.append(WrappedTerm("Cond", reads=args, writes=[out]))
            return [out]
        else:
            raise NotImplementedError(
                f"Not implemented translation of operation: {
                    formula} {type(formula)}"
            )

        raise RuntimeError("Unreachable")
