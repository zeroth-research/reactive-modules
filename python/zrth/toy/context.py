from pysmt.shortcuts import Symbol, Int, Real, Bool, Plus
from pysmt.typing import INT, REAL, BOOL
from pysmt.fnode import FNode as Expr
import pysmt.operators as op

from zrth import _zrth
from zrth.context import Context as ContextBase

from typing import Callable

import inspect

from itertools import chain

PyVal = _zrth.PyVal
WrappedModule = _zrth.toy.WrappedModule
WrappedTerm = _zrth.toy.WrappedTerm


class Choose:
    def __init__(self, reads: list):
        self.reads: list = reads

    def args(self):
        return self.reads


class Case:
    def __init__(self, cond, result):
        assert isinstance(cond, (bool, Expr, Choose)), cond
        self.cond = cond
        self.result = result

    def args(self):
        return (self.cond, self.result)


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


class Context(ContextBase):
    def __init__(self):
        super().__init__(_zrth.toy.WrappedContext())

    def var(self, name: str, ty: str) -> Symbol:
        return Symbol(name, to_pysmt_type(ty))

    def next_var(self, var: Symbol) -> Symbol:
        """
        Get next variable for `var`.
        """
        assert isinstance(var, Expr), var
        return Symbol(f"{var.symbol_name()}'", var.symbol_type())

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
        elif isinstance(ctrl, (tuple, list)):
            if not all(":" in v for v in ctrl):
                raise RuntimeError("A variable is missing a type annotation")
            ctrl = tuple(
                self.var(v[0].strip(), v[1].strip())
                for v in map(lambda s: s.split(":"), ctrl)
            )
            cur_vars = [*ctrl]
        else:
            raise RuntimeError(
                f"Expect variables to be a string, tuple or list, got: {ctrl}"
            )

        if isinstance(extl, str):
            V = extl.split(",")
            if not all(":" in v for v in V):
                raise RuntimeError("A variable is missing a type annotation")
            extl = tuple(
                self.var(v[0].strip(), v[1].strip())
                for v in map(lambda s: s.split(":"), V)
            )
            cur_vars.extend(extl)
        elif isinstance(extl, (tuple, list)):
            if not all(":" in v for v in extl):
                raise RuntimeError("A variable is missing a type annotation")
            extl = tuple(
                self.var(v[0].strip(), v[1].strip())
                for v in map(lambda s: s.split(":"), extl)
            )
            cur_vars.extend(extl)
        else:
            raise RuntimeError(
                f"Expect variables to be a string, tuple or list, got: {extl}"
            )

        return ctrl, extl, cur_vars

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
        extl_nxt_arg = extl_nxt[0] if len(extl_nxt) == 1 else extl_nxt

        # trace the init function
        if init:
            init_ret = self.trace(init, extl_nxt_arg)
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

    def _choose(self, *args):
        return Choose(args)

    def _case(self, cond, res):
        return Case(cond, res)

    def trace(self, fun: Callable, *args, **kwargs):
        # we want to access the context from the function in order to
        # create terms via API that we cannot map to Python operations.
        # For that, we need to add it as a new argument.
        def wrapped_fun():
            assert "nxt" not in fun.__globals__
            assert "Choose" not in fun.__globals__
            assert "Case" not in fun.__globals__
            fun.__globals__["nxt"] = self.next_var
            fun.__globals__["Choose"] = self._choose
            fun.__globals__["Case"] = self._case
            r = fun(*args, **kwargs)
            del fun.__globals__["nxt"]
            del fun.__globals__["Case"]
            del fun.__globals__["Choose"]
            return r

        r = wrapped_fun()

        return handle_return_value(r)

    def get_pyval_sym(self, sym: Expr) -> PyVal:
        assert sym.is_symbol(), sym

        return self._context.get_sym(
            sym.symbol_name(), from_pysmt_type(sym.symbol_type())
        )

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
                WrappedTerm("Id", outvar, [self.get_pyval_sym(self.next_var(var))])
            )

        update_terms = []
        assert len(update_ret) == len(ctrl)
        for var, expr in zip(ctrl, update_ret):
            tmp, outvar = walker.translate(expr)
            update_terms.extend(tmp)

            # map the output of the expression to the output wire
            assert len(outvar) == 1
            update_terms.append(
                WrappedTerm("Id", outvar, [self.get_pyval_sym(self.next_var(var))])
            )

        cur_vars = [self.get_pyval_sym(v) for v in chain(ctrl, extl)]
        nxt_vars = [self.get_pyval_sym(self.next_var(v)) for v in chain(ctrl, extl)]
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

        if isinstance(formula, Case):
            return [args]
        elif isinstance(formula, Choose):
            reads = list(chain.from_iterable(args))
            assert len(reads) > 0 and len(reads) % 2 == 0, reads
            out = self._ctx.tmp_sym(reads[1].ty())
            terms.append(WrappedTerm("Choose", reads=reads, writes=[out]))
            return [out]
        else:
            assert all(isinstance(a, PyVal) for a in args), args
            opty = formula.node_type()
            if opty == op.SYMBOL:
                return [
                    self._ctx.get_sym(
                        formula.symbol_name(), from_pysmt_type(formula.symbol_type())
                    )
                ]
            elif opty == op.PLUS:
                assert len(args) == 2
                assert args[0].ty() == args[1].ty(), args
                out = self._ctx.tmp_sym(args[0].ty())
                terms.append(WrappedTerm("Arith::Add", reads=args, writes=[out]))
                return [out]
            elif opty == op.LT:
                assert len(args) == 2
                out = self._ctx.tmp_sym("Bool")
                terms.append(WrappedTerm("Cmp::Lt", reads=args, writes=[out]))
                return [out]
            elif opty == op.OR:
                assert len(args) == 2
                out = self._ctx.tmp_sym("Bool")
                terms.append(WrappedTerm("Logical::Or", reads=args, writes=[out]))
                return [out]
            elif opty == op.NOT:
                assert len(args) == 1
                out = self._ctx.tmp_sym("Bool")
                terms.append(WrappedTerm("Logical::Not", reads=args, writes=[out]))
                return [out]
            elif opty == op.ITE:
                assert len(args) == 3
                assert args[1].ty() == args[2].ty(), args
                out = self._ctx.tmp_sym(args[1].ty())
                terms.append(WrappedTerm("Ite", reads=args, writes=[out]))
                return [out]
            if opty == op.INT_CONSTANT:
                return [PyVal.int(formula.constant_value())]
            else:
                raise NotImplementedError(
                    f"Not implemented translation of operation: {
                        formula} {type(formula)}"
                )

        raise RuntimeError("Unreachable")
