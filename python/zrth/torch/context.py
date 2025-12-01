from zrth import _zrth
from zrth.context import Context as ContextBase
#from . term import Var, Term

from typing import Callable, Any

import inspect

from itertools import chain

PyVal = _zrth.PyVal
WrappedModule = _zrth.torch.WrappedModule
WrappedAtom = _zrth.torch.WrappedAtom
WrappedTerm = _zrth.torch.WrappedTerm

from sympy import Expr, Symbol, Add, Lt, Or, Not, ITE, Mul
from sympy.core.sympify import converter
from torch import Tensor as TorchTensor

class Tensor_(Expr):
    """A SymPy expression node that holds a PyTorch tensor"""
    is_commutative = False   # treat tensors like non-commutative atoms

    def __new__(cls, tensor):
        obj = super().__new__(cls)
        obj.tensor = tensor   # store the actual PyTorch tensor
        return obj

    def _sympystr(self, printer):
        return f"Tensor({self.tensor})"

converter[TorchTensor] = lambda x: Tensor_(x)



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


def from_sympy_type(ty) -> str:
    if ty == INT:
        return "Int"
    if ty == REAL:
        return "Real"
    if ty == BOOL:
        return "Bool"
    raise NotImplementedError(f"Unknown type: {ty}")

class SympyContext(ContextBase):
    def __init__(self, ctx_impl):
        super().__init__(ctx_impl)

    def var(self, name: str) -> Symbol:
        return Symbol(name)

    def next_var(self, var: Symbol) -> Symbol:
        """
        Get next variable for `var`.
        """
        assert isinstance(var, Symbol), var
        return Symbol(f"{var.name}'")

    def _parse_variables(self, ctrl, extl):
        if isinstance(ctrl, str):
            ctrl = tuple( self.var(v.strip()) for v in ctrl.split(",") )
            cur_vars = [*ctrl]
        elif isinstance(ctrl, (tuple, list)) and len(ctrl) > 0:
            if isinstance(ctrl[0], str):
                ctrl = tuple( self.var(v) for v in ctrl)
            elif isinstance(ctrl[0], Symbol):
                if not all(isinstance(c, Symbol) and c.is_symbol() for c in ctrl):
                    raise RuntimeError(
                        f"Expected variables to be all sympy variables, got: {ctrl}"
                    )
            else:
                raise RuntimeError(
                    f"Expected variables to be a tuple of strings or sympy variables, got: {ctrl}"
                )
            cur_vars = [*ctrl]
        else:
            raise RuntimeError(
                f"Expect variables to be a non-empty string, tuple or list, got: {ctrl}"
            )

        if isinstance(extl, str):
            extl = tuple(
                self.var(v)
                for v in  extl.split(",")
            )
        elif isinstance(extl, (tuple, list)) and len(extl) > 0:
            if isinstance(extl[0], str):
                extl = tuple( self.var(v) for v in extl)
            elif isinstance(extl[0], Symbol):
                if not all(isinstance(c, Symbol) and c.is_symbol() for c in extl):
                    raise RuntimeError(
                        f"Expected variables to be all sympy variables, got: {extl}"
                    )
            else:
                raise RuntimeError(
                    f"Expected variables to be a tuple of strings or sympy variables, got: {extl}"
                )
            cur_vars.extend(extl)
        else:
            raise RuntimeError(
                f"Expect variables to be a string, tuple or list, got: {extl}"
            )

        return ctrl, extl, cur_vars

    # TODO: get rid of it in the SMT context
    def trace(self, fun: Callable, *args, **kwargs):
        """
        Execute a given function with binding our names like `next` into it.
        """
        # we want to access the context from the function in order to
        # create terms via API that we cannot map to Python operations.
        # For that, we need to add it as a new argument.
        def wrapped_fun():
            assert "next" not in fun.__globals__
            fun.__globals__["next"] = self.next_var
            r = fun(*args, **kwargs)
            del fun.__globals__["next"]
            return r

        r = wrapped_fun()

        return handle_return_value(r)

    def get_pyval_sym(self, sym: Expr) -> PyVal:
        assert sym.is_symbol, sym

        return self._context.var( sym.name)





class Context(SympyContext):
    def __init__(self):
        super().__init__(_zrth.torch.WrappedContext())

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

        atom = WrappedAtom(
            self._context,
            cur_vars,
            nxt_vars,
            init_terms,
            update_terms,
        )

        # TODO: here we unnecessarily copy the terms (they are once copied into
        # Atom and then again into Module)
        module = WrappedModule(self._context, cur_vars, nxt_vars, atom)
        if name is not None:
            module.set_name(name)
        return module

    def _cond(self, cnd, iftrue, iffalse):
        return Cond(args)

    def to_terms(self, ctrl, extl, init_ret, update_ret) -> (list, list, list, list):
        """
        MODIFIES `self._context`
        """
        walker = SympyToTerms(self.unwrap())

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


# TODO: move to its own file and generalize
class TranslateToTerms:
    def __init__(self, ctx) -> None:
        """
        MODIFIES `ctx`
        """
        self._ctx = ctx
        self.terms = None

    @classmethod
    def get_children(cls, formula):
        # TO OVERRIDE
        return formula.args()


    def visit_node(formula, args):
        # TO OVERRIDE
        print(" " * depth, "Visiting node", formula, args)


    def translate(self, formula):
        """
        Translate a formula into reactive module terms.
        Return the terms and the output variable (wire) for the formula.
        """
        self.terms = []
        r = self._visit(formula)
        return self.terms, r

    def _visit(self, formula, depth=0):

        print(" " * depth, "Visiting", formula)

        args = []
        for child in self.get_children(formula):
            args.extend(self._visit(child, depth + 1))

        return self.visit_node(formula, args)





# NOTE: we want to handle also Choose objects and therefore we have to
# traverse the expression ourselves, without using sympy.visiters.Dagvisiter
class SympyToTerms(TranslateToTerms):
    def __init__(self, ctx) -> None:
        """
        MODIFIES `ctx`
        """
        super().__init__(ctx)

    @classmethod
    def get_children(cls, formula):
        return formula.args

    def visit_node(self, formula, args):
        terms = self.terms

        # constants
        # NOTE: check for bool must be before int (bool is an instance of int in Python...)
        if isinstance(formula, bool):
            return [PyVal.bool(formula)]
        elif isinstance(formula, int):
            return [PyVal.int(formula)]

        assert all(isinstance(a, PyVal) for a in args), args
        opty = formula.func
        if opty == Symbol:
            return [
                self._ctx.var( formula.name)
            ]
        elif opty == Add:
            assert len(args) == 2
            assert args[0].ty() == args[1].ty(), args
            out = self._ctx.tmp_var()
            terms.append(WrappedTerm("Add", reads=args, writes=[out]))
            return [out]
        elif opty == Mul:
            assert len(args) == 2
            assert args[0].ty() == args[1].ty(), args
            out = self._ctx.tmp_var()
            terms.append(WrappedTerm("Mul", reads=args, writes=[out]))
            return [out]
        elif opty == Lt:
            assert len(args) == 2
            out = self._ctx.tmp_var()
            terms.append(WrappedTerm("Lt", reads=args, writes=[out]))
            return [out]
        elif opty == Or:
            assert len(args) == 2
            out = self._ctx.tmp_var()
            terms.append(WrappedTerm("Or", reads=args, writes=[out]))
            return [out]
        elif opty == Not:
            assert len(args) == 1
            out = self._ctx.tmp_var()
            terms.append(WrappedTerm("Not", reads=args, writes=[out]))
            return [out]
        elif opty == ITE:
            assert len(args) == 3
            assert args[1].ty() == args[2].ty(), args
            out = self._ctx.tmp_var()
            terms.append(WrappedTerm("Cond", reads=args, writes=[out]))
            return [out]
        elif opty == Tensor_:
            return [PyVal.tensor(formula.tensor)]
        else:
            raise NotImplementedError(
                f"Not implemented translation of operation: {formula} {type(formula)}"
            )

        raise RuntimeError("Unreachable")
