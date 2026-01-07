from torch import Tensor
from random import randrange
from zrth import _zrth
from zrth.context import Context as ContextBase
from zrth.expr import Expr, Var, Transform as ExprTransform

from typing import Callable, Any

from itertools import chain

PyVal = _zrth.PyVal
WrappedModule = _zrth.torch.WrappedModule
WrappedTerm = _zrth.torch.WrappedTerm


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


def nxt(var: Var) -> Var:
    """
    Get next variable for `var`.
    """
    assert isinstance(var, Var), var
    return Var(f"nxt({var.name})")


class IfThen(Expr):
    """
    An expression representing `IfThen` term.

    This expression is used inside `Choose` terms.
    """

    def __init__(self, cond: Expr | bool, expr: Any):
        super().__init__("ifthen", [cond, expr])

    def cond(self):
        return self.get_children()[0]

    def expr(self):
        return self.get_children()[1]

    def is_concrete(self) -> bool:
        return not any(isinstance(c, Expr) for c in self.get_children())


def is_concrete_choose(alist: list[Any]) -> bool:
    print("--- choose ---")
    print(isinstance(alist[-1], Expr))
    print(any(not a.is_concrete() for a in alist[:-1]))
    print("--- end choose ---")
    last = alist[-1]
    # the last arg is IfThen
    if isinstance(last, IfThen):
        return all(a.is_concrete() for a in alist)

    # the last arg is not IfThen (it is the default)
    return not isinstance(alist[-1], Expr) and all(a.is_concrete() for a in alist[:-1])


def _choose(alist):
    assert all(isinstance(a, IfThen) for a in alist), alist
    if all(a.is_concrete() for a in alist):
        # execute choose concretely
        sat_args = [arg for arg in alist if arg.cond()]
        if sat_args:
            return sat_args[randrange(len(sat_args))].expr()

        # return None
        raise RuntimeError("No satisfiable branch in a choose statement")

    return Expr("choose", alist)


def _choose_or(alist):
    choices = alist[:-1]
    last = alist[-1]

    # the last argument may not be `IfThen`, in which case
    # it is the default (unconditional) argument
    assert not isinstance(last, IfThen), alist
    assert all(isinstance(a, IfThen) for a in choices), alist

    if not isinstance(last, Expr) and all(a.is_concrete() for a in choices):
        # execute choose_or concretely
        sat_args = [arg for arg in choices if arg.cond()]
        if sat_args:
            return sat_args[randrange(len(sat_args))].expr()
        return last

    return Expr("choose_or", alist)


def choose(*args):
    """
    Implementation of choose construct.
    """
    alist = list(args)
    if isinstance(alist[-1], IfThen):
        return _choose(alist)
    else:
        return _choose_or(alist)


class Context_(ContextBase):
    __next_fresh_var = 0

    def __init__(self, ctx_impl):
        super().__init__(ctx_impl)
        # map variables to terms that define them
        self.var_to_term: dict[Var, Term] = {}

    def var(self, name: str) -> Var:
        return Var(name)

    def fresh_var(self) -> Var:
        Context_.__next_fresh_var += 1
        return Var(f"__x_{Context_.__next_fresh_var}")

    def constant(self, *args):
        return Tensor(*args)

    @staticmethod
    def next_var(var: Var) -> Var:
        """
        Get next variable for `var`.
        """
        return nxt(var)

    def _parse_variables(self, ctrl, extl):
        if isinstance(ctrl, str):
            ctrl = tuple(self.var(v.strip()) for v in ctrl.split(","))
            cur_vars = [*ctrl]
        elif isinstance(ctrl, (tuple, list)) and len(ctrl) > 0:
            if isinstance(ctrl[0], str):
                ctrl = tuple(self.var(v) for v in ctrl)
            elif isinstance(ctrl[0], Var):
                if not all(isinstance(c, Var) and c.is_symbol() for c in ctrl):
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
            extl = tuple(self.var(v) for v in extl.split(","))
        elif isinstance(extl, (tuple, list)) and len(extl) > 0:
            if isinstance(extl[0], str):
                extl = tuple(self.var(v) for v in extl)
            elif isinstance(extl[0], Var):
                if not all(isinstance(c, Var) and c.is_symbol() for c in extl):
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

    def trace(self, fun: Callable, *args, **kwargs):
        # print("----- tracing ----")
        # run the function
        r = fun(*args, **kwargs)
        # print("----- finished tracing ----")

        return r


class Context(Context_):
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

        # parse and create tuples of input variables
        ctrl, extl, cur_vars = self._parse_variables(ctrl, extl)
        extl_nxt = [self.next_var(v) for v in extl]

        # if the user uses a single variable, it is more natural in the `init` and `update` to unwrap it
        ctrl_arg = ctrl[0] if len(ctrl) == 1 else ctrl
        extl_arg = extl[0] if len(extl) == 1 else extl
        # extl_nxt_arg = extl_nxt[0] if len(extl_nxt) == 1 else extl_nxt

        # trace the init function (the function will be traced upon given symbolic arguments)
        if init:
            init_ret = handle_return_value(init(extl_arg))
            assert len(init_ret) == len(ctrl), (init_ret, ctrl)
        else:
            init_terms = []

        # trace the update function
        update_ret = handle_return_value(update(ctrl_arg, extl_arg))
        assert len(update_ret) == len(ctrl)

        # translate all that we have into terms
        cur_vars, nxt_vars, init_terms, update_terms = self.to_terms(
            ctrl, extl, init_ret, update_ret
        )

        # create the Rust module
        return WrappedModule(
            self._context, cur_vars, nxt_vars, init_terms, update_terms
        )

    def var_to_pyval(self, var: Var) -> PyVal:
        return self.unwrap().var(var.name)

    def to_terms(self, ctrl, extl, init_ret, update_ret) -> (list, list, list, list):
        """
        MODIFIES `self._context`
        """
        walker = Translate(self)

        init_terms = []
        assert len(init_ret) == len(ctrl)
        for var, expr in zip(ctrl, init_ret):
            tmp, outvar = walker.translate(expr)
            init_terms.extend(tmp)

            # map the output of the expression to the output wire
            assert len(outvar) == 1, outvar
            init_terms.append(
                WrappedTerm("Id", outvar, [self.var_to_pyval(self.next_var(var))])
            )

        update_terms = []
        assert len(update_ret) == len(ctrl)
        for var, expr in zip(ctrl, update_ret):
            tmp, outvar = walker.translate(expr)
            update_terms.extend(tmp)

            # map the output of the expression to the output wire
            assert len(outvar) == 1
            update_terms.append(
                WrappedTerm("Id", outvar, [self.var_to_pyval(self.next_var(var))])
            )

        cur_vars = [self.var_to_pyval(v) for v in chain(ctrl, extl)]
        nxt_vars = [self.var_to_pyval(self.next_var(v)) for v in chain(ctrl, extl)]
        return cur_vars, nxt_vars, init_terms, update_terms


def rel_to_str(op: str) -> str:
    if op == "lt":
        return "Lt"
    if op == "gt":
        return "Gt"
    if op == "le":
        return "Le"
    if op == "ge":
        return "Ge"
    if op == "eq":
        return "Eq"

    raise NotImplementedError(f"Unknown relation: {opty}")


def arith_to_str(op) -> str:
    if op == "add":
        return "Add"
    if op == "mul":
        return "Mul"
    if op == "matmul":
        return "MatMul"

    raise NotImplementedError(f"Unknown operation: {op}")


class Translate(ExprTransform):
    def __init__(self, ctx) -> None:
        """
        MODIFIES `ctx`
        """
        self.ctx = ctx
        # FIXME: this is confusing
        self._ctx = ctx.unwrap()
        self.terms = []

    def translate(self, formula):
        """
        Translate a formula into reactive module terms.
        Return the terms and the output variable (wire) for the formula.
        """
        self.terms = []
        r = self.transform(formula)
        return self.terms, r

    def default(self, expr, args):
        raise NotImplementedError(f"Not implemented translation of operation: `{expr}`")

    # def before_all(self, expr: Expr, args: list):
    #    print("VIS", expr, type(expr))

    # def forall(self, expr: Expr, args: list):
    #    assert all(isinstance(a, PyVal) for a in args), args

    def visit_type_bool(self, expr):
        return [PyVal.bool(expr)]

    def visit_type_int(self, expr):
        return [PyVal.int(expr)]

    def visit_type(self, expr, args):
        if isinstance(expr, Tensor):
            return [PyVal.tensor(expr)]
        return self.default(expr, args)

    def visit_var(self, expr, args):
        return [self._ctx.var(expr.name)]

    def visit_arith(self, expr, args, op):
        assert len(expr.args) == 2
        assert args[0].dtype() == args[1].dtype(), args
        out = self._ctx.tmp_var()
        self.terms.append(WrappedTerm(arith_to_str(op), reads=args, writes=[out]))
        return [out]

    def visit_cmp(self, expr, args, op):
        assert len(args) == 2
        out = self._ctx.tmp_var()
        self.terms.append(WrappedTerm(rel_to_str(op), reads=args, writes=[out]))
        return [out]

    def visit_logic(self, expr, args, op):
        if op == "not":
            assert len(args) == 1
            out = self._ctx.tmp_var()
            self.terms.append(WrappedTerm("Not", reads=args, writes=[out]))
        else:
            assert len(args) == 2
            assert op in ("and", "or"), op
            out = self._ctx.tmp_var()
            op = "Or" if op == "or" else "And"
            self.terms.append(WrappedTerm(op, reads=args, writes=[out]))
        return [out]

    def visit_ite(self, expr, args):
        assert len(args) == 3
        assert args[1].dtype() == args[2].dtype(), args
        out = self._ctx.tmp_var()
        self.terms.append(WrappedTerm("Cond", reads=args, writes=[out]))
        return [out]

    def visit_ifthen(self, expr, args):
        assert len(args) == 2
        out = self._ctx.tmp_var()
        self.terms.append(WrappedTerm("IfThen", reads=args, writes=[out]))
        return [out]

    def visit_choose(self, expr, args):
        assert len(args) > 0, args
        out = self._ctx.tmp_var()
        self.terms.append(WrappedTerm("Choose", reads=args, writes=[out]))
        return [out]

    def visit_choose_or(self, expr, args):
        assert len(args) > 0, args
        out = self._ctx.tmp_var()
        self.terms.append(WrappedTerm("ChooseOr", reads=args, writes=[out]))
        return [out]
