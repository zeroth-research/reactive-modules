from torch import Tensor
from zrth import _zrth
from zrth.context import Context as ContextBase
from zrth.expr import Expr, Var

from typing import Callable


from .translation import to_terms
from .exprs import nxt
from .ll import *


class Context(ContextBase):
    __next_fresh_var = 0

    def __init__(self):
        super().__init__(_zrth.torch.WrappedContext())
        # map variables to terms that define them
        self.var_to_term: dict[Var, Term] = {}

    def var(self, name: str) -> Var:
        return Var(name)

    def fresh_var(self) -> Var:
        Context.__next_fresh_var += 1
        return Var(f"__x_{Context.__next_fresh_var}")

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

    def module_from_methods(
        self,
        ctrl: str | tuple[str],
        extl: str | tuple[str],
        init: Callable[[], None],
        update: Callable[[], None],
        name: str | None = None,
    ) -> Module:
        """
        Create the Rust module (:class:`torch.ll.Module`) from the `init` and `update` methods.
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
        cur_vars, nxt_vars, init_terms, update_terms = to_terms(
            self, ctrl, extl, init_ret, update_ret
        )

        # create the Rust module
        return Module(self._context, cur_vars, nxt_vars, init_terms, update_terms)


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
