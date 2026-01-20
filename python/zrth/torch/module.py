from ..expr import nxt, choose, Expr, Sym, sym
from .. import get_ctx

from .ll import DType, Module as RustModule

from typing import Callable, Any


class Module:
    def __init__(
        self,
        ctrl: str | tuple[str] | None,
        extl: str | tuple[str] | None = None,
        name: str = None,
    ):
        if ctrl is None:
            # this is a blank module, it is gonig to be populated
            # from outside (e.g., when wrapping an already existing :class:`WrappedModule`
            # into this class)
            self._ctx = None
            self._module = None
            return

        if not hasattr(self, "update"):
            raise RuntimeError(f"Module {type(self)} has no `update` method.")

        if not hasattr(self, "init"):
            self.init = None
        # self.init = self._wrap_method(self.init) if hasattr(self, "init") else None
        # self.update = self._wrap_method(self.update)

        extl = extl or ()
        self._module = rust_module_from_methods(
            ctrl, extl, self.init, self.update, name=name
        )

        # store the context this module was created in
        self._ctx = get_ctx()

    # def _wrap_method(self, m):
    #    """
    #    Wrap method such that if it is given symbolic arguments,
    #    it will be traced and otherwise it will get executed normally.
    #    """
    #
    #    def wrapper(*args):
    #        if any(isinstance(a, Expr) for a in args):
    #            return self._ctx.trace(m, *args)
    #
    #        return m(*args)
    #
    #    return wrapper

    def __repr__(self) -> str:
        return self._module.__repr__()

    def __str__(self) -> str:
        return self._module.__str__()

    def to_html(self, path: str, open: bool = False):
        self._module.to_html(self._ctx.unwrap(), path)

        if open:
            from sys import platform
            from subprocess import run

            if platform == "linux":
                _ = run(["xdg-open", path])
            else:
                _ = run(["open", path])
            # FIXME: not exhaustive


def parse_variables(variables) -> tuple[Sym]:
    """
    Parse variables (ctrl or extl) given to the module.

    Variables can be given as a string e.g., 'a: Tensor<2>, b: Bool'
    or as a list, e.g., ['a: Tensor<2>', 'b: Bool'] or as a list
    of Sym objects.

    This methods translates `variables` into a list of Sym objects.
    Note that we do not return pairs of (latched, next) symbols, because
    we allow to pass Sym directly and thus the return type would be inconsistent
    if we returned pairs when inputs are strings. We can get next through
    `Sym.nxt` method.

    :return: a list of `Sym` objects.
    """
    if isinstance(variables, str):
        V = variables.split(",")
    elif isinstance(variables, (tuple, list)):
        V = variables
    else:
        raise RuntimeError(
            f"Expect variables to be a string, tuple or list, got: {variables}"
        )

    syms = []
    for v in V:
        if isinstance(v, str):
            if not ":" in v:
                raise RuntimeError(f"A variable `{v}` is missing a type annotation")
            v = v.split(":")
            v = sym(v[0].strip(), DType.from_str(v[1].strip()), create_pair=True)[0]
        elif not isinstance(v, Sym):
            raise RuntimeError(
                f"A variable must be given as a string or `Sym` object, got {v} ({type(v)})"
            )
            # test if the symbol has been created with next symbol,
            # this is a requirement
            v.nxt()

        syms.append(v)

    return syms


def parse_module_arguments(ctrl, extl):
    ctrl = parse_variables(ctrl)
    extl = parse_variables(extl)

    return ctrl, extl


def trace(fun: Callable, *args, **kwargs):
    # print("----- tracing ----")
    # run the function
    r = fun(*args, **kwargs)
    # print("----- finished tracing ----")

    return r


def rust_module_from_methods(
    ctrl: str | tuple[str],
    extl: str | tuple[str],
    init: Callable[[], None],
    update: Callable[[], None],
    name: str | None = None,
) -> RustModule:
    """
    Create the Rust module from the `init` and `update` methods.
    """

    # parse and create tuples of input variables
    ctrl, extl = parse_module_arguments(ctrl, extl)

    # take only the latched Symbols from the tuples of variables,
    # next values will be accessed through `nxt` or `[1]`
    # if the user uses a single variable, it is more natural in the `init` and `update` to unwrap it
    ctrl_arg = ctrl[0] if len(ctrl) == 1 else tuple(*ctrl)
    extl_arg = extl[0] if len(extl) == 1 else tuple(*extl)

    # trace the init function
    if init:
        # say context to gather terms
        init_ret, init_terms = trace_fun(ctrl, init, extl_arg)
        assert len(init_ret) == len(ctrl), (init_ret, ctrl)
    else:
        init_terms = []

    # trace the update function
    update_ret, update_terms = trace_fun(ctrl, update, ctrl_arg, extl_arg)
    assert len(update_ret) == len(ctrl), (len(update_ret), len(ctrl))

    # create the Rust module
    return RustModule.sequential(
        init_terms,
        update_terms,
        [(s.wire(), s.nxt().wire()) for s in ctrl]
        + [(s.wire(), s.nxt().wire()) for s in extl],
    )


def trace_fun(ctrl, fun, *args):
    ctx = get_ctx()

    # say context to gather terms
    terms = []
    ctx.push_terms_frame(terms)

    # get the expressions (and terms) for the function
    ret = handle_return_value(fun(*args))

    # wire the output of the function with control variables
    assert len(ret) == len(ctrl)
    ret = [
        Expr(
            "assign",
            expr,  # what
            s.nxt(),  # to
        )
        for s, expr in zip(ctrl, ret)
    ]

    ctx.pop_terms_frame()
    return ret, terms


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
