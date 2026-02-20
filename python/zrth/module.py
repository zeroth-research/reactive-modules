# from .context import Context, get_ctx
from .zrth import DType, Module as RustModule, Transition, Wire

from typing import Callable


class WirePair(tuple):
    def __new__(cls, *elms):
        assert all(isinstance(e, Wire) for e in elms)
        assert len(elms) == 2
        return super().__new__(cls, elms)

    def latched(self) -> Wire:
        return self[0]

    def nxt(self) -> Wire:
        return self[1]


class ReactiveModuleDef:
    """
    A class serving as the base for the definition of a reactive module.

    This class wrapps Rust module (`zrth.Module` defined in `src/module.rs`).
    The Rust module is created by defining and calling `convert` method that assigns
    the generated Rust module into `self._module`. This method must be defined
    by sub-classes.
    Alternatively, the module can be also passed as an argument to the ctor in sub-classes
    the Rust module creation happens outside (e.g., by parallel composition) and
    we want only to wrap the result.

    Because of the reasons above, this class does *NOT* create the Rust module automatically,
    for that, see [ReactiveModule].

    ## Examples

    Typically, you use `ReactiveModuleDef` in such situations:

    ```
    class MyModuleType(ReactiveModuleDef):
        pass

    class MyModule1(MyModuleType):
        ...

        def convert(self):
            ...

    class MyModule2(MyModuleType):
        def __init__(self):
            ...

            # no more inheritance expected
            self.convert()

        def convert(self):
            ...

    class MyModule3(MyModule1):
        ...

    m1 = MyModule1(...).convert()
    m2 = MyModule2(...)
    m3 = MyModule3().convert()
    ```

    Note that if you do not want to inherit from `MyModule*` further,
    you can call `convert` directly from *their* `__init__`.
    If you know that you will inherit from `MyModule*` exactly once
    (the child sub-class will be the one that actually defines the reactive module),
    you can make the sub-class `__init__` to call `convert` by calling `call_convert_after_init`
    from `__init_subclass__` (or via Python metaclasses).
    See [ReactiveModule.__init_subclass__] for an example.
    """

    def __init__(
        self,
        intf: str | tuple[str, ...] | None = None,
        extl: str | tuple[str, ...] | None = None,
        prvt: str | tuple[str, ...] | None = None,
        rust_module: None | RustModule = None,
        name: None | str = None,
    ):
        # self._ctx = get_ctx()
        self._intf_wires, self._intf_names = parse_variables(intf)
        self._extl_wires, self._extl_names = parse_variables(extl)
        self._prvt_wires, self._prvt_names = parse_variables(prvt)

        # Not used ATM
        self._name = name

        # If not given directly, it will be set by `convert()`
        self._module = rust_module

    @property
    def intf(self) -> tuple[Wire, ...]:
        return tuple(w[0] for w in self._intf_wires)

    @property
    def intf2(self) -> tuple[WirePair, ...]:
        return self._intf_wires

    @property
    def extl(self) -> tuple[Wire, ...]:
        return tuple(w[0] for w in self._extl_wires)

    @property
    def extl2(self) -> tuple[WirePair, ...]:
        return self._extl_wires

    @property
    def prvt(self) -> tuple[Wire, ...]:
        return tuple(w[0] for w in self._prvt_wires)

    @property
    def prvt2(self) -> tuple[WirePair, ...]:
        return self._prvt_wires

    @property
    def obs(self) -> tuple[Wire, ...]:
        """Observable wires: external inputs + interface outputs"""
        return self.extl + self.intf

    @property
    def obs2(self) -> tuple[WirePair, ...]:
        """Observable wires: external inputs + interface outputs"""
        return self.extl2 + self.intf2

    @property
    def ctrl(self) -> tuple[Wire, ...]:
        """Controlled wires: interface outputs + private wires"""
        return self.intf + (self.prvt or ())

    @property
    def ctrl2(self) -> tuple[WirePair, ...]:
        """Controlled wires: interface outputs + private wires"""
        return self.intf2 + (self.prvt2 or ())

    @property
    def intf_names(self) -> tuple[str, ...]:
        return self._intf_names

    @property
    def extl_names(self) -> tuple[str, ...]:
        return self._extl_names

    @property
    def prvt_names(self) -> tuple[str, ...]:
        return self._prvt_names

    def name_to_wires(self) -> dict[str, WirePair]:
        return {
            name: wp
            for name, wp in zip(
                self._extl_names + self._intf_names + self._prvt_names,
                self._extl_wires + self._intf_wires + self._prvt_wires,
            )
        }

    def convert(self):
        raise RuntimeWarning("Convert not defined")

    @property
    def atoms(self):
        return self.unwrap().atoms()

    def init_as_transition(self) -> Transition:
        return self._module.init_as_transition()

    def update_as_transition(self) -> Transition:
        return self._module.update_as_transition()

    def unwrap(self) -> RustModule | None:
        return self._module

    def __repr__(self) -> str:
        return f"""
{self.__class__} {{
  intf: {", ".join(f"{n}: {v}" for n, v in zip(self._intf_names, self._intf_wires))}
  extl: {", ".join(f"{n}: {v}" for n, v in zip(self._extl_names, self._extl_wires))}
  prvt: {", ".join(f"{n}: {v}" for n, v in zip(self._prvt_names, self._prvt_wires))}
  module:
  {self._module.__repr__() if self._module else "<no rust module>"}
}}
"""

    def __str__(self) -> str:
        return f"""
ReactiveModule {{
  intf: {", ".join(f"{n}: {v}" for n, v in zip(self._intf_names, self._intf_wires))}
  extl: {", ".join(f"{n}: {v}" for n, v in zip(self._extl_names, self._extl_wires))}
  prvt: {", ".join(f"{n}: {v}" for n, v in zip(self._prvt_names, self._prvt_wires))}
  module:
  {self._module.__repr__() if self._module else "<no rust module>"}
}}
"""


# def to_html(self, path: str, open: bool = False):
#    self._module.to_html(self._ctx.unwrap(), path)
#
#    if open:
#        from sys import platform
#        from subprocess import run
#
#        if platform == "linux":
#            _ = run(["xdg-open", path])
#        else:
#            _ = run(["open", path])
#        # FIXME: not exhaustive


class ReactiveModule(ReactiveModuleDef):
    """
    A class representing the definition of a reactive module.

    The reactive module is defined by methods `init` and `update`
    and the underlying Rust module is created *automatically*
    after init. This class can be sub-classed only *once*
    to a class that defines `update` and `init`.
    If you need more complex sub-classing, see [ReactiveModuleDef].

    ## Examples

    ```
    class MyModule(ReactiveModule):
        def init(...): ...
        def update(...): ...

    m = MyModule(...)
    ```
    """

    def __init__(
        self,
        intf: str | tuple[str, ...] | None,
        extl: str | tuple[str, ...] | None = None,
        prvt: str | tuple[str, ...] | None = None,
        name: None | str = None,
    ):
        super().__init__(intf, extl, prvt, name=name)

        # convert will be called automatically after sub-class's __init__ finishes

    # def convert(self):
    #     assert hasattr(self, "update"), type(self)
    #     assert self._module is None
    #
    #     if self.prvt:
    #         raise NotImplementedError("Private variables not implemented yet")
    #
    #     if not hasattr(self, "init"):
    #         self.init = None
    #
    #     self._module = rust_module_from_methods(
    #         self.ctrl, self.extl, self.init, self.update, name=self._name
    #     )
    #
    def __init_subclass__(cls, **kwargs):
        """
        Hook to automatically call conversion after subclass' __init__.
        """
        call_convert_after_init(__class__, cls, **kwargs)


def call_convert_after_init(thiscls, cls, **kwargs):
    """
    Hook to automatically call conversion after subclass' __init__.
    The conversion *must* be done after subclasses are fully initialized
    because the conversion uses data from the classes
    """
    super(thiscls).__init_subclass__(**kwargs)

    # Save original __init__
    original_init = cls.__init__

    # Wrap the init to call `convert` after the original init finishes
    def wrapped_init(self, *args, **kwargs):
        # Call original __init__
        original_init(self, *args, **kwargs)

        self.convert()

    # Replace __init__ with wrapped version
    cls.__init__ = wrapped_init


def parse_variables(
    variables,
) -> tuple[tuple[WirePair, ...], tuple[str, ...]]:
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

    :return: a list of `(Wire, Wire)` and a list of names
    """
    if not variables:
        return (), ()

    if isinstance(variables, str):
        V = variables.split(",")
    elif isinstance(variables, (tuple, list)):
        V = variables
    else:
        raise RuntimeError(
            f"Expect variables to be a string, tuple or list, got: {variables}"
        )

    syms: list[WirePair] = []
    names: list[str] = []
    for v in V:
        if not isinstance(v, str):
            raise ValueError(f"Invalid variable specification: `{v}`")

        if ":" not in v:
            raise RuntimeError(f"A variable `{v}` is missing a type annotation")

        v: list[str] = v.split(":")
        name, dtype = v[0].strip(), v[1].strip()
        dtype = DType.from_str(dtype)
        syms.append(WirePair(Wire(dtype), Wire(dtype)))
        names.append(name)

    return tuple(syms), tuple(names)


#
# def rust_module_from_methods(
#     ctrl: tuple[WirePair],
#     extl: tuple[WirePair],
#     init: Callable[[], None],
#     update: Callable[[], None],
#     name: str | None = None,
# ) -> RustModule:
#     """
#     Create the Rust module from the `init` and `update` methods.
#     """
#     assert all(isinstance(wt, WirePair) for wt in ctrl), ctrl
#     assert all(isinstance(wt, WirePair) for wt in extl), extl
#     # assert all(isinstance(v, Sym) for v in prvt)
#
#     # take only the latched Symbols from the tuples of variables,
#     # next values will be accessed through `nxt` or `[1]`
#     # if the user uses a single variable, it is more natural in the `init` and `update` to unwrap it
#     ctrl_arg = ctrl[0] if len(ctrl) == 1 else tuple(*ctrl)
#     extl_arg = extl[0] if len(extl) == 1 else tuple(*extl)
#
#     # trace the init function
#     if init:
#         # say context to gather terms
#         init_ret, init_terms = trace_fun(ctrl, init, extl_arg)
#         assert len(init_ret) == len(ctrl), (init_ret, ctrl)
#     else:
#         init_terms = []
#
#     # trace the update function
#     update_ret, update_terms = trace_fun(ctrl, update, ctrl_arg, extl_arg)
#     assert len(update_ret) == len(ctrl), (len(update_ret), len(ctrl))
#
#     # create the Rust module
#     return RustModule.sequential(
#         init_terms,
#         update_terms,
#         [(s.wire(), s.nxt().wire()) for s in ctrl]
#         + [(s.wire(), s.nxt().wire()) for s in extl],
#     )


#
# def trace_fun(ctrl, fun, *args):
#     ctx = get_ctx()
#
#     # say context to gather terms
#     terms = []
#     ctx.push_terms_frame(terms)
#
#     # get the expressions (and terms) for the function
#     ret = handle_return_value(fun(*args))
#
#     # wire the output of the function with control variables
#     assert len(ret) == len(ctrl)
#     ret = [
#         Expr(
#             "assign",
#             expr,  # what
#             s.nxt(),  # to
#         )
#         for s, expr in zip(ctrl, ret)
#     ]
#
#     ctx.pop_terms_frame()
#     return ret, terms
#
#
# def handle_return_value(r):
#     """
#     Transform the return value into a list
#     """
#
#     if r is None:
#         r = []
#     elif isinstance(r, tuple):
#         r = [x for x in r]
#     elif not isinstance(r, list):
#         r = [r]
#
#     # return [x.var if isinstance(x, Assignment) else x for x in r]
#     return r
