from .context import Context, nxt
from .. import smt as zrth_smt

WrappedModule = zrth_smt.WrappedModule


class Transition:
    def __init__(self, /, impl):
        self._wrappedtransition = impl

    def unwrap(self):
        return self._wrappedtransition

    def dbg(self):
        return self._wrappedtransition.dbg()


class Module:
    def __init__(
        self,
        ctrl: str | tuple[str],
        extl: str | tuple[str] | None = None,
        name: str = None,
        ctx: Context = None,
    ):
        if ctrl is None:
            # will build other way
            # FIXME: this is hacky!
            self._ctx = None
            self._module = None
            return

        if not hasattr(self, "update"):
            raise RuntimeError(f"Module {type(self)} has no `update` method.")

        self._ctx = ctx or Context()

        # TODO: check also the signature of init and update

        init = self.init if hasattr(self, "init") else None
        extl = extl or ()
        self._module, self._ctrl, self._extl = self._ctx.module_from_methods(
            ctrl, extl, init, self.update, name=name
        )

    def ctrl(self, as_pair=False):
        """
        Return the contolled variables (in the order defined during initialization of the module).
        If `as_pair` is True, the function returns a tuple of pairs
        where each element is (latched, next) symbols.
        """
        if as_pair:
            return ((v, nxt(v)) for v in self._ctrl)

        return self._ctrl

    def extl(self, as_pair=False):
        """
        Return the external variables (in the order defined during initialization of the module).
        If `as_pair` is True, the function returns a tuple of pairs
        where each element is (latched, next) symbols.
        """
        if as_pair:
            return ((v, nxt(v)) for v in self._extl)

        return self._extl

    def dbg(self) -> None:
        self._module.dbg()

    def unwrap(self) -> WrappedModule:
        return self._module

    @staticmethod
    def nxt(v):
        return nxt(v)

    @staticmethod
    def parallel(modules: list) -> WrappedModule:
        return WrappedModule.parallel([m.unwrap() for m in modules])

    def to_smtlib(self, what: str = None) -> str:
        return self._module.to_smtlib(what)

    def __str__(self) -> str:
        return self._module.__str__()

    def __repr__(self) -> str:
        return self._module.__repr__()

    def ctx(self):
        return self._ctx

    def unwrap(self):
        return self._module

    def init_as_transition(self):
        return Transition(impl=self._module.init_as_transition())

    def update_as_transition(self):
        return Transition(impl=self._module.update_as_transition())

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


class ModuleUnrolling:
    def __init__(self, m: Module) -> None:
        self._module = m
        self._transitions = _zrth.smt.WrappedWiredTransitions()

    def init(self):
        self._transitions.init(self._module.unwrap(), self._module._ctx.unwrap())

    def step(self):
        self._transitions.step(self._module.unwrap(), self._module._ctx.unwrap())

    def dbg(self):
        self._transitions.dbg()

    def last_state(self):
        self._transitions.last_state()


class Unrolling:
    def __init__(self) -> None:
        self._transitions = _zrth.smt.WrappedWiredTransitions()

    def wire_transition(self, t, ctx):
        self._transitions.wire_transition(t.unwrap(), ctx.unwrap())

    def dbg(self):
        self._transitions.dbg()

    def last_state(self):
        self._transitions.last_state()
