from .context import Context, nxt

from zrth import _zrth

WrappedModule = _zrth.smt.WrappedModule


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
