from .context import Context, nxt
from ..expr import Expr, Var


class Module:
    def __init__(
        self,
        ctrl: str | tuple[str] | None,
        extl: str | tuple[str] | None = None,
        name: str = None,
        ctx: Context = None,
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

        self._ctx = ctx or Context()

        self.init = self._wrap_method(self.init) if hasattr(self, "init") else None
        self.update = self._wrap_method(self.update)

        extl = extl or ()
        self._module = self._ctx.module_from_methods(
            ctrl, extl, self.init, self.update, name=name
        )

    def _wrap_method(self, m):
        """
        Wrap method such that if it is given symbolic arguments,
        it will be traced and otherwise it will get executed normally.
        """

        def wrapper(*args):
            if any(isinstance(a, Expr) for a in args):
                return self._ctx.trace(m, *args)

            return m(*args)

        return wrapper

    def fresh_variable(self):
        return self._ctx.fresh_var()

    def constant(self, *args):
        return self._ctx.constant(*args)

    @staticmethod
    def nxt(v):
        if isinstance(v, Var):
            return nxt(v)
        # else this is a constant already describing next value
        return v

    def choose(self, *args):
        return self._ctx.choose_impl(*args)

    def dbg(self) -> None:
        self._module.dbg()

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
