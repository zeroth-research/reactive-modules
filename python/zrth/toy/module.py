from .context import Context
from typing import Callable, Any
from copy import copy

from .. smt.module import Module as SmtModule


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
        self._module = self._ctx.module_from_methods(
            ctrl, extl, init, self.update, name=name
        )

    def dbg(self) -> None:
        self._module.dbg()

    def translate_to(self, ty: str) -> "Module":
        new = SmtModule(None)
        # FIXME: this works now, but we'll need a translation
        # between contexts in general..
        new._ctx = self._ctx
        new._module = self._module.translate_to(ty)
        return new

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
