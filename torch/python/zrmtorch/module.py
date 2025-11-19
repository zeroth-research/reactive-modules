from ast import Call
from typing import Callable

from numpy import isin
from bindings.context import Context

from subprocess import run


class Module:
    def __init__(
        self,
        ctrl: str | tuple[str],
        extl: str | tuple[str] | None = None,
        name: str = None,
        ctx: Context = None,
    ):
        assert ctrl, "Need controlled variables"

        if not hasattr(self, "update"):
            raise RuntimeError(f"Module {type(self)} has no `update` method.")

        self._ctx = ctx or Context()

        # TODO: check also the signature of init and update

        init = self.init if hasattr(self, "init") else None
        extl = extl or ()
        self._module = self._ctx.module_from_methods(
            ctrl, extl, init, self.update, name=name
        )

    def to_html(self, path: str, open: bool = False):
        self._module.to_html(self._ctx.unwrap(), path)

        if open:
            from sys import platform

            if platform == "linux":
                _ = run(["xdg-open", path])
            else:
                _ = run(["open", path])
            # FIXME: not exhaustive


class SimpleModule:
    def __init__(
        self,
        vars: list[str],
        init: Callable[[], None],
        update: Callable[[], None],
        name: str = None,
        ctx: Context = None,
    ):
        self._ctx = ctx or Context()
        if vars is None and init is None:
            self._module = self._ctx.module_from_fn(update, name=name)
        else:
            self._module = self._ctx.module(vars, init, update, name=name)

    @staticmethod
    def with_ctx(
        ctx: Context,
        vars: list[str],
        init: Callable[[], None],
        update: Callable[[], None],
    ):
        return SimpleModule(vars, init, update, ctx)

    @staticmethod
    def from_fn(
        fun: Callable[[], None],
    ):
        return SimpleModule(None, None, fun, None)

    def to_html(self, path: str, open: bool = False):
        self._module.to_html(self._ctx.unwrap(), path)

        if open:
            from sys import platform

            if platform == "linux":
                _ = run(["xdg-open", path])
            else:
                _ = run(["open", path])
            # FIXME: not exhaustive
