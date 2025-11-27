from .context import Context


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
