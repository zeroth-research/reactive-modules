from zrth import WiredTransitions


class ModuleUnrolling:
    def __init__(self, m) -> None:
        self._module = m
        # initialized by `init`
        self._transitions: None | WiredTransitions = None

    def transitions(self) -> WiredTransitions | None:
        return self._transitions

    def init(self):
        self._transitions = WiredTransitions()
        self._transitions.init(self._module.unwrap(), self._module._ctx.unwrap())

    def step(self):
        assert self._transitions
        self._transitions.step(self._module.unwrap(), self._module._ctx.unwrap())

    def to_html(self, path: str, open: bool = False):
        assert self._transitions
        self._transitions.to_html(self._module._ctx.unwrap(), path)

        if open:
            from sys import platform
            from subprocess import run

            if platform == "linux":
                _ = run(["xdg-open", path])
            else:
                _ = run(["open", path])
            # FIXME: not exhaustive

    def dbg(self):
        assert self._transitions
        self._transitions.dbg()

    def last_state(self):
        assert self._transitions
        self._transitions.last_state()
