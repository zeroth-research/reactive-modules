"""Generic textual TUI for stepping through any gym.Env (or Env-wrapped module).

Usage::

    from tui import EnvApp
    from zrth import Env

    app = EnvApp(Env(my_env), action_labels=["left", "right"])
    app.run()

Controls
--------
0 … n-1   select action (highlighted in the action bar)
Space     step with the selected action
R         reset
Q         quit
"""
import torch
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer, DataTable, Label, Static
from textual.containers import Horizontal, Vertical


# ── tensor / scalar helpers ──────────────────────────────────────────────────

_GYM_INTERNALS = frozenset({
    "action_space", "observation_space", "np_random",
    "reward_range", "spec", "metadata",
})


def _fmt_py(v) -> str:
    """Compact string for a Python native scalar."""
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, int):
        return str(v)
    return repr(v)


class EnvApp(App):
    """TUI for stepping through any ``gym.Env`` or ``zrth.Env`` instance.

    Calls ``env.reset()`` / ``env.step(action)`` directly and inspects
    ``env.__dict__`` for state variables.

    Controls
    --------
    0 … n-1   select action (highlighted in the action bar)
    Space     step with the selected action
    R         reset
    Q         quit
    """

    CSS = """
    Screen { background: #0d1117; }

    #top-row {
        height: auto;
        max-height: 14;
        margin: 1 2 0 2;
    }

    #state-panel {
        width: 1fr;
        border: round #30363d;
        padding: 0 1;
        margin-right: 1;
    }

    #output-panel {
        width: 1fr;
        border: round #30363d;
        padding: 0 1;
    }

    .panel-title {
        color: #8b949e;
    }

    #action-bar {
        margin: 1 2 0 2;
        border: round #30363d;
        padding: 0 1;
        height: 3;
    }

    #history-label {
        color: #8b949e;
        margin: 1 2 0 2;
    }

    DataTable {
        margin: 0 2 1 2;
        height: 1fr;
    }

    #state-table  { height: auto; }
    #output-table { height: auto; }
    """

    BINDINGS = [
        Binding("space", "do_step",  "Step",        priority=True),
        Binding("enter", "do_step",  "Step",        show=False, priority=True),
        Binding("r",     "do_reset", "Reset"),
        Binding("q",     "quit",     "Quit"),
    ]

    # ── construction ─────────────────────────────────────────────────────────

    def __init__(self, env, action_labels: list[str] | None = None, title: str | None = None):
        super().__init__()
        self._env = env
        self._n: int = env.action_space.n
        self._labels = action_labels or [str(i) for i in range(self._n)]
        self._app_title = title or type(env).__name__

        self._cur_action = 0
        self._step = 0

        self._state_keys: list[str] = []
        self._output_keys = ["obs", "reward", "terminated", "truncated"]

        self._cur_state: dict = {}
        self._cur_outputs: dict = {}
        self._prev_combined: dict = {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _env_state(self) -> dict:
        """Extract scalar state variables from env.__dict__, filtering internals."""
        return {
            k: v
            for k, v in self._env.__dict__.items()
            if k not in _GYM_INTERNALS
            and not k.startswith("_")
            and isinstance(v, (bool, int, float))
        }

    def _changed(self) -> set[str]:
        combined = {**self._cur_state, **self._cur_outputs}
        return {
            k for k, v in combined.items()
            if k in self._prev_combined and self._prev_combined[k] != v
        }

    # ── compose ──────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top-row"):
            with Vertical(id="state-panel"):
                yield Label("State", classes="panel-title")
                yield DataTable(id="state-table", cursor_type="none")
            with Vertical(id="output-panel"):
                yield Label("Outputs", classes="panel-title")
                yield DataTable(id="output-table", cursor_type="none")
        yield Static("", id="action-bar")
        yield Label("── History ──", id="history-label")
        yield DataTable(id="history-table", cursor_type="none")
        yield Footer()

    # ── startup ──────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self.title = self._app_title
        self.query_one("#state-table",  DataTable).add_columns("Variable", "Value", "Δ")
        self.query_one("#output-table", DataTable).add_columns("Output",   "Value", "Δ")

        self._run_reset()
        self._state_keys = list(self._cur_state.keys())
        self.query_one("#history-table", DataTable).add_columns(
            "Step", "Action", *self._state_keys, *self._output_keys, "Changed"
        )

        self._prev_combined = {}
        self._refresh_all("reset")

    # ── display refresh ──────────────────────────────────────────────────────

    def _refresh_table(self, table_id: str, data: dict, changed: set[str]) -> None:
        tbl = self.query_one(table_id, DataTable)
        tbl.clear()
        for name, val in data.items():
            vs = _fmt_py(val)
            if name in changed:
                ns = f"[bold green]{name}[/bold green]"
                vs = f"[bold green]{vs}[/bold green]"
                ch = "[bold red]✓[/bold red]"
            else:
                ns, ch = name, "[dim]—[/dim]"
            tbl.add_row(ns, vs, ch)

    def _refresh_action_bar(self) -> None:
        parts = []
        for i, label in enumerate(self._labels):
            key = f"[bold]{i}[/bold]"
            if i == self._cur_action:
                parts.append(f" {key} [reverse bold green] {label} [/reverse bold green]")
            else:
                parts.append(f" {key} [dim]{label}[/dim]")
        self.query_one("#action-bar", Static).update(
            "Action: " + "  ".join(parts) + "  [dim]→ Space to step[/dim]"
        )

    def _refresh_all(self, event: str) -> None:
        changed = self._changed() if event != "reset" else set()
        self._refresh_table("#state-table",  self._cur_state,   changed)
        self._refresh_table("#output-table", self._cur_outputs,  changed)
        self._refresh_action_bar()
        self._add_history_row(changed, event)
        self._prev_combined = {**self._cur_state, **self._cur_outputs}

    def _add_history_row(self, changed: set[str], event: str) -> None:
        ht = self.query_one("#history-table", DataTable)
        step_str   = str(self._step)
        action_str = "reset" if event == "reset" else self._labels[self._cur_action]

        def _cell(k):
            v = self._cur_state.get(k, self._cur_outputs.get(k))
            s = _fmt_py(v) if v is not None else "?"
            return f"[bold green]{s}[/bold green]" if k in changed else s

        state_cells  = [_cell(k) for k in self._state_keys]
        output_cells = [_cell(k) for k in self._output_keys]
        changed_str  = ", ".join(sorted(changed)) if changed else "[dim]—[/dim]"

        ht.add_row(step_str, action_str, *state_cells, *output_cells, changed_str)
        ht.scroll_end(animate=False)

    # ── env interaction ──────────────────────────────────────────────────────

    def _run_reset(self) -> None:
        result = self._env.reset()
        # Handle both standard gym (obs, info) and non-standard (obs, reward, terminated, truncated)
        if len(result) == 2:
            obs, info = result
            self._cur_outputs = {"obs": obs, "reward": 0.0, "terminated": False, "truncated": False}
        else:
            obs, reward, terminated, truncated = result[0], result[1], result[2], result[3]
            self._cur_outputs = {"obs": obs, "reward": reward, "terminated": terminated, "truncated": truncated}
        self._cur_state = self._env_state()

    def _run_step(self) -> None:
        result = self._env.step(self._cur_action)
        obs, reward, terminated, truncated = result[0], result[1], result[2], result[3]
        self._cur_outputs = {"obs": obs, "reward": reward, "terminated": terminated, "truncated": truncated}
        self._cur_state = self._env_state()

    # ── key handler (0..n-1 to select action) ────────────────────────────────

    def on_key(self, event) -> None:
        if event.key.isdigit():
            idx = int(event.key)
            if 0 <= idx < self._n:
                self._cur_action = idx
                self._refresh_action_bar()
                event.stop()

    # ── actions ──────────────────────────────────────────────────────────────

    def action_do_step(self) -> None:
        self._step += 1
        self._run_step()
        self._refresh_all("step")

    def action_do_reset(self) -> None:
        self._step = 0
        self._cur_action = 0
        self._prev_combined = {}
        self.query_one("#history-table", DataTable).clear()
        self._run_reset()
        self._refresh_all("reset")
