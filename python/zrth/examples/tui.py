"""Generic textual TUI for stepping through any reactive module.

Two apps are available:

``InterpreterApp`` — for hand-built modules (e.g. the 2-bit counter)::

    from zrth.examples.tui import InterpreterApp
    import torch

    app = InterpreterApp(
        module=my_module,
        observe={"b1": b1[0], "b0": b0[0]},
        inputs={"enable": (enable[1], torch.tensor([False]))},
        title="2-Bit Counter",
    )
    app.run()

``EnvApp`` — for gym ``Env`` subclasses; wires are discovered automatically::

    from zrth.examples.tui import EnvApp
    from my_envs import SimpleEnv

    EnvApp(SimpleEnv(), action_labels=["left", "right"]).run()

API — InterpreterApp
--------------------
observe : dict[str, Wire]
    Wires whose values are shown after every step.  These should be the
    *latched* (current) side of each state variable.

inputs : dict[str, tuple[Wire, Tensor]], optional
    External-input wires together with their default values.  Boolean
    scalar inputs can be toggled interactively with the number keys
    1, 2, …  Other tensor shapes are displayed read-only.

title : str
    App title shown in the header.
"""
import torch
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer, DataTable, Label, Static
from textual.containers import Horizontal, Vertical

from .interpreter import Interpreter


# ── tensor helpers ────────────────────────────────────────────────────────────

def _fmt(t: torch.Tensor) -> str:
    """Compact string for a tensor value."""
    if t.numel() == 1:
        v = t.item()
        if t.dtype == torch.bool:
            return "T" if bool(v) else "F"
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(int(v))
    return str(t.tolist())


def _is_bool_scalar(t: torch.Tensor) -> bool:
    return t.dtype == torch.bool and t.numel() == 1


# ── app ───────────────────────────────────────────────────────────────────────

class InterpreterApp(App):
    """Generic TUI for stepping through any reactive module with the Interpreter.

    Controls
    --------
    Space / Enter  step with current inputs
    1, 2, …        toggle boolean input N
    R              reset (re-run init, clear history)
    Q              quit
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

    #input-panel {
        width: 1fr;
        border: round #30363d;
        padding: 0 1;
    }

    .panel-title {
        color: #8b949e;
        margin-bottom: 0;
    }

    #input-hint {
        color: #444d56;
        margin-top: 0;
    }

    #history-label {
        color: #8b949e;
        margin: 1 2 0 2;
    }

    DataTable {
        margin: 0 2 1 2;
        height: 1fr;
    }

    #state-table { height: auto; }
    """

    BINDINGS = [
        Binding("space",  "do_step",  "Step",        priority=True),
        Binding("enter",  "do_step",  "Step",        show=False, priority=True),
        Binding("r",      "do_reset", "Reset"),
        Binding("q",      "quit",     "Quit"),
    ]

    # ── construction ─────────────────────────────────────────────────────────

    def __init__(
        self,
        module,
        observe: dict,
        inputs: dict | None = None,
        title: str = "Reactive Module Interpreter",
    ):
        """
        Parameters
        ----------
        module
            A ``Module`` built with ``Module.sequential`` / ``Module.parallel``.
        observe : dict[str, Wire]
            Mapping from display name to the *latched* Wire to watch.
        inputs : dict[str, tuple[Wire, Tensor]], optional
            Mapping from display name to ``(Wire, default_tensor)``.
            Boolean scalars can be toggled interactively.
        title : str
            Shown in the app header.
        """
        super().__init__()
        self._module = module
        self._observe = observe
        self._inputs: dict = inputs or {}
        self._app_title = title

        self._interp = Interpreter(module)

        # mutable current input values (reset to defaults on R)
        self._cur_inputs: dict[str, torch.Tensor] = {
            name: val.clone()
            for name, (_, val) in self._inputs.items()
        }
        self._input_keys: list[str] = list(self._inputs.keys())

        self._step = 0
        self._prev_state: dict[str, torch.Tensor] = {}
        # track column layout so history rows match headers
        self._has_inputs = bool(self._inputs)

    # ── compose ───────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top-row"):
            with Vertical(id="state-panel"):
                yield Label("Observed state", classes="panel-title")
                yield DataTable(id="state-table", cursor_type="none")
            with Vertical(id="input-panel"):
                yield Label("Inputs", classes="panel-title")
                yield Static("", id="input-display")
                yield Label(
                    "[dim]1-9: toggle bool input  •  Space: step[/dim]",
                    id="input-hint",
                )
        yield Label("── History ──", id="history-label")
        yield DataTable(id="history-table", cursor_type="none")
        yield Footer()

    # ── startup ───────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self.title = self._app_title
        # state table
        st = self.query_one("#state-table", DataTable)
        st.add_columns("Wire", "Value", "Δ")

        # history table — column layout depends on whether we have inputs
        ht = self.query_one("#history-table", DataTable)
        observe_names = list(self._observe.keys())
        if self._has_inputs:
            ht.add_columns("Step", *self._input_keys, *observe_names, "Changed")
        else:
            ht.add_columns("Step", *observe_names, "Changed")

        # run init
        self._interp.initialize()
        self._prev_state = {}
        self._update_all("reset")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _read_state(self) -> dict[str, torch.Tensor]:
        return {
            name: self._interp.get(wire.id())
            for name, wire in self._observe.items()
        }

    def _changed(self, state: dict[str, torch.Tensor]) -> set[str]:
        changed: set[str] = set()
        for name, val in state.items():
            prev = self._prev_state.get(name)
            if prev is not None and not torch.equal(val, prev):
                changed.add(name)
        return changed

    def _update_all(self, action: str) -> None:
        state = self._read_state()
        changed = self._changed(state) if action != "reset" else set()
        self._refresh_state_table(state, changed)
        self._refresh_input_display()
        self._add_history_row(state, changed, action)
        self._prev_state = state

    def _refresh_state_table(
        self,
        state: dict[str, torch.Tensor],
        changed: set[str],
    ) -> None:
        st = self.query_one("#state-table", DataTable)
        st.clear()
        for name, val in state.items():
            val_str = _fmt(val)
            if name in changed:
                name_str = f"[bold green]{name}[/bold green]"
                val_str = f"[bold green]{val_str}[/bold green]"
                ch_str = "[bold red]✓[/bold red]"
            else:
                name_str = name
                ch_str = "[dim]—[/dim]"
            st.add_row(name_str, val_str, ch_str)

    def _refresh_input_display(self) -> None:
        if not self._inputs:
            self.query_one("#input-display", Static).update(
                "[dim](no external inputs)[/dim]"
            )
            return
        lines = []
        for i, name in enumerate(self._input_keys):
            val = self._cur_inputs[name]
            key_hint = f"[bold]{i + 1}[/bold]"
            if _is_bool_scalar(val):
                ind = "[bold green]ON [/bold green]" if val.item() else "[dim]OFF[/dim]"
                lines.append(f" {key_hint}  {name:<16} {ind}")
            else:
                lines.append(f" {key_hint}  {name:<16} {_fmt(val)}")
        self.query_one("#input-display", Static).update("\n".join(lines))

    def _build_env_inputs(self) -> dict:
        return {
            wire.id(): self._cur_inputs[name]
            for name, (wire, _) in self._inputs.items()
        }

    def _add_history_row(
        self,
        state: dict[str, torch.Tensor],
        changed: set[str],
        action: str,
    ) -> None:
        ht = self.query_one("#history-table", DataTable)

        step_str = str(self._step)
        state_cells = []
        for name, val in state.items():
            s = _fmt(val)
            if name in changed:
                s = f"[bold green]{s}[/bold green]"
            state_cells.append(s)

        changed_str = ", ".join(sorted(changed)) if changed else "[dim]—[/dim]"

        if action == "reset":
            if self._has_inputs:
                input_cells = ["—"] * len(self._input_keys)
            else:
                input_cells = []
        else:
            input_cells = [_fmt(self._cur_inputs[n]) for n in self._input_keys]

        if self._has_inputs:
            ht.add_row(step_str, *input_cells, *state_cells, changed_str)
        else:
            ht.add_row(step_str, *state_cells, changed_str)
        ht.move_cursor(row=ht.row_count - 1)

    # ── key handler (number keys for input toggling) ──────────────────────────

    def on_key(self, event) -> None:
        key = event.key
        if key.isdigit():
            idx = int(key) - 1
            if 0 <= idx < len(self._input_keys):
                name = self._input_keys[idx]
                val = self._cur_inputs[name]
                if _is_bool_scalar(val):
                    self._cur_inputs[name] = torch.tensor([not val.item()])
                    self._refresh_input_display()
                    event.stop()

    # ── actions ───────────────────────────────────────────────────────────────

    def action_do_step(self) -> None:
        self._step += 1
        self._interp.step(self._build_env_inputs())
        self._update_all("step")

    def action_do_reset(self) -> None:
        self._interp.initialize()
        self._step = 0
        self._cur_inputs = {
            name: val.clone()
            for name, (_, val) in self._inputs.items()
        }
        self._prev_state = {}
        self.query_one("#history-table", DataTable).clear()
        self._update_all("reset")


# ══════════════════════════════════════════════════════════════════════════════
# EnvApp — for gym Env subclasses (Python-level simulation)
# ══════════════════════════════════════════════════════════════════════════════

# Gymnasium attributes that are not part of user-defined state
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
    """TUI for stepping through any ``zrth.gym.Env`` subclass.

    Uses Python-level simulation: calls ``env.reset()`` / ``env.step(q_values)``
    directly and inspects ``env.__dict__`` for state variables.  No changes to
    ``zrth_module.py`` are required.

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
        """
        Parameters
        ----------
        env
            A ``zrth.gym.Env`` subclass instance.
        action_labels : list[str], optional
            Human-readable label for each action index.  Defaults to "0", "1", …
        title : str, optional
            App title.  Defaults to the env class name.
        """
        super().__init__()
        self._env = env
        self._n: int = env.action_space.n
        self._labels = action_labels or [str(i) for i in range(self._n)]
        self._app_title = title or type(env).__name__

        self._cur_action = 0
        self._step = 0

        # Discovered after first reset() — kept stable across resets
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

    def _action_tensor(self) -> torch.Tensor:
        t = torch.zeros(self._n)
        t[self._cur_action] = 1.0
        return t

    def _changed(self) -> set[str]:
        combined = {**self._cur_state, **self._cur_outputs}
        return {
            k for k, v in combined.items()
            if k in self._prev_combined and self._prev_combined[k] != v
        }

    # ── compose ───────────────────────────────────────────────────────────────

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

    # ── startup ───────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self.title = self._app_title
        self.query_one("#state-table",  DataTable).add_columns("Variable", "Value", "Δ")
        self.query_one("#output-table", DataTable).add_columns("Output",   "Value", "Δ")

        # Run reset to discover state keys, then set up history columns
        self._run_reset()
        self._state_keys = list(self._cur_state.keys())
        self.query_one("#history-table", DataTable).add_columns(
            "Step", "Action", *self._state_keys, *self._output_keys, "Changed"
        )

        self._prev_combined = {}
        self._refresh_all("reset")

    # ── display refresh ───────────────────────────────────────────────────────

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

    # ── env interaction ───────────────────────────────────────────────────────

    def _run_reset(self) -> None:
        obs, reward, terminated, truncated = self._env.reset()
        self._cur_outputs = {
            "obs": obs, "reward": reward,
            "terminated": terminated, "truncated": truncated,
        }
        self._cur_state = self._env_state()

    def _run_step(self) -> None:
        obs, reward, terminated, truncated = self._env.step(self._action_tensor())
        self._cur_outputs = {
            "obs": obs, "reward": reward,
            "terminated": terminated, "truncated": truncated,
        }
        self._cur_state = self._env_state()

    # ── key handler (0..n-1 to select action) ────────────────────────────────

    def on_key(self, event) -> None:
        if event.key.isdigit():
            idx = int(event.key)
            if 0 <= idx < self._n:
                self._cur_action = idx
                self._refresh_action_bar()
                event.stop()

    # ── actions ───────────────────────────────────────────────────────────────

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
