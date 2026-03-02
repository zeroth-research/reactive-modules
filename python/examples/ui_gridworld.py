"""Interactive TUI for GridWorldEnv — navigate a 3×3 grid to the corner.

Run with:
    cd python && uv run python examples/ui_gridworld.py

Actions:
    0  up
    1  down
    2  left
    3  right

Goal: reach (x=2, y=2) — state 8 in flattened encoding.
"""
import sys
sys.path.insert(0, "tests")

from gym.environments import GridWorldEnv
from tui import EnvApp

EnvApp(
    GridWorldEnv(),
    action_labels=["up", "down", "left", "right"],
    title="GridWorldEnv — 3×3 grid",
).run()
