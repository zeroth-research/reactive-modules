"""Interactive TUI for SimpleEnv — a 3-state chain with partial observability.

Run with:
    cd python && uv run python ui/ui_simpleenv.py

Actions:
    0  left  (move state toward 0)
    1  right (move state toward 2; reaching 2 = goal, terminates episode)
"""
import sys
sys.path.insert(0, "tests")

from gym.environments import SimpleEnv
from zrth.gym import Wrapper
from tui import EnvApp

simple = SimpleEnv()
wrapped = Wrapper(simple)

EnvApp(
    wrapped,
    action_labels=["left", "right"],
    title="SimpleEnv — 3-state chain",
).run()
