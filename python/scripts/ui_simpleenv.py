"""Interactive TUI for SimpleEnv — a 3-state chain with partial observability.

Run with:
    cd python && uv run python scripts/ui_simpleenv.py

Actions:
    0  left  (move state toward 0)
    1  right (move state toward 2; reaching 2 = goal, terminates episode)
"""
import sys
sys.path.insert(0, "tests")

from gym.environments import SimpleEnv
from zrth.examples import EnvApp

EnvApp(
    SimpleEnv(),
    action_labels=["left", "right"],
    title="SimpleEnv — 3-state chain",
).run()
