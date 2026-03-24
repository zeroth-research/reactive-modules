"""Interactive TUI for the 2-bit counter — powered by Wrapper + EnvApp.

Run with:
    cd python && uv run python ui/ui_twobitcounter.py
"""
import sys
sys.path.insert(0, "tests")

from gym.environments import TwoBitCounterEnv
from zrth.gym import Wrapper
from tui import EnvApp

counter = TwoBitCounterEnv()
wrapped = Wrapper(counter)

EnvApp(
    wrapped,
    action_labels=["hold", "enable"],
    title="2-Bit Counter",
).run()
