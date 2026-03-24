"""Step through the 2-bit counter using Wrapper.

Run with:
    cd python && uv run python ui/run_twobitcounter.py
"""
import sys
sys.path.insert(0, "tests")

from gym.environments import TwoBitCounterEnv
from zrth.gym import Wrapper


counter = TwoBitCounterEnv()
wrapped = Wrapper(counter)


def show(wrapped, label=""):
    tag = f"  <- {label}" if label else ""
    print(f"  b1={wrapped.b1}  b0={wrapped.b0}  ({int(wrapped.b1) * 2 + int(wrapped.b0)}){tag}")


ENABLE = 1
HOLD = 0

print("=== init ===")
wrapped.reset()
show(wrapped, "reset")

print("\n=== count up (enable=1 each step) ===")
for _ in range(4):
    wrapped.step(ENABLE)
    show(wrapped)

print("\n=== hold (enable=0) ===")
wrapped.step(ENABLE)
show(wrapped, "count to 1")
for _ in range(3):
    wrapped.step(HOLD)
    show(wrapped, "hold")

print("\n=== mixed ===")
wrapped.reset()
show(wrapped, "reset")
for label, action in [
    ("enable", ENABLE), ("hold", HOLD), ("enable", ENABLE),
    ("hold", HOLD), ("enable", ENABLE), ("enable", ENABLE),
]:
    wrapped.step(action)
    show(wrapped, label)
