"""TwoBitCounterEnv gym wrapper fixture.

2-bit digital counter with enable action.
Property: b0 = False holds infinitely often (when enable is always on).
"""
from zrth.gym import Env
from zrth import Module, Bool, LIA
from tests.gym.environments import TwoBitCounterEnv


def module() -> Module:
    # `b0`/`b1` are TwoBitCounterEnv's private state; their sorts must be explicit.
    return Env(TwoBitCounterEnv(), attrs=Bool([1, 1]), theory=LIA)
