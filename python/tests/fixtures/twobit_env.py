"""TwoBitCounterEnv gym wrapper fixture.

2-bit digital counter with enable action.
Property: b0 = False holds infinitely often (when enable is always on).
"""
from zrth.gym import Wrapper
from zrth import Module
from tests.gym.environments import TwoBitCounterEnv


def module() -> Module:
    return Wrapper(TwoBitCounterEnv())
