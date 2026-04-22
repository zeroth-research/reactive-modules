"""SimpleEnv gym wrapper fixture.

Chain environment: state ∈ {0,1,2}, action moves left/right.
Property: state reaches 2 infinitely often.
"""
from zrth.gym import Wrapper
from zrth import Module
from tests.gym.environments import SimpleEnv


def module() -> Module:
    return Wrapper(SimpleEnv())
