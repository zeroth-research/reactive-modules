"""SimpleEnv gym wrapper fixture.

Chain environment: state ∈ {0,1,2}, action moves left/right.
Property: state reaches 2 infinitely often.
"""
from zrth.gym import Env
from zrth import Module, Real
from tests.gym.environments import SimpleEnv


def module() -> Module:
    # `state` is SimpleEnv's only private attribute; its sort must be explicit.
    return Env(SimpleEnv(), attrs=Real([1, 1]))
