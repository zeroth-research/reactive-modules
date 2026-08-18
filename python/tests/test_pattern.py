import pytest
import gymnasium as gym
from zrth import Module, Wire, Term, Sort, LIA, Bool, Real, Int, BitVec, Var, X
from gymnasium import spaces


def convert_method(method, read: dict[str, Wire], write: dict[str, Wire], result: list[Wire]):
    # `Uninterpreted` is single-read-or-single-write; emit one Term per wire
    # so the multi-arity "method" is modeled as a bag of independent
    # uninterpreted reads / writes.
    terms = []
    for r in read.values():
        terms.append(Term(LIA.Uninterpreted("method"), [], [r]))
    for w in list(write.values()) + list(result):
        terms.append(Term.constant(LIA.Uninterpreted("method"), [w]))
    return terms


class SimpleEnv(gym.Env, Module):

    def __new__(cls, *args, **kwargs):
        q_values = Var(Real([1, 1]))
        observation = Var(Real([1, 1]))
        reward = Var(Real([1, 1]))
        terminated = Var(Real([1, 1]))
        truncated = Var(Real([1, 1]))
        state = Var(Real([1, 1]))

        result = (X(observation), X(reward), X(terminated), X(truncated))
        reset = convert_method(cls.reset, read={}, write={"self.state": X(state)}, result=result)
        step = convert_method(cls.step, read={"q_value": q_values, "self.state": state},
                              write={"self.state": X(state)}, result=result)

        obs = [q_values, observation, reward, terminated, truncated]
        prvt = [state]
        return super().__new__(cls, init=reset, update=step, obs=obs, prvt=prvt)

    def __init__(self):
        """Initialize simple chain environment"""
        gym.Env.__init__(self)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)

    def _get_observation(self):
        # Agent only sees if it's at goal or not (partial observability)
        return 1 if self.state == 2 else 0

    def reset(self):
        super().reset(seed=None)
        self.state = 0  # True state (private)
        observation = self._get_observation()
        reward = 0.0  # No reward at reset
        terminated = False  # Episode just started
        truncated = False
        return observation, reward, terminated, truncated

    def step(self, q_values):
        action = q_values.argmax().item()
        # action 0 = move left, action 1 = move right
        if action == 1:  # right
            self.state = min(self.state + 1, 2)
        else:  # left
            self.state = max(self.state - 1, 0)

        # Reward +1 if we reach state 2, otherwise 0
        reward = 1.0 if self.state == 2 else 0.0

        # Episode terminates when reaching state 2
        terminated = self.state == 2
        truncated = False

        observation = self._get_observation()  # Get partial observation
        return observation, reward, terminated, truncated


def test_instantiate():
    e = SimpleEnv()
    print(type(e))
