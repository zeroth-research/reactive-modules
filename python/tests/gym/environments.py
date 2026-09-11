"""Reusable gym.Env fixtures for the test suite.

We keep this minimal: most analyzer-feature tests define their own tiny env
inline (in test_analyzer.py). SimpleEnv is the one we share — used by
test_env.py (Env-class surface) and test_training.py (DQN training)."""

import gymnasium as gym
from gymnasium import spaces


class SimpleEnv(gym.Env):
    """Chain env: state moves left/right; reward 1 at goal (state == 2)."""

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)

    def _get_observation(self):
        return 1 if self.state == 2 else 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self._get_observation(), {}

    def step(self, q_values):
        action = q_values.argmax().item()
        if action == 1:
            self.state = min(self.state + 1, 2)
        else:
            self.state = max(self.state - 1, 0)
        reward = 1.0 if self.state == 2 else 0.0
        terminated = self._get_observation() == 1
        return self._get_observation(), reward, terminated, False, {}


class TwoBitCounterEnv(gym.Env):
    """2-bit counter: the action enables a tick; b0 toggles and carries into b1.

    The gym twin of the `twobit_lia` fixture's hand-built module — same two
    bits, same `enable`, so the two encodings of one system can be compared."""

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2)  # enable: off / on
        self.observation_space = spaces.Discrete(2)

    def _get_observation(self):
        return 1 if self.b0 and self.b1 else 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.b0 = False
        self.b1 = False
        return self._get_observation(), {}

    def step(self, q_values):
        enable = q_values.argmax().item() == 1
        carry = self.b0 and enable  # reads b0 before the toggle below
        self.b0 = (not self.b0) if enable else self.b0
        self.b1 = (not self.b1) if carry else self.b1
        reward = 1.0 if self.b0 and self.b1 else 0.0
        return self._get_observation(), reward, False, False, {}
