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
