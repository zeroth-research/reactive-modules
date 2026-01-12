import gymnasium as gym
from gymnasium import spaces
from .zrth_module import Module


class SimpleEnv(gym.Env, Module):
    def __init__(self, names, ctx=None):
        """Initialize simple chain environment
        
        Args:
            names: Dictionary with wire declarations, e.g. {'extl': ['q_values'], 'intf': ['observation', 'reward', 'terminated'], 'prvt': ['state']}
            ctx: Context object for wire registry (if None, uses global shared context)
        """
        gym.Env.__init__(self)
        Module.__init__(self, ctx, names)
        
        # Gym spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)
        
        self.state = 0  # True state (private, not directly observable)
        
        # _finalize_conversion() is called automatically after this by __init_subclass__
        
    def _get_observation(self):
        # Agent only sees if it's at goal or not (partial observability)
        return 1 if self.state == 2 else 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0  # True state (private)
        return self._get_observation(), {}  # Return partial observation
    
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
        terminated = (self.state == 2)
        truncated = False
        
        observation = self._get_observation()  # Get partial observation
        return observation, reward, terminated, truncated, {}