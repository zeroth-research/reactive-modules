from gymnasium import spaces
from .zrth_module import Env

# TODO: remove this when we can get the DType from the analyzer instead of hardcoding it here
from zrth.zrth import DType

class SimpleEnv(Env):
    """Simple chain environment with partial observability"""

    # TODO: remove this when we can get the DType from the analyzer instead of hardcoding it here
    q_values: DType.Float([2])
    observation: DType.Float([1])
    state: DType.Float([1])
    
    def __init__(self):
        """Initialize simple chain environment"""
        super().__init__()
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)

    def _get_observation(self):
        # Agent only sees if it's at goal or not (partial observability)
        return 1 if self.state == 2 else 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
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
        terminated = self._get_observation() == 1
        truncated = False

        observation = self._get_observation()  # Get partial observation
        return observation, reward, terminated, truncated


class GridWorldEnv(Env):
    """3x3 grid world environment"""

    # TODO: remove this when we can get the DType from the analyzer instead of hardcoding it here
    q_values: DType.Float([4])
    observation: DType.Float([1])
    x: DType.Float([1])
    y: DType.Float([1])
    
    def __init__(self):
        """Initialize 3x3 grid world environment"""
        super().__init__()
        
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Discrete(9)  # 3x3 grid positions

        self.goal_x = 2
        self.goal_y = 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = 0  # Start at origin
        self.y = 0
        observation = self.y * 3 + self.x  # Flatten 2D position to 1D
        reward = 0.0
        terminated = False
        truncated = False
        return observation, reward, terminated, truncated
    
    def step(self, q_values):
        action = q_values.argmax().item()
        
        # Update position based on action (with boundary checking)
        if action == 0:  # up
            self.y = max(self.y - 1, 0)
        elif action == 1:  # down
            self.y = min(self.y + 1, 2)
        elif action == 2:  # left
            self.x = max(self.x - 1, 0)
        else:  # right (action == 3)
            self.x = min(self.x + 1, 2)
        
        # Check if reached goal
        at_goal_x = self.x == 2
        at_goal_y = self.y == 2
        at_goal = at_goal_x and at_goal_y
        
        reward = 1.0 if at_goal else 0.0
        terminated = at_goal
        truncated = False
        
        observation = self.y * 3 + self.x  # Flatten position
        return observation, reward, terminated, truncated

class ComplexDecisionEnv(Env):
    """Environment testing: nested decisions, boolean ops, augmented assignments, numpy"""

    # TODO: remove this when we can get the DType from the analyzer instead of hardcoding it here
    q_values: DType.Float([10])
    observation: DType.Float([1])
    score: DType.Float([1])
    multiplier: DType.Float([1])
    bonus_active: DType.Bool([1])
    
    def __init__(self):
        """Initialize complex decision environment
        
        Tests:
        - Nested if/elif/else (3 levels deep)
        - Boolean operations (and, or, not)
        - Augmented assignments (+=, -=, *=)
        """
        super().__init__()
        
        self.action_space = spaces.Discrete(10)  # 10 possible actions
        self.observation_space = spaces.Box(low=0, high=10, shape=(1,))
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.score = 0.0
        self.multiplier = 1.0
        self.bonus_active = False
        
        observation = self.score
        reward = 0.0
        terminated = False
        truncated = False
        return observation, reward, terminated, truncated
    
    def step(self, q_values):
        action = q_values.argmax().item()
        
        # Test augmented assignments
        self.score += 1.0  # Increment score every step
        
        # Test nested if/elif/else with boolean operations
        if action < 3:
            # Low actions (0, 1, 2)
            if action == 0:
                self.multiplier *= 2.0  # Double multiplier
                self.bonus_active = True
            elif action == 1:
                self.multiplier -= 0.5  # Decrease multiplier
                # Test boolean AND
                if self.multiplier > 0.0 and self.bonus_active:
                    self.score += 5.0
            else:  # action == 2
                self.bonus_active = False
                self.score -= 1.0  # Penalty
        elif action < 7:
            # Mid actions (3, 4, 5, 6)
            if action == 3:
                # Test boolean OR
                if self.score > 10.0 or self.bonus_active:
                    self.multiplier += 1.0
            elif action == 4:
                # Test boolean NOT
                if not self.bonus_active:
                    self.score += 2.0
            elif action == 5:
                # Test nested boolean operations
                high_score = self.score > 5.0
                has_bonus = self.bonus_active
                good_multiplier = self.multiplier >= 1.0
                
                # (high_score AND has_bonus) OR good_multiplier
                if (high_score and has_bonus) or good_multiplier:
                    self.score *= 1.5
            else:  # action == 6
                # Reset bonus but keep score
                self.bonus_active = False
                self.multiplier = 1.0
        else:
            # High actions (7, 8, 9)
            if action == 7:
                # Test comparison chains (if supported)
                if self.score >= 0.0:  # Simplified for now
                    if self.score <= 20.0:
                        self.bonus_active = True
            elif action == 8:
                # Big bonus with conditions
                if self.multiplier > 1.0:
                    bonus = self.multiplier * 10.0
                    self.score += bonus
            else:  # action == 9
                # Reset everything
                self.score = 0.0
                self.multiplier = 1.0
                self.bonus_active = False
        
        # Calculate reward using private state
        reward = self.score * self.multiplier
        
        # Terminate if score too high or too low
        terminated = self.score > 100.0 or self.score < -10.0
        truncated = False
        
        observation = self.score
        return observation, reward, terminated, truncated

class EarlyReturnEnv(Env):
    """Environment testing early returns in if/else branches"""

    # TODO: remove this when we can get the DType from the analyzer instead of hardcoding it here
    q_values: DType.Float([5])
    observation: DType.Float([1])
    counter: DType.Float([1])
    
    def __init__(self):
        """Initialize early return test environment
        
        Tests:
        - Early returns inside if/else branches
        - Return value merging across branches
        """
        super().__init__()
        
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(1,))
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.counter = 0.0
        observation = self.counter
        reward = 0.0
        terminated = False
        truncated = False
        return observation, reward, terminated, truncated
    
    def step(self, q_values):
        action = q_values.argmax().item()
        
        # Early return cases
        if action == 0:
            # Immediate termination with zero reward
            observation = 0.0
            reward = 0.0
            terminated = True
            truncated = False
            return observation, reward, terminated, truncated
        
        if action == 1:
            # Small penalty and terminate
            observation = self.counter
            reward = -1.0
            terminated = True
            truncated = False
            return observation, reward, terminated, truncated
        
        # Normal processing for other actions
        self.counter += 1.0
        
        if action == 2:
            reward = 1.0
        elif action == 3:
            reward = 2.0
        else:
            reward = 0.5
        
        observation = self.counter
        terminated = False
        truncated = False
        return observation, reward, terminated, truncated


class ComparisonChainEnv(Env):
    """Environment testing comparison chains (a < b < c)"""

    # TODO: remove this when we can get the DType from the analyzer instead of hardcoding it here
    q_values: DType.Float([5])
    observation: DType.Float([1])
    value: DType.Float([1])
    
    def __init__(self):
        """Initialize comparison chain test environment
        
        Tests:
        - Comparison chains: 0 < x < 10
        - Multiple chained comparisons
        """
        super().__init__()
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=20, shape=(1,))
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.value = 5.0
        observation = self.value
        reward = 0.0
        terminated = False
        truncated = False
        return observation, reward, terminated, truncated
    
    def step(self, q_values):
        action = q_values.argmax().item()
        
        # Update value based on action
        if action == 0:
            self.value -= 2.0
        elif action == 1:
            self.value += 0.0  # No change
        else:
            self.value += 3.0
        
        # Use comparison chains to determine reward
        if 0.0 < self.value < 10.0:
            # Value is in the sweet spot
            reward = 10.0
        elif 10.0 <= self.value <= 15.0:
            # Value is acceptable
            reward = 5.0
        else:
            # Value is out of range
            reward = 0.0
        
        # Terminate if value goes negative or too high
        terminated = self.value < 0.0 or self.value > 20.0
        truncated = False
        
        observation = self.value
        return observation, reward, terminated, truncated
