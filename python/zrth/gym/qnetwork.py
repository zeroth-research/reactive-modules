import torch
import torch.nn as nn
from .zrth_module import Module


class QNetwork(nn.Module, Module):
    def __init__(self, names, state_size, action_size, hidden_size=2, seed=42, ctx=None):
        """Initialize Q-Network
        
        Args:
            names: Dictionary with wire declarations, e.g. {'extl': ['observation'], 'intf': ['q_values'], 'prvt': []}
            state_size: Dimension of input state
            action_size: Number of actions
            hidden_size: Hidden layer size
            seed: Random seed
            ctx: Context object for wire registry (if None, uses global shared context)
        """
        nn.Module.__init__(self)
        Module.__init__(self, ctx, names)
        
        # Neural network layers
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)  # Output Q-values for each action