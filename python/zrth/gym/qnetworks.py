import torch
import torch.nn as nn
from .zrth_module import Module


class SimpleQNet(nn.Module, Module):
    def __init__(self, state_size, action_size, hidden_size, extl, intf, seed=42):
        """Initialize Q-Network

        Args:
            state_size: Dimension of input state
            action_size: Number of actions
            hidden_size: Hidden layer size
            extl: List of external input wire names
            intf: List of interface output wire names
            seed: Random seed
        """
        nn.Module.__init__(self)
        Module.__init__(self, extl, intf)
        
        # Neural network layers
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)  # Output Q-values for each action


class GridWorldQNet(nn.Module, Module):
    def __init__(self, state_size, action_size, hidden_size1, hidden_size2, extl, intf, seed=42):
        """Initialize three-layer Q-Network

        Args:
            state_size: Dimension of input state
            action_size: Number of actions
            hidden_size1: First hidden layer size
            hidden_size2: Second hidden layer size
            extl: List of external input wire names
            intf: List of interface output wire names
            seed: Random seed
        """
        nn.Module.__init__(self)
        Module.__init__(self, extl, intf)
        
        # Neural network layers
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-values for each action

