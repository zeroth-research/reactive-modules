import torch
import torch.nn as nn
from .zrth_module import NN


class SimpleQNet(NN):
    """Simple Q-Network with one hidden layer"""
    
    def __init__(self, state_size, action_size, hidden_size, seed=42):
        """Initialize Q-Network

        Args:
            state_size: Dimension of input state
            action_size: Number of actions
            hidden_size: Hidden layer size
            seed: Random seed
        """
        super().__init__()
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)


# class GridWorldQNet(nn.Module, Module):
#     """Three-layer Q-Network for grid world"""
    
#     extl = ["observation: Tensor<1; Float>"]
#     intf = ["q_values: Tensor<4; Float>"]
    
#     def __init__(self, state_size, action_size, hidden_size1, hidden_size2, seed=42):
#         """Initialize three-layer Q-Network

#         Args:
#             state_size: Dimension of input state
#             action_size: Number of actions
#             hidden_size1: First hidden layer size
#             hidden_size2: Second hidden layer size
#             seed: Random seed
#         """
#         nn.Module.__init__(self)
        
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, hidden_size1)
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         self.fc3 = nn.Linear(hidden_size2, action_size)

#     def forward(self, state):
#         x = torch.relu(self.fc1(state))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)
