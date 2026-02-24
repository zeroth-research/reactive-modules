import torch
import torch.nn as nn
import torch.optim as optim


class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size, lr=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        from .qnetworks import SimpleQNet
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        
        # Q-network
        self.q_network = SimpleQNet(
            extl=["observation: Tensor<1; Float>"],
            intf=["q_values: Tensor<2; Float>"],
            state_size=1,
            action_size=2,
            hidden_size=2,
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Exploration rate
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        
    def get_q_values(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.squeeze()
    
    def train(self, state, action, reward, next_state, done):
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.LongTensor([action])
        reward_tensor = torch.FloatTensor([reward])
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        done_tensor = torch.FloatTensor([done])
        
        # Current Q-value
        current_q = self.q_network(state_tensor).gather(1, action_tensor.unsqueeze(1))
        
        # Target Q-value
        with torch.no_grad():
            max_next_q = self.q_network(next_state_tensor).max(1)[0]
            target_q = reward_tensor + (1 - done_tensor) * self.gamma * max_next_q
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q.squeeze())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)