from . import Context, SimpleEnv, DQNAgent
import numpy as np
import torch
import random
    
if __name__ == "__main__":
    # Create context
    ctx = Context()
    
    # Initialize environment with automatic conversion
    env = SimpleEnv(
        ctx,
        names={'extl': ['q_values'], 'intf': ['observation', 'reward', 'terminated'], 'prvt': ['state']}
    )
    
    state_size = 1  # Single binary observation (0 or 1)
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Training parameters
    num_episodes = 500
    max_steps = 50

    # Training
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.array([state], dtype=np.float32)
        total_reward = 0
        
        for step in range(max_steps):
            # Get Q-values from network
            q_values = agent.get_q_values(state)

            # Epsilon-greedy: sometimes use random Q-values for exploration
            if random.random() < agent.epsilon:
                # Random exploration: create random Q-values
                q_values = torch.FloatTensor([random.random(), random.random()])
            
            # Environment does argmax and executes
            next_state, reward, terminated, truncated, _ = env.step(q_values)
            next_state = np.array([next_state], dtype=np.float32)
            done = terminated or truncated

            # For training, we need the actual action taken
            action = q_values.argmax().item()
            
            # Train agent on this transition
            agent.train(state, action, reward, next_state, float(done))

            total_reward += reward
            state = next_state
            
            if done:
                break
        
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
    
    # Test the trained agent
    state, _ = env.reset()
    state = np.array([state], dtype=np.float32)
    agent.epsilon = 0  # No exploration, pure exploitation

    print("\nTesting trained agent:")
    for step in range(10):
        q_values = agent.get_q_values(state)
        next_state, reward, terminated, truncated, _ = env.step(q_values)
        action = q_values.argmax().item()  # For display
        print(f"State: {int(state[0])}, Action: {action}, Q-values: {q_values.tolist()}, Reward: {reward}, Next State: {next_state}")
        
        if terminated or truncated:
            print("Goal reached!")
            break
