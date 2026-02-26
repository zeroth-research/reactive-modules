import numpy as np
from .environments import SimpleEnv
from .qnetworks import SimpleQNet
from .agent import DQNAgent
from .train import train


def test_training():
    """Atom param wires should have identical Wire IDs before and after training."""
    num_episodes = 500
    max_steps = 50

    env = SimpleEnv()
    q_network = SimpleQNet(state_size=1, action_size=env.action_space.n, hidden_size=2,)
    agent = DQNAgent(q_network)

    print("\nPARAMETERS BEFORE TRAINING:")
    params_before = []
    for nn_atom in q_network.atoms():
        for param in nn_atom.param:
            print(f"Param wire: {param}, ID: {param.id}")
            params_before.append(param)

    print("\nTRAINING:")
    train(env, agent, num_episodes, max_steps)
    
    print("\nPARAMETERS AFTER TRAINING:")
    params_after = []
    for nn_atom in q_network.atoms():
        for param in nn_atom.param:
            print(f"Param wire: {param}, ID: {param.id}")
            params_after.append(param)

    # Same Wire IDs before and after
    assert len(params_before) == len(params_after)
    for before, after in zip(params_before, params_after):
        assert before == after

if __name__ == "__main__":
    test_training()