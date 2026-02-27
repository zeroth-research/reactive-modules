import numpy as np
from .environments import SimpleEnv
from .qnetworks import SimpleQNet
from .agent import DQNAgent
from .train import train
from zrth import Wire, DType


def test_training():
    """Atom param wires should have identical Wire IDs before and after training."""
    num_episodes = 500
    max_steps = 50

    action = [Wire(DType.Float([2])), Wire(DType.Float([2]))]
    input = [Wire(DType.Float([1])), Wire(DType.Float([1]))]

    env = SimpleEnv(action=action)
    q_network = SimpleQNet(extl=input, state_size=1, action_size=env.action_space.n, hidden_size=2,)
    agent = DQNAgent(q_network)

    print(f'Env action wires: {env.action}')
    print(f'Env observation wires: {env.observation}')
    print(f'Env reward wires: {env.reward}')
    print(f'Env terminated wires: {env.terminated}')
    print(f'Env truncated wires: {env.truncated}')
    print(f'QNetwork input wires: {q_network.extl}')
    print(f'QNetwork output wires: {q_network.intf}')

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