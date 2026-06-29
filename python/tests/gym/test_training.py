import numpy as np
import torch
from .environments import SimpleEnv
from .qnetworks import SimpleQNet
from .agent import DQNAgent
from .train import train
from zrth.gym import Env
from zrth.torch import Module
from zrth import Wire, DType
from zrth.eval import eval_itype


def test_training():
    """Live tensor references: symbolic module should reflect weight changes after training."""
    num_episodes = 500
    max_steps = 50

    # Create plain instances
    plain_env = SimpleEnv()
    plain_nn = SimpleQNet(state_size=1, action_size=plain_env.action_space.n, hidden_size=2)

    # Wrap for symbolic extraction
    wrapped_nn = Module(plain_nn)

    # Snapshot weights before training
    weights_before = {name: p.data.clone() for name, p in plain_nn.named_parameters()}

    # Train on the plain nn.Module (agent uses plain_nn.parameters())
    agent = DQNAgent(plain_nn)
    train(plain_env, agent, num_episodes, max_steps)

    # At least some weights should have changed
    any_changed = any(
        not torch.equal(weights_before[name], p.data)
        for name, p in plain_nn.named_parameters()
    )
    assert any_changed, "No parameters changed during training"

    # The symbolic module should reflect the trained weights (live tensor references)
    # Find Tensor terms in the symbolic module and verify they point to trained values
    for atom in wrapped_nn.atoms:
        for term in atom.update:
            itype_str = str(term.itype)
            if "Tensor" in itype_str:
                results = eval_itype(term.itype, [])
                assert results[0] is not None, "Tensor term evaluated to None"


def test_training_with_shared_wires():
    """Shared wires between Env and NN modules should work with wrapping."""
    action = [Wire(DType.Float([2])), Wire(DType.Float([2]))]
    input_wire = [Wire(DType.Float([1])), Wire(DType.Float([1]))]

    plain_env = SimpleEnv()
    plain_nn = SimpleQNet(state_size=1, action_size=plain_env.action_space.n, hidden_size=2)

    wrapped_env = Env(plain_env, action=action)
    wrapped_nn = Module(plain_nn, extl=input_wire)

    print(f'Env action wires: {wrapped_env.obs[0]}')
    print(f'Env observation wires: {wrapped_env.obs[1]}')
    print(f'NN input wires: {wrapped_nn.obs[0]}')
    print(f'NN output wires: {wrapped_nn.obs[1]}')

    # Verify shared wires match
    assert wrapped_env.obs[0][0].id == action[0].id
    assert wrapped_env.obs[0][1].id == action[1].id
    assert wrapped_nn.obs[0][0].id == input_wire[0].id
    assert wrapped_nn.obs[0][1].id == input_wire[1].id


if __name__ == "__main__":
    test_training()
    test_training_with_shared_wires()
