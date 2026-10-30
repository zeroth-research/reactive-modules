import math

import pytest
import torch
import torch.nn as nn

from zrth import Module as compose
from zrth.petri import Arc, Place, Transition
from zrth.rl import reinforce, returns_to_go, rollout


def gated_counter():
    go, out = Arc("enabling"), Arc("production")
    produce = Transition(rate=1.0, inputs=[go], outputs=[out])
    count = Place(inputs=[out])
    return compose(produce, count), go, count


def test_rollout_covers_the_horizon_one_entry_per_step():
    system, go, count = gated_counter()
    logps, rewards, lengths = rollout(
        system,
        lambda state: ({go.enabled: True}, torch.tensor(0.0)),
        horizon=5.0,
        reward=lambda state, dt: state[count.count] * dt,
        seed=1,
    )
    assert len(logps) == len(rewards) == len(lengths) > 3
    assert math.isclose(sum(lengths), 5.0)
    assert all(dt > 0 for dt in lengths)


def test_returns_to_go_undiscounted_and_discounted():
    assert returns_to_go([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) == [6.0, 5.0, 3.0]
    g = returns_to_go([1.0, 1.0], [2.0, 1.0], discount=math.log(2))
    assert g == pytest.approx([1.0 + 0.25 * 1.0, 1.0])


def test_reinforce_learns_to_enable_and_is_reproducible():
    system, go, count = gated_counter()

    def train():
        torch.manual_seed(0)
        logits = nn.Linear(1, 2)
        one = torch.ones(1)

        def act(state):
            dist = torch.distributions.Categorical(logits=logits(one))
            a = dist.sample()
            return {go.enabled: bool(a.item())}, dist.log_prob(a)

        history = reinforce(
            system,
            logits,
            act,
            reward=lambda state, dt: state[count.count] * dt,
            horizon=5.0,
            iterations=12,
            batch=4,
            lr=0.1,
        )
        return history, torch.softmax(logits(one), 0)[1].item()

    history, p_enable = train()
    assert p_enable > 0.8
    assert sum(history[-3:]) > sum(history[:3])
    assert train() == (history, p_enable)
