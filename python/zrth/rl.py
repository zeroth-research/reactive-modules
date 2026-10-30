"""Train a policy on runs of a module, through `Simulation`.

A policy here is whatever drives the module's inputs step by step. For training it is split
in two: `policy`, a `torch.nn.Module` holding the parameters, and `act(state)`, a function
that samples the inputs of the next step from the state the step starts in and returns them
with the log-probability of the choice. `reward(state, dt)` is what a step earns: it sees
the state the step started in and how long the step lasted.

    history = reinforce(system, policy, act, reward=reward, horizon=30.0, iterations=40)

`reinforce` is the plain policy-gradient algorithm: a batch of episodes under `act`, each
action credited with the discounted return-to-go from its step, advantages standardised over
the batch, one Adam step on `-sum(log_prob * advantage)`. It returns the mean episode
return of every iteration. The seeds of the episodes follow `seed`, so a run is reproducible.
"""

import math

import torch

from .simulate import Simulation


def rollout(system, act, horizon, reward, seed, t=None):
    """One run of `system` up to `horizon` under `act`: the log-probabilities of the
    actions taken, the rewards of the steps and their lengths, one entry per step."""
    sim = Simulation(system, t=t, seed=seed)
    logps, rewards, lengths = [], [], []
    while sim.now < horizon:
        state = sim.trace.states[-1]
        inputs, logp = act(state)
        before = sim.now
        sim.step(inputs, until=horizon)
        logps.append(logp)
        lengths.append(sim.now - before)
        rewards.append(reward(state, lengths[-1]))
    return logps, rewards, lengths


def returns_to_go(rewards, lengths, discount=0.0):
    """`G_t = r_t + exp(-discount * dt_t) * G_{t+1}`: what the action at every step is
    credited with, the rewards from its step on, discounted at `discount` per unit of time."""
    G, returns = 0.0, []
    for r, dt in zip(reversed(rewards), reversed(lengths)):
        G = r + math.exp(-discount * dt) * G
        returns.append(G)
    return list(reversed(returns))


def reinforce(system, policy, act, reward, horizon, iterations, batch=8, discount=0.0, lr=0.02, seed=0, t=None):
    """REINFORCE on `system`; see the module docstring. Returns the mean episode return of
    each iteration."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    for iteration in range(iterations):
        logps, advantages, returns = [], [], []
        for b in range(batch):
            lp, rewards, lengths = rollout(system, act, horizon=horizon, reward=reward, seed=seed + iteration * batch + b, t=t)
            logps += lp
            advantages += returns_to_go(rewards, lengths, discount)
            returns.append(sum(rewards))
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        loss = -(torch.stack(logps) * advantages).sum() / batch
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(sum(returns) / batch)
    return history
