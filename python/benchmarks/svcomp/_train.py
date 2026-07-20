"""learn a neural ranking function for a program module, verify it.

Trains a small ReLU net V (Linear -> ReLU -> sum) outside the reactive-module
framework (plain torch + Adam), then verifies the candidate with Z3 (V >= 0 and
V(s) - V(s') >= delta on the loop domain).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import z3
from torch import nn

# torch must load before the zrth C-extension (see _bench)
from ._bench import Bench  # noqa: F401  (ensures torch/zrth import order)
from ._equiv import _run_block
from ._invariants import infer_invariants
from ._verify_ranking import build_obligation, smt_oneshot


# ---------------------------------------------------------------------------
# The ranking net (ported from neural-termination TorchNRF, scalar case)
# ---------------------------------------------------------------------------

class TorchNRF(nn.Module):
    """V(x) = sum_j ReLU(W_j . x + b_j) — a positive sum of ReLUs, so V >= 0 by
    construction (frozen all-ones output layer, no output bias)."""

    def __init__(self, input_dim: int, hidden_dim: int = 7, quantize: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.quantize = quantize
        if quantize:
            with torch.no_grad():           # keep rounded init in {-1,0,1}
                self.fc1.weight.uniform_(-1.5, 1.5)
                self.fc1.bias.uniform_(-1.5, 1.5)

    def _q(self, t: torch.Tensor) -> torch.Tensor:
        return t if not self.quantize else t + (t.round() - t).detach()  # straight-through

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(F.linear(x, self._q(self.fc1.weight), self._q(self.fc1.bias)))
        return h.sum(dim=1)                  # all-ones output layer, no bias

    def to_layers(self, scale: float = 1.0):
        """Integer weights: [(W1, b1), (out_ones, zeros)] as numpy int arrays."""
        W1 = np.round(self.fc1.weight.detach().cpu().numpy().astype(np.float64) / scale)
        b1 = np.round(self.fc1.bias.detach().cpu().numpy().astype(np.float64) / scale)
        out = np.round(np.ones((1, W1.shape[0]), dtype=np.float64) / scale)
        return [(W1.astype(int), b1.astype(int)),
                (out.astype(int), np.zeros((1,), dtype=int))]


# ---------------------------------------------------------------------------
# Rollout: sample in-domain states from the module, take one program step
# ---------------------------------------------------------------------------

def _t(v: int) -> torch.Tensor:
    return torch.tensor([[int(v)]], dtype=torch.int64)


def _in_domain(bench: Bench, state: dict[str, int]) -> bool:
    d = z3.simplify(bench.domain({n: z3.IntVal(state[n]) for n in bench.state}))
    if z3.is_true(d):
        return True
    if z3.is_false(d):
        return False
    s = z3.Solver(); s.add(d)
    return s.check() == z3.sat


def _step(prog, ctrl, state: dict[str, int]) -> dict[str, int]:
    """One program step: latched `state` -> next state (via the update block)."""
    st = {ctrl[n][0]: _t(state[n]) for n in state}
    _run_block(prog.atoms, st, lambda a: a.update)
    return {n: int(st[ctrl[n][1]].reshape(-1)[0]) for n in state}


def _init_state(prog, ctrl, extl, bench: Bench, inputs: dict[str, int]) -> dict[str, int]:
    """Run the init block with the given extl (nondet) inputs -> initial state."""
    st = {extl[name][1]: _t(val) for name, val in inputs.items()}
    _run_block(prog.atoms, st, lambda a: a.init)
    return {n: int(st[ctrl[n][1]].reshape(-1)[0]) for n in bench.state}


# PAS: high-variance Gaussian with one anticorrelated pair (ported from nt).
_PAS_RHO = 0.25


def _pas_sample(dim: int, sigma: float, rng) -> np.ndarray:
    if dim == 0:
        return np.zeros(0, dtype=np.float64)
    if dim < 2:
        return np.round(rng.normal(0.0, sigma, size=dim)).astype(np.float64)
    cov = np.eye(dim) * (sigma * sigma)
    i, j = int(rng.integers(dim)), int(rng.integers(dim))
    if i != j:
        cov[i, j] = cov[j, i] = -_PAS_RHO * sigma * sigma
    return np.round(rng.multivariate_normal(np.zeros(dim), cov)).astype(np.float64)


def rollout(bench: Bench, n_traj: int, max_len: int, sigma: float,
            rng) -> tuple[np.ndarray, np.ndarray]:
    """nt-style trajectory rollouts: PAS-sample the inputs, init, execute the
    module up to `max_len`, collecting consecutive in-domain (s, T(s)) pairs."""
    prog, ctrl, extl = bench.build()
    n_in = len(bench.inputs)
    S, Sp = [], []
    n_trials = n_traj if n_in > 0 else 1          # deterministic program -> 1 trajectory
    for _ in range(n_trials):
        pas = _pas_sample(n_in, sigma, rng)
        inputs = {bench.inputs[k]: int(pas[k]) for k in range(n_in)}
        if bench.precondition is not None and not bench.precondition(inputs):
            continue
        s = _init_state(prog, ctrl, extl, bench, inputs)
        for _ in range(max_len):
            if not _in_domain(bench, s):
                break
            sp = _step(prog, ctrl, s)
            S.append([s[n] for n in bench.state])
            Sp.append([sp[n] for n in bench.state])
            s = sp
    return (np.array(S, dtype=np.float64).reshape(-1, len(bench.state)),
            np.array(Sp, dtype=np.float64).reshape(-1, len(bench.state)))


# ---------------------------------------------------------------------------
# Orchestrator: rollout -> train -> round -> verify (round-and-rebuild)
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    name: str
    verified: bool
    n_pairs: int
    final_loss: float
    layers: object = None
    reason: str | None = None


def learn_ranking(bench: Bench, delta: float = 1.0, hidden_dim: int = 7, seed: int = 0,
                  n_trajectories: int = 1000, max_trajectory_length: int = 1000,
                  initial_variance: float = 100.0, n_epochs: int = 1000,
                  lr: float = 0.05, outer: int = 20,
                  scales: tuple[float, ...] = (0.5, 1.0),
                  verifier=smt_oneshot, use_invariants: bool = True) -> TrainResult:
    """nt-matched: PAS trajectory rollouts, AdamW hinge loss, outer
    round-and-rebuild. Defaults mirror nt's learn_nrf_cfa.

    ``verifier`` is any ``Obligation -> VerifyResult`` (like nt's
    ``verifier_method``): the round-and-rebuild loop builds the obligation for
    each candidate (via ``build_obligation``, the composition seam) and accepts
    the first V it verifies. Default is ``smt_oneshot``; a Farkas or
    invariant-augmented verifier — or the Phase-2 composed-system verifier
    (program ⊕ V as one module) — swaps in without touching the trainer."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    sigma = float(np.sqrt(initial_variance))
    S, Sp = rollout(bench, n_trajectories, max_trajectory_length, sigma, rng)
    if S.shape[0] == 0:
        return TrainResult(bench.name, False, 0, float("nan"), reason="no in-domain samples")
    Xs = torch.from_numpy(S)
    Xsp = torch.from_numpy(Sp)
    dim = len(bench.state)

    # Houdini invariants (V-independent): inferred once, reused for every candidate.
    invariants = infer_invariants(bench) if use_invariants else []

    final_loss = float("inf")
    last_layers = None
    for _o in range(outer):
        model = TorchNRF(dim, hidden_dim).double()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        for _e in range(n_epochs):
            loss = torch.relu(model(Xsp) - model(Xs) + delta).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            final_loss = float(loss.detach())
            if final_loss <= 1e-6:
                break
        for scale in scales:
            layers = model.to_layers(scale)
            last_layers = layers
            if verifier(build_obligation(bench, layers, delta, invariants)).verified:
                return TrainResult(bench.name, True, S.shape[0], final_loss, layers)
    return TrainResult(bench.name, False, S.shape[0], final_loss, last_layers,
                       reason="trained but not verified")


if __name__ == "__main__":
    import sys
    from . import discover
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for b in discover():
        if only and only not in b.name:
            continue
        r = learn_ranking(b)
        tag = "VERIFIED" if r.verified else "unverified"
        extra = f" ({r.reason})" if r.reason else ""
        print(f"{tag:10s} {r.name}: {r.n_pairs} pairs, loss {r.final_loss:.4g}{extra}")
