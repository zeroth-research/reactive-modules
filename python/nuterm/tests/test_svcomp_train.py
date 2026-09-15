"""Rollout sampling (`benchmarks.svcomp._train.rollout`).

The rank is learned from the rounds the claim counts, so which rounds a
trajectory contributes is the trainer's one substantive decision. These pin it
for both shapes of claim: termination, where leaving the domain is final, and
recurrence, where it is not.
"""

import numpy as np

from benchmarks.svcomp._property import Liveness
from benchmarks.svcomp._termination import system_of, terminates
from benchmarks.svcomp._train import rollout
from tests._fixtures import loop_bench
from zrth.sugar import ite


def _rollout(bench, claim, max_len=50):
    return rollout(bench, system_of(bench), 1, max_len, 1.0,
                   np.random.default_rng(0), claim)


def test_a_run_that_starts_outside_the_domain_is_still_sampled():
    """A recurrence claim's run leaves the counting rounds and comes back, and
    its first round may be outside them: this counter starts at the value the
    property names. Stopping at the first round outside the domain -- which is
    right for termination, where that is a fixed point -- collects nothing at
    all here, and the trainer then has no rank to learn."""
    bench = loop_bench(("x",), lambda x: ite(x >= 9, 0, x + 1), init=lambda: (0,))
    S, Sp = _rollout(bench, Liveness(lambda W, S: S["x"] != 0))
    assert S.shape[0] > 0, "a wrap-around run was sampled as though it had stopped"
    # Only the counting rounds: the round from x == 0 is not one of them.
    assert 0 not in S[:, 0], S[:, 0]
    assert set(S[:, 0].astype(int)) == set(range(1, 10)), sorted(set(S[:, 0]))


def test_a_terminating_run_stops_at_its_fixed_point():
    """The other shape: under `terminates()` the domain is `some column moves`,
    so leaving it *is* a fixed point and every later round repeats. The
    trajectory ends there rather than running out its length."""
    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x), init=lambda: (4,))
    S, Sp = _rollout(bench, terminates(), max_len=500)
    assert S.shape[0] == 4, S[:, 0]           # 4 -> 3 -> 2 -> 1 -> 0, then stuck
    assert list(S[:, 0]) == [4, 3, 2, 1]


def test_the_pairs_are_consecutive_states():
    """`Sp` is the successor of `S`, round by round -- what the hinge loss is
    written against."""
    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x), init=lambda: (3,))
    S, Sp = _rollout(bench, terminates())
    assert list(S[:, 0]) == [3, 2, 1] and list(Sp[:, 0]) == [2, 1, 0]
