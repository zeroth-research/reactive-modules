"""Rollout sampling (`benchmarks.svcomp._train.rollout`).

The rank is learned from the rounds the claim counts, so which rounds a
trajectory contributes is the trainer's one substantive decision. These pin it
for both shapes of claim: termination, where leaving the domain is final, and
recurrence, where it is not.
"""

import numpy as np

from benchmarks.svcomp._property import Liveness
from benchmarks.svcomp._termination import system_of, terminates
from benchmarks.svcomp._train import learn_ranking, rollout
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


def test_a_lexicographic_rank_is_fitted_component_by_component():
    """`ranks=2` asks for a rank of two components, and each is fitted on the
    rounds the one before it does not cover -- so the second specialises rather
    than re-learning the first. What comes back is certified by
    `lex_decrease`: on each counting round some component drops while every
    earlier one does not increase."""
    # while (x > 0) { if (y > 0) y-- ; else { x--; y = x } } -- the inner bound
    # is the outer variable, so the two are ordered rather than combinable.
    def update(c):
        x, y = c
        inner = y > 0
        return (ite(x > 0, ite(inner, x, x - 1), x),
                ite(x > 0, ite(inner, y - 1, x - 1), y))

    bench = loop_bench(("x", "y"), update, init=lambda: (4, 4))
    r = learn_ranking(bench, ranks=2, outer=8, n_epochs=500)
    assert r.verified, r.reason
    assert len(r.nets) == 2, "the second component was never fitted"
    assert len(r.witness.ranks) == 2, "the witness names one rank, not two"


def test_one_component_is_the_default_and_a_plain_drop():
    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x), init=lambda: (5,))
    r = learn_ranking(bench, outer=4, n_epochs=300)
    assert r.verified and len(r.nets) == 1 and len(r.witness.ranks) == 1


def test_a_component_with_nothing_left_over_is_not_fitted():
    """A rank that already covers every round makes the next component
    pointless, so `ranks` is a ceiling and not a count."""
    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x), init=lambda: (5,))
    r = learn_ranking(bench, ranks=3, outer=4, n_epochs=300)
    assert r.verified and len(r.nets) == 1, r.nets
