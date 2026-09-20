"""Check that certificates are compared by what they mean, not how they read.

The comparison exists because the textual one is wrong on the first row it
meets, so that is what these cases pin: one invariant in four spellings is
one answer, a genuinely tighter invariant is reported as the stronger one
and in the right direction, two invariants that order neither way are
reported as incomparable, and a certificate this front end cannot read is
reported as unread rather than as a disagreement.

Three rows carry all of it, every one from the measured suites, and the
certificates the spellings come from are the ones the routes actually
printed -- taken from the recorded passes rather than invented, so a case
here is a case the tool meets.
"""
import sys
from pathlib import Path

import pytest

# The harness's own modules are scripts, imported by directory rather than
# as a package; pytest's `importlib` mode does not put a test file's
# directory on the path, so this does.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_certs import Judge, certificates, compare_row, context_for  # noqa: E402
from suites import all_rows                                             # noqa: E402

ROWS = {r.key: r for r in all_rows()}

# `0 <= s0 <= 100`, as the seven routes of the recorded pass each wrote it.
# Four distinct spellings, one formula.
SPELLINGS = {
    "houdini": {"inv": "(and (>= s0 0) (<= s0 100))"},
    "vampire": {"inv": "(and (<= 0 s0) (<= s0 100))"},
    "smt-linear": {"inv": "(and (>= s0 0) (>= (+ 100 (- s0)) 0))"},
    "sygus": {"inv": "(and (<= (+ (- 100) (* (- 101) s0)) 0) "
                     "(<= (+ (- 100) (* 1 s0)) 0))"},
}


def test_one_invariant_in_four_spellings_is_one_answer():
    """The case the textual comparison gets wrong, and the reason for cvc5.

    Every route here derived the same interval. A string diff reports four
    disagreements; there are none."""
    seen = compare_row(ROWS["fbk/m_countdown/InvBase"], dict(SPELLINGS))

    assert seen.verdict == "same"
    assert len(seen.groups) == 1
    assert sorted(seen.groups[0].routes) == sorted(SPELLINGS)


def test_a_tighter_invariant_is_reported_as_the_stronger_one():
    """`0 <= s0 <= 100` implies `s0 <= 100` and is not implied by it, so it
    is the stronger claim and the pair is ordered rather than merely
    different."""
    seen = compare_row(ROWS["fbk/m_countdown/InvBase"],
                       {"tight": SPELLINGS["houdini"],
                        "loose": {"inv": "(<= s0 100)"}})

    assert seen.verdict == "stronger"
    assert len(seen.groups) == 2
    at = {g.routes[0]: i for i, g in enumerate(seen.groups)}
    # One direction only, and it is the tight one that implies the loose.
    assert seen.stronger == [(at["tight"], at["loose"])]
    assert not seen.undecided


def test_two_invariants_neither_of_which_implies_the_other():
    """The third answer. `s0 <= 50` and `50 <= s0` each hold of states the
    other does not, so neither route looked harder -- they looked
    elsewhere."""
    seen = compare_row(ROWS["fbk/m_countdown/InvBase"],
                       {"low": {"inv": "(<= s0 50)"},
                        "high": {"inv": "(<= 50 s0)"}})

    assert seen.verdict == "incomparable"
    assert len(seen.groups) == 2
    assert not seen.stronger and not seen.undecided


def test_some_pairs_ordered_and_some_not_is_its_own_answer():
    """Three answers where one pair is ordered and the rest are not.

    `0 <= s0 <= 3` implies `s0 <= 3`, and `s0 >= 50` orders against neither.
    Reporting this as `stronger` because *a* pair is ordered would hide the
    pair that is not -- which is the reason the verdict counts pairs rather
    than asking whether any is ordered, and 19 of the recorded rows have
    three groups or more for it to matter on."""
    seen = compare_row(ROWS["fbk/m_countdown/InvBase"],
                       {"tight": {"inv": "(and (<= 0 s0) (<= s0 3))"},
                        "loose": {"inv": "(<= s0 3)"},
                        "elsewhere": {"inv": "(>= s0 50)"}})

    assert seen.verdict == "mixed"
    assert len(seen.groups) == 3
    at = {g.routes[0]: i for i, g in enumerate(seen.groups)}
    assert seen.stronger == [(at["tight"], at["loose"])]


def test_a_chain_of_three_is_stronger_rather_than_mixed():
    """Every pair ordered is the other side of the same count: these three
    nest, so the row is a chain and nothing is hidden by saying so."""
    seen = compare_row(ROWS["fbk/m_countdown/InvBase"],
                       {"a": {"inv": "(and (<= 0 s0) (<= s0 3))"},
                        "b": {"inv": "(and (<= 0 s0) (<= s0 30))"},
                        "c": {"inv": "(<= 0 s0)"}})

    assert seen.verdict == "stronger"
    assert len(seen.groups) == 3
    # Three unordered pairs, every one of them ordered by implication.
    assert len({frozenset(p) for p in seen.stronger}) == 3


def test_a_certificate_this_front_end_cannot_read_is_unread():
    """`--infer ai` writes Lean, and a row it answered is not thereby a row
    the routes disagreed on. It is reported apart from the groups, so it
    cannot be counted as either agreement or difference."""
    seen = compare_row(ROWS["fbk/m_countdown/InvBase"],
                       {"houdini": SPELLINGS["houdini"],
                        "ai": {"inv": "fun s => 0 ≤ s 0 0 ∧ s 0 0 ≤ 100"}})

    assert seen.verdict == "same"           # of the ones that could be read
    assert [g.routes for g in seen.groups] == [["houdini"]]
    assert list(seen.unread) == ["ai"]


def test_the_environment_a_row_asks_for_is_put_back():
    """`petri_mod.py` is twelve modules chosen by `$PETRI_NET`, so loading
    one must not leave the variable set for whatever is compared next."""
    import os

    row = ROWS["petri/reset/slots-le"]
    assert row.env == {"PETRI_NET": "reset"}
    before = os.environ.get("PETRI_NET")

    context_for(row)

    assert os.environ.get("PETRI_NET") == before


def test_a_ranking_function_is_compared_for_equality_only():
    """Two ranking functions that differ are not thereby ordered -- `s0` and
    `2*s0` are both valid and neither is the stronger claim -- so they are
    grouped and no implication is asked of them."""
    seen = compare_row(ROWS["limits/m_countdown/Countdown"],
                       {"a": {"inv": "(<= 0 s0)", "ranking": "s0"},
                        "b": {"inv": "(<= 0 s0)", "ranking": "(* 2 s0)"},
                        "c": {"inv": "(<= 0 s0)", "ranking": "(+ s0 0)"}})

    assert seen.verdict == "same"                    # the invariants agree
    # `s0` and `s0 + 0` are one ranking function; `2*s0` is another.
    assert [sorted(g.routes) for g in seen.rank_groups] == [["a", "c"], ["b"]]


def test_certificates_are_keyed_by_column_not_by_route():
    """`houdini` and `houdini-vampire` are one `--infer` route measured over
    two solvers, so the `route` field beside a cell says `houdini` for both.
    The key suffix is what tells them apart, and is what this reads."""
    runs = {
        "r/one::houdini": {"gen": {"inferred": {"inv": "(<= 0 s0)"}},
                           "route": "houdini"},
        "r/one::houdini-vampire": {"gen": {"inferred": {"inv": "(<= 1 s0)"}},
                                   "route": "houdini"},
        "r/one::sygus": {"gen": {"inferred": {}}, "route": "sygus"},
    }

    assert certificates(runs) == {
        "r/one": {"houdini": {"inv": "(<= 0 s0)"},
                  "houdini-vampire": {"inv": "(<= 1 s0)"}}}
    # And a route filter selects by that same suffix.
    assert list(certificates(runs, keep=("houdini",))["r/one"]) == ["houdini"]


@pytest.mark.parametrize("a, b, implies, same", [
    ("(<= 0 s0)", "(<= 0 s0)", True, True),
    ("(and (<= 0 s0) (<= s0 5))", "(<= 0 s0)", True, False),
    ("(<= 0 s0)", "(and (<= 0 s0) (<= s0 5))", False, False),
    ("(<= s0 0)", "(<= 1 s0)", False, False),
])
def test_the_judge_decides_implication_and_equivalence(a, b, implies, same):
    """The two queries under everything above, on predicates small enough to
    check by eye."""
    judge = Judge(context_for(ROWS["fbk/m_countdown/InvBase"]))

    assert judge.implies(judge.parse(a), judge.parse(b)) is implies
    assert judge.same(judge.parse(a), judge.parse(b)) is same
