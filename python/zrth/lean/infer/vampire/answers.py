"""Vampire asked for an answer rather than a yes (`Answers`): the answer
literals of a refutation, read back as the holes' values (`Derived`).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from ...core.errors import Refused
from ...smt.vampire import run_vampire


# `% SZS answers Tuple [([1,100]|[0,100]|[0,100])|_] for cert`, and the
# single form `% SZS answers Tuple [[0,100]|_]`.
#
# The alternatives after `|` are a *disjunctive* answer: the refutation
# established that one of them is a witness, not that each is. A split
# refutation -- AVATAR's, which is what closes these questions -- reports one
# tuple per branch it closed, and a branch closed under a hypothesis that
# does not hold contributes a tuple that is not a witness at all. Measured on
# `m_countdown`: twelve alternatives, ten of them the `[0,100]` that is an
# invariant and the first two `[1,100]`, which is not preserved (`s0 = 1`
# steps to `0`). So every alternative is a candidate and they are tried in
# turn against the obligations; taking the first is how this route spent a
# while reporting no certificate for a module it had been handed one for.
_ANSWER = "SZS answers Tuple "


def answer_tuples(line: str) -> "list[list[str]] | None":
    """Every alternative in a `SZS answers` line, one string per hole.

    Scanned with a bracket depth rather than matched with a regex: a hole
    the refutation never pinned down prints as `∀X0.[X0]`, brackets and all,
    and a pattern that stops at the first `]` reads that as the end of the
    tuple and comes back one value short.
    """
    at = line.find(_ANSWER)
    if at < 0:
        return None
    rest = line[at + len(_ANSWER):].lstrip()
    if not rest.startswith("["):
        return None
    rest = rest[1:].lstrip()
    if rest.startswith("("):                         # the per-disjunct form
        rest = rest[1:].lstrip()
    out = []
    while rest.startswith("["):
        one, rest = _one_tuple(rest)
        if one is None:
            break
        out.append(one)
        rest = rest.lstrip()
        if not rest.startswith("|"):
            break
        rest = rest[1:].lstrip()
    return out or None


def _one_tuple(rest: str) -> "tuple[list[str] | None, str]":
    """One `[a,b,...]`, and what is left of the line after it."""
    depth, out, cur = 0, [], ""
    for i, ch in enumerate(rest[1:], start=2):
        if ch in "([":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "]":
            if depth == 0:
                out.append(cur)
                return out, rest[i:]
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(cur)
            cur = ""
            continue
        cur += ch
    return None, ""


@dataclass(frozen=True)
class Derived:
    """What Vampire answered: the alternatives, each one value per hole.

    Any one of them may be the witness, and which one is settled by putting
    it back into the template and re-asking the obligations.
    """

    answers: tuple[tuple[int, ...], ...]
    secs: float


class Answers:
    """Vampire in question-answering mode, on this run's deadline."""

    # Both ways of naming subformulas, because on a conjecture with holes
    # neither is the right one -- see the module docstring for what each
    # answers that the other does not. Named rather than a bare tuple of
    # flags so the log can say which one came back.
    STRATEGIES = (("naming", ()), ("no naming", ("--naming", "0")))
    # The short rung's wall clock. Measured over `tests/limits` with the
    # declared limit left alone: every answer but `m_max`'s and `m_relu`'s
    # arrives inside eight seconds, so one pass at this over every strategy
    # is what keeps the questions that do answer from waiting behind the
    # ones that do not.
    SHORT = 8.0

    def __init__(self, exe: str, *, seconds: float, log=print):
        self.exe = exe
        self.deadline = time.monotonic() + seconds
        # What Vampire is *told* it has, which is not what it is given.
        # `--time_limit` is a scheduling input before it is a cap: Vampire
        # slices it between strategies, so a small one runs a different
        # search rather than the same one cut short. Measured on `m_max`'s
        # question -- `--time_limit 20` reaches the limit, `--time_limit 25`
        # answers after 2.5 s. So every call declares the run's whole
        # budget, and what actually bounds a call is the wall clock below.
        self.declared = max(1, int(seconds))
        self.log = log
        self.calls = 0
        self.spent = 0.0

    def left(self) -> float:
        return self.deadline - time.monotonic()

    def ask(self, script: str, holes: int) -> "Derived | None":
        """The first answer any strategy gives, short rung first.

        The rungs are wall clock, and the declared limit does not move with
        them -- which is the whole reason a short rung is safe here. A short
        `--time_limit` would have Vampire run a *different* search; a short
        wall runs the same one and stops it early, and most answers measured
        over `tests/limits` arrive in under four seconds.
        """
        for short in (True, False):
            for n, (name, extra) in enumerate(self.STRATEGIES):
                left = self.left()
                # On the long rung what is left is shared out between the
                # strategies still to come, so the last is still asked.
                budget = (min(self.SHORT, left) if short
                          else left / (len(self.STRATEGIES) - n))
                answer = self._call(script, holes, budget, extra)
                if answer is not None:
                    self.log(f"[vampire] answered with {name}")
                    return answer
        return None

    def _call(self, script: str, holes: int, budget: float,
              extra: tuple) -> "Derived | None":
        if budget < 1:
            return None
        # Default mode, not `portfolio`: measured on this route's own
        # question for `m_countdown`, `--mode portfolio --schedule casc`
        # reaches the time limit where the default mode answers `[0,100]`
        # at once. A portfolio strategy is tuned to find a refutation, and
        # what is wanted here is the *substitution* a refutation carries.
        #
        # `--time_limit` tells Vampire what schedule to run; `timeout` is
        # the share of the deadline it actually gets.
        run = run_vampire(self.exe, script, [
            "--question_answering", "plain",
            *extra,
            "--time_limit", f"{self.declared}",
        ], timeout=budget, prefix="verith-vampire-qa-")
        self.calls += 1
        self.spent += run.secs
        if run.error is not None:
            raise Refused(
                f"--infer vampire wrote a question Vampire cannot read: "
                f"{run.error[:300]}"
            )
        return self._read(run.out, holes, run.secs)

    def _read(self, out: str, holes: int, dt: float) -> "Derived | None":
        alts = next(
            (t for t in (answer_tuples(ln) for ln in out.splitlines())
             if t is not None), None)
        if alts is None:
            return None
        seen, kept = set(), []
        for parts in alts:
            values = []
            for part in parts:
                v = _integer(part.strip())
                if v is None:
                    # A hole the refutation never had to pin down: any value
                    # serves, and zero is the one the certificate reads
                    # best. It is checked with the rest before anything is
                    # emitted.
                    v = 0
                values.append(v)
            if len(values) != holes:
                self.log(f"[vampire] an answer has {len(values)} values for "
                         f"{holes} holes; ignoring it")
                continue
            if tuple(values) not in seen:
                seen.add(tuple(values))
                kept.append(tuple(values))
        return Derived(tuple(kept), dt) if kept else None


def _integer(text: str) -> "int | None":
    text = text.strip()
    m = re.fullmatch(r"\$uminus\((\d+)\)", text)
    if m:
        return -int(m.group(1))
    return int(text) if re.fullmatch(r"-?\d+", text) else None
