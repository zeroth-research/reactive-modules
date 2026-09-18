"""TA2Magic by synthesis: the certificate Vampire's answer literals derive.

``--infer vampire``.  Nothing here proposes a candidate.  The route states
the certificate's obligations with the certificate *itself* left open --
a template whose coefficients are existentially quantified -- and Vampire
answers with the coefficients, or does not answer.  Where ``--infer
houdini`` asks "is this fact inductive?" once per fact, this asks "what
numbers make the whole thing inductive?" once.

That is Vampire's *answer literal* mechanism (``--question_answering
plain``).  Handed a conjecture ``?[A,B]: phi(A,B)`` it refutes the negation
and reports the substitutions the refutation used::

    % SZS answers Tuple [([1,100]|[0,100]|[0,100])|_] for cert

which carries `0 <= s0 <= 100` for `m_countdown`, derived rather than
checked.  The alternatives are a *disjunctive* answer -- one of them is a
witness, not each of them, because a split refutation closes each branch
under its own hypothesis -- so all of them are tried below, and the one the
obligations accept is the certificate.

What the obligations have to look like
======================================
Three restrictions, and the first is the one that decides the shape of this
whole module.

**No ``$ite``, anywhere.**  Vampire's answer-literal search is superposition
plus theory reasoning, and a conditional in the goal defeats it -- measured:
`m_countdown`'s own step obligation, stated with the transition as one
``$ite``, does not come back inside 40 s, and the same obligation split into
its two guarded branches is answered in under one.  So the transition is
*branch-split* before it is printed: every ``ite`` condition the update and
the init terms mention becomes a case, cvc5's rewriter folds the
conditionals away under each assignment of them, and each case is one more
implication with its guard as a hypothesis.  A term that still holds an
``ite`` after that -- a condition over something the split did not reach --
is refused by name rather than printed into a question Vampire cannot
answer.

**No uninterpreted function for the round.**  Same measurement: defining
``nxt(S)`` by an axiom and asking about ``nxt`` is a time limit where
inlining the same arithmetic is an answer.  So the successor state is
substituted into the obligation, once per branch.

**Integer state.**  TPTP has ``$real`` and Vampire reads it, but the
templates below are integer intervals and integer coefficients, and a Real
component would need the value-set shape that `--infer houdini` carries.
Refused by name.

The templates, smallest first
=============================
A template is a formula with holes, and the holes are what Vampire is asked
for.  They are tried in order, because every extra hole is a wider search
and the narrow one is usually enough:

* **intervals** -- ``A_i <= s_i <= B_i`` per component.  Two holes each.
* **intervals and differences** -- the above, plus ``C_ij <= s_i - s_j <=
  D_ij`` for each pair.  What a module whose components move together needs,
  and quadratic in the width, which is why it is second.

Under ``--buchi`` the ranking function is more holes in the same question
rather than a second search: ``R_0*s_0 + ... + R_n-1*s_n-1 + R_c``, asked
for alongside the invariant, so what comes back is an invariant that admits
a rank and a rank that drops on it.  The obligation is stated as ``rank(s) >
0`` where the property fails and ``rank(s') < rank(s)``, which is ite-free
and implies the clamped ``Int.toNat`` form `rule_buchi` asks for: where
``r > 0`` and ``r' < r``, ``max(r',0) < max(r,0)`` whatever the sign of
``r'``.

What comes back is checked
==========================
An answer literal is a substitution *a* refutation used, and a refutation of
a mis-stated question proves nothing about the module -- nor does one branch
of a disjunctive answer.  So the coefficients Vampire returns are put back
into the template and the four obligations are re-asked of cvc5 -- through
the same
:mod:`zrth.lean.houdini_solver` seam `--infer houdini` uses, in
milliseconds.  A certificate that does not survive that is reported as not
found, with what Vampire said.  This is not Houdini: nothing is filtered and
nothing is proposed, it is the one check that the derivation is honest.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

from ..cert import CertificateData
from ..common import Refused
from ..houdini_solver import (
    DEFAULT_TIMEOUT,
    Cvc5Solver,
    _kill_group,
)
from . import TA2Magic
from .houdini import Candidate, Obligations
from ..smt_synth import SynthContext, smt_int

try:
    import cvc5                                      # type: ignore
    from cvc5 import Kind
except ImportError:                                  # pragma: no cover
    cvc5 = None
    Kind = None

# The templates, in the order they are tried.
TEMPLATES = ("intervals", "differences")
DEFAULT_TEMPLATE = TEMPLATES[0]

# A branch is one assignment of the transition's `ite` conditions, so the
# case count doubles with each. Past this the question is too long to be
# worth asking, and the route says so rather than printing it.
_MAX_CONDITIONS = 5
_MAX_BRANCHES = 2 ** _MAX_CONDITIONS
# Differences are quadratic in the width; past this the second template is
# more holes than any answer-literal search has been measured to close.
_MAX_WIDTH_FOR_DIFFERENCES = 4


# ══════════════════════════════════════════════════════════════════════════
# Branches: the transition without its conditionals
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Branch:
    """One case of a round: its guard, and the state terms under that guard.

    `guard` is the conjunction of condition literals that selects this case
    and `state` is the terms with every `ite` folded away by cvc5's rewriter
    under that assignment. The guard is a hypothesis of the obligation
    rather than a condition inside it, which is the whole point: Vampire
    answers an implication and does not answer a conditional.
    """

    guard: tuple                # cvc5 Bool terms, all of them true here
    state: tuple                # the folded terms, ite-free


def ite_conditions(terms) -> list:
    """Every condition an `ite` in `terms` branches on, outermost first.

    Deduplicated by printed form: a transition that tests `s0 = 0` in three
    components is one case split, not three.
    """
    seen: dict = {}

    def walk(t):
        if t.getKind() == Kind.ITE:
            seen.setdefault(str(t[0]), t[0])
        for child in t:
            walk(child)

    for t in terms:
        walk(t)
    return list(seen.values())


def split(ctx: SynthContext, terms, what: str) -> list[Branch]:
    """`terms`, split into the ite-free cases of their own conditions.

    The round and the initial state are split *apart*, because they are not
    quantified over the same things: the initial state is a function of the
    inputs alone, so a guard taken from the round -- which tests the latched
    state -- would leave a variable in the entry obligation that nothing
    binds. Every assignment is emitted, the unsatisfiable ones included: an
    unsatisfiable guard makes its implication vacuously true, which costs
    Vampire a clause and costs this module the logic that would have worked
    out which ones those are.
    """
    tm = ctx.tm
    rewriter = cvc5.Solver(tm)
    terms = list(terms)
    conds = ite_conditions(terms)
    if len(conds) > _MAX_CONDITIONS:
        raise Refused(
            f"--infer vampire splits {what} on the {len(conds)} conditions "
            f"its `ite`s test, which is {2 ** len(conds)} cases and past the "
            f"{_MAX_BRANCHES} this route will print. `--infer houdini` "
            f"states the round once, conditionals and all."
        )
    out = []
    for bits in product((True, False), repeat=len(conds)):
        subs = [tm.mkBoolean(b) for b in bits]
        folded = tuple(rewriter.simplify(t.substitute(conds, subs))
                       if conds else t for t in terms)
        guard = tuple(c if b else tm.mkTerm(Kind.NOT, c)
                      for c, b in zip(conds, bits))
        out.append(Branch(guard, folded))
    return out


# ══════════════════════════════════════════════════════════════════════════
# TPTP
# ══════════════════════════════════════════════════════════════════════════

_BINARY = {
    Kind.ADD: "$sum", Kind.SUB: "$difference", Kind.MULT: "$product",
    Kind.INTS_DIVISION: "$quotient_e", Kind.INTS_MODULUS: "$remainder_e",
}
_RELATION = {
    Kind.LT: "$less", Kind.LEQ: "$lesseq",
    Kind.GT: "$greater", Kind.GEQ: "$greatereq",
}
_CONNECTIVE = {Kind.AND: " & ", Kind.OR: " | ", Kind.IMPLIES: " => ",
               Kind.XOR: " <~> "}


class Tptp:
    """A printer from cvc5 terms to TPTP, over one naming of the constants.

    `names` maps a constant's printed form to the TPTP variable standing for
    it -- the state `S0..`, the inputs `E0..`, and the template's holes,
    which are variables of the conjecture's existential rather than symbols
    of the problem. Anything not in it is a constant this route did not put
    there, and is refused rather than printed as a fresh symbol Vampire
    would happily quantify over.
    """

    def __init__(self, names: dict):
        self.names = names

    def __call__(self, t) -> str:
        k = t.getKind()
        kids = list(t)
        if k == Kind.CONSTANT:
            name = self.names.get(str(t))
            if name is None:
                raise Refused(
                    f"--infer vampire met the symbol `{t}` in an obligation "
                    f"and has no TPTP variable for it"
                )
            return name
        if k == Kind.CONST_INTEGER:
            return _numeral(t.getIntegerValue())
        if k == Kind.CONST_BOOLEAN:
            return "$true" if t.getBooleanValue() else "$false"
        if k == Kind.ITE:
            raise Refused(
                "--infer vampire cannot state an obligation that still holds "
                "an `ite` after the transition was branch-split: Vampire's "
                "answer-literal search does not see through a conditional. "
                "`--infer houdini` proves the same obligation with one."
            )
        if k == Kind.NEG:
            return f"$uminus({self(kids[0])})"
        if k == Kind.NOT:
            return f"~({self(kids[0])})"
        if k in _BINARY:
            return self._fold(_BINARY[k], kids)
        if k in _RELATION:
            return self._chain(_RELATION[k], kids)
        if k in _CONNECTIVE:
            return "(" + _CONNECTIVE[k].join(self(c) for c in kids) + ")"
        if k == Kind.EQUAL:
            op = " <=> " if kids[0].getSort().isBoolean() else " = "
            return self._chain_infix(op, kids)
        if k == Kind.DISTINCT:
            return self._chain_infix(" != ", kids)
        raise Refused(
            f"--infer vampire has no TPTP for `{t.getKind()}` (in `{t}`). "
            f"`--infer houdini` states obligations in SMT-LIB, which cvc5 "
            f"and Vampire both read whole."
        )

    def _fold(self, op: str, kids) -> str:
        """`$sum` and friends are binary in TPTP; cvc5's are n-ary."""
        out = self(kids[0])
        for c in kids[1:]:
            out = f"{op}({out},{self(c)})"
        return out

    def _chain(self, op: str, kids) -> str:
        parts = [f"{op}({self(a)},{self(b)})"
                 for a, b in zip(kids, kids[1:])]
        return parts[0] if len(parts) == 1 else "(" + " & ".join(parts) + ")"

    def _chain_infix(self, op: str, kids) -> str:
        parts = [f"({self(a)}{op}{self(b)})" for a, b in zip(kids, kids[1:])]
        return parts[0] if len(parts) == 1 else "(" + " & ".join(parts) + ")"


def _numeral(v) -> str:
    """TPTP writes a negative integer as `$uminus` of a positive one."""
    return f"$uminus({-int(v)})" if v < 0 else str(int(v))


def _all(parts) -> str:
    kept = [p for p in parts if p != "$true"]
    if not kept:
        return "$true"
    return kept[0] if len(kept) == 1 else "(" + " & ".join(kept) + ")"


# ══════════════════════════════════════════════════════════════════════════
# The template
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Template:
    """The certificate with its coefficients left open.

    `holes` are the TPTP variables of the conjecture's existential, in the
    order Vampire reports them, which is how an answer tuple is read back.
    `rows` say how to rebuild the invariant from them: each is a term
    printer over the answer, in SMT-LIB, because SMT-LIB is what the
    certificate carries.
    """

    name: str
    holes: tuple[str, ...]
    # `(hole_lo, hole_hi, smt_body)` -- `lo <= body <= hi`.
    rows: tuple[tuple[str, str, str], ...]
    rank: tuple[str, ...] = ()      # rank coefficient holes, then the constant

    @property
    def all_holes(self) -> tuple[str, ...]:
        return self.holes + self.rank


def template(ctx: SynthContext, which: str, *, ranked: bool) -> Template:
    """The `which` template over this module's components."""
    width = len(ctx.state)
    rows, holes = [], []
    for i in range(width):
        lo, hi = f"A{i}", f"B{i}"
        rows.append((lo, hi, f"s{i}"))
        holes += [lo, hi]
    if which == "differences":
        for i, j in combinations(range(width), 2):
            lo, hi = f"C{i}_{j}", f"D{i}_{j}"
            rows.append((lo, hi, f"(- s{i} s{j})"))
            holes += [lo, hi]
    rank = tuple([f"R{i}" for i in range(width)] + ["Rc"]) if ranked else ()
    return Template(which, tuple(holes), tuple(rows), rank)


def templates(ctx: SynthContext, *, ranked: bool) -> list[Template]:
    """The templates this module is worth asking about, smallest first."""
    out = [template(ctx, "intervals", ranked=ranked)]
    if len(ctx.state) > 1 and len(ctx.state) <= _MAX_WIDTH_FOR_DIFFERENCES:
        out.append(template(ctx, "differences", ranked=ranked))
    return out


# ══════════════════════════════════════════════════════════════════════════
# The question
# ══════════════════════════════════════════════════════════════════════════


class Question:
    """One module's obligations as a TPTP conjecture with holes in it."""

    def __init__(self, ctx: SynthContext, ob: Obligations,
                 entry: list[Branch], step: list[Branch]):
        self.ctx, self.ob = ctx, ob
        self.entry, self.step = entry, step
        self.state = [f"S{i}" for i in range(len(ctx.state))]
        self.inputs = [f"E{i}" for i in range(len(ob.el) + len(ob.en))]
        names = {str(c): n for c, n in zip(ob.s, self.state)}
        names.update({str(c): n for c, n in
                      zip(list(ob.el) + list(ob.en), self.inputs)})
        self.names = names
        self.p = Tptp(names)

    # --- the template, as TPTP over a state ------------------------------

    def _rows(self, tpl: Template, at: list[str]) -> str:
        """`tpl` read at `at`, which is a TPTP term per component."""
        out = []
        for lo, hi, body in tpl.rows:
            term = self._body(body, at)
            out += [f"$lesseq({lo},{term})", f"$lesseq({term},{hi})"]
        return _all(out)

    def _rank(self, tpl: Template, at: list[str]) -> str:
        parts = [f"$product({c},{v})" for c, v in zip(tpl.rank, at)]
        out = tpl.rank[-1]
        for p in parts:
            out = f"$sum({out},{p})"
        return out

    @staticmethod
    def _body(body: str, at: list[str]) -> str:
        """A template row's SMT body, as TPTP over `at`.

        The bodies are this module's own two shapes, so they are built here
        rather than parsed: a component, or a difference of two.
        """
        if body.startswith("(- "):
            i, j = (int(x[1:]) for x in body[3:-1].split())
            return f"$difference({at[i]},{at[j]})"
        return at[int(body[1:])]

    # --- the obligations --------------------------------------------------

    def conjecture(self, tpl: Template, prp, *, safety: bool) -> str:
        p, ob = self.p, self.ob
        obligations = []
        # init_inv. The initial state is a function of the inputs alone, so
        # this is quantified over the inputs and split on its own
        # conditions -- a guard from the round would leave a latched-state
        # variable here that nothing binds.
        for br in self.entry:
            hyp = _all([p(ob.init_pre)] + [p(g) for g in br.guard])
            obligations.append(self._forall(
                self.inputs,
                f"({hyp} => {self._rows(tpl, [p(t) for t in br.state])})"))
        for br in self.step:
            guard = _all([p(g) for g in br.guard])
            here = self._rows(tpl, self.state)
            nxt = [p(t) for t in br.state]
            hyp = _all([here, p(ob.update_pre), guard])
            obligations.append(self._forall(
                self.state + self.inputs,
                f"({hyp} => {self._rows(tpl, nxt)})"))
            if not safety:
                # hrank, ite-free: positive where the property fails, and
                # smaller after the round. Implies the `Int.toNat` form.
                drops = _all([
                    f"$greater({self._rank(tpl, self.state)},0)",
                    f"$less({self._rank(tpl, nxt)},"
                    f"{self._rank(tpl, self.state)})",
                ])
                fails = f"~({p(self._at(prp, ob.s))})"
                obligations.append(self._forall(
                    self.state + self.inputs,
                    f"({_all([here, fails, p(ob.update_pre), guard])} "
                    f"=> {drops})"))
        if safety:
            obligations.append(self._forall(
                self.state,
                f"({self._rows(tpl, self.state)} "
                f"=> {p(self._at(prp, ob.s))})"))
        holes = ",".join(f"{h}:$int" for h in tpl.all_holes)
        body = _all(obligations)
        return f"tff(cert, conjecture, ?[{holes}]: {body}).\n"

    def _at(self, term, vs):
        return term.substitute(self.ctx.state, vs)

    @staticmethod
    def _forall(vs: list[str], body: str) -> str:
        if not vs:
            return body
        return "![" + ",".join(f"{v}:$int" for v in vs) + "]: " + body


# ══════════════════════════════════════════════════════════════════════════
# Asking Vampire for an answer rather than a yes
# ══════════════════════════════════════════════════════════════════════════

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

    def __init__(self, exe: str, *, seconds: float, log=print):
        self.exe = exe
        self.deadline = time.monotonic() + seconds
        self.log = log
        self.calls = 0
        self.spent = 0.0

    def left(self) -> float:
        return self.deadline - time.monotonic()

    def ask(self, script: str, holes: int, limit: float) -> "Derived | None":
        budget = min(limit, self.left())
        if budget < 1:
            return None
        with tempfile.NamedTemporaryFile(
            "w", suffix=".p", prefix="verith-vampire-qa-", delete=False
        ) as f:
            f.write(script)
            path = f.name
        # Default mode, not `portfolio`: measured on this route's own
        # question for `m_countdown`, `--mode portfolio --schedule casc`
        # reaches the time limit where the default mode answers `[0,100]`
        # at once. A portfolio strategy is tuned to find a refutation, and
        # what is wanted here is the *substitution* a refutation carries.
        cmd = [
            self.exe, "--input_syntax", "tptp",
            "--question_answering", "plain",
            "--time_limit", f"{max(1, int(budget))}",
            path,
        ]
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                start_new_session=True)
        try:
            out, _ = proc.communicate(timeout=budget + 10)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            out = ""
        finally:
            _kill_group(proc)
            Path(path).unlink(missing_ok=True)
        dt = time.perf_counter() - t0
        self.calls += 1
        self.spent += dt
        error = next((ln for ln in out.splitlines()
                      if "User error" in ln or "Parsing Error" in ln), None)
        if error is not None:
            raise Refused(
                f"--infer vampire wrote a question Vampire cannot read: "
                f"{error.strip()[:300]}"
            )
        return self._read(out, holes, dt)

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


# ══════════════════════════════════════════════════════════════════════════
# The route
# ══════════════════════════════════════════════════════════════════════════


class TA2MagicVampire(TA2Magic):
    """The certificate Vampire derives, checked before it is handed on."""

    def __init__(self, module, *, vampire: str, timeout: float = DEFAULT_TIMEOUT,
                 artifacts=None, log=print):
        super().__init__("")
        if cvc5 is None:                             # pragma: no cover
            raise Refused(
                "--infer vampire needs the `cvc5` package to encode the "
                "module, which is not importable. Install with: "
                "uv pip install cvc5"
            )
        self.module = module
        self.exe = vampire
        self.timeout = float(timeout)
        self.artifacts = artifacts
        self.log = log

    # --- driver -----------------------------------------------------------

    def infer(self, cd: CertificateData) -> CertificateData:
        # `SynthContext.build` refuses a component this route cannot weigh
        # as an integer -- Real, bitvector, matrix-shaped -- before anything
        # here is printed, which is the same gate `--infer smt-linear` uses.
        ctx = SynthContext.build(self.module, cd, route="vampire")
        self._check_sorts(ctx)
        self.ctx = ctx
        self.ob = ob = Obligations(ctx)
        entry = split(ctx, ob.init, "the initial state")
        step = split(ctx, ob.next, "the round")
        q = Question(ctx, ob, entry, step)
        self.log(f"[vampire] {len(step)} branch(es) of the round, "
                 f"{len(entry)} of the initial state, "
                 f"{len(ctx.state)} component(s)")

        asker = Answers(self.exe, seconds=self.timeout, log=self.log)
        tried = []
        for tpl in templates(ctx, ranked=not cd.is_safety):
            if asker.left() < 1:
                break
            script = q.conjecture(tpl, ctx.prp, safety=cd.is_safety)
            self.log(f"[vampire] template {tpl.name}: "
                     f"{len(tpl.all_holes)} holes, {len(script)} chars")
            tried.append(tpl.name)
            answer = asker.ask(script, len(tpl.all_holes), asker.left())
            if answer is None:
                self.log(f"[vampire] no answer for {tpl.name} "
                         f"({asker.spent:.1f} s spent)")
                continue
            # A disjunctive answer says one of these is a witness, so each is
            # a candidate until the obligations say otherwise. Checking one
            # is cvc5 on four small queries, but there can be a dozen of
            # them, so the run's own deadline bounds the loop -- after the
            # first, which is always worth the check.
            for n, values in enumerate(answer.answers):
                if n and asker.left() < 0:
                    self.log(f"[vampire] out of time with "
                             f"{len(answer.answers) - n} alternative(s) "
                             f"of the answer unchecked")
                    break
                named = dict(zip(tpl.all_holes, values))
                self.log("[vampire] Vampire answered "
                         + ", ".join(f"{h}={v}" for h, v in named.items()))
                inv, rank = self._read_back(tpl, named, cd)
                if self._checks_out(inv, rank, cd):
                    return self._emit(cd, inv, rank, tpl)
                self.log("[vampire] that answer does not satisfy the "
                         "obligations when they are restated")
        self.log(f"[vampire] {asker.calls} Vampire call(s), "
                 f"{asker.spent:.1f} s")
        return self._none(cd, tried, asker)

    # --- what the answer means --------------------------------------------

    def _read_back(self, tpl: Template, named: dict,
                   cd: CertificateData) -> tuple:
        """The template with Vampire's numbers in it, as SMT-LIB."""
        facts = []
        for lo, hi, body in tpl.rows:
            facts.append(f"(<= {smt_int(named[lo])} {body})")
            facts.append(f"(<= {body} {smt_int(named[hi])})")
        inv = ("true" if not facts else facts[0] if len(facts) == 1
               else "(and " + " ".join(facts) + ")")
        if cd.is_safety:
            return inv, None
        terms = [f"(* {smt_int(named[c])} s{i})"
                 for i, c in enumerate(tpl.rank[:-1]) if named[c]]
        const = named[tpl.rank[-1]]
        parts = ([smt_int(const)] if const or not terms else []) + terms
        rank = parts[0] if len(parts) == 1 else "(+ " + " ".join(parts) + ")"
        return inv, rank

    def _checks_out(self, inv_src: str, rank_src: "str | None",
                    cd: CertificateData) -> bool:
        """The four obligations, restated and put to cvc5.

        An answer literal is the substitution *a* refutation used, and this
        route wrote the question it refuted -- so the certificate is checked
        against the module's own encoding before anything is emitted, the
        same way `--pre-check cvc5` would check one supplied by hand.
        """
        ob, ctx = self.ob, self.ctx
        try:
            facts = [Candidate(inv_src, ctx.env.parse_expr(inv_src))]
            rank = (None if rank_src is None
                    else Candidate(rank_src, ctx.env.parse_expr(rank_src)))
        except Exception as e:                       # noqa: BLE001
            self.log(f"[vampire] the answer does not parse back: {e}")
            return False
        solver = Cvc5Solver(ctx.tm, seconds=max(5.0, min(30.0, self.timeout)),
                            log=self.log)
        queries = [ob.holds_at_entry(facts), ob.preserved(facts, facts)]
        if cd.is_safety:
            queries.append(ob.implies(facts, ctx.prp))
        else:
            queries.append(ob.drops(facts, rank))
        for name, query in zip(("init_inv", "step_inv", "the last"), queries):
            answer = solver.prove(query, 10)
            if not answer.proved:
                self.log(f"[vampire] {name} does not hold of the answer "
                         f"({answer.verdict.value})")
                return False
        return True

    # --- reading the module -----------------------------------------------

    def _check_sorts(self, ctx: SynthContext) -> None:
        """Every component an `$int` variable, not merely readable as one.

        `SynthContext` lets a bitvector through -- it has an integer
        *reading*, its unsigned value, which is what `--infer smt-linear`
        weighs it by -- but the terms that reach the printer are still
        bitvector operations, and TPTP has no theory for them. Met here, by
        sort, rather than as `no TPTP for Kind.CONST_BITVECTOR` four steps
        later.
        """
        bad = [f"s{i} is {s}" for i, s in enumerate(ctx.env.state_sorts)
               if not s.isInteger()]
        bad += [f"{c} is {c.getSort()}"
                for c in list(ctx.extl_next) + list(ctx.extl_latched)
                if not c.getSort().isInteger()]
        if bad:
            raise Refused(
                f"--infer vampire states its obligations in TPTP arithmetic, "
                f"which has integers and no bitvectors, and this module has "
                f"{', '.join(bad)}. `--infer houdini` states them in SMT-LIB "
                f"-- still not bitvectors, but it says so of the candidate "
                f"shapes rather than of the printer; `--infer smt-linear` "
                f"weighs a bitvector as its unsigned value."
            )

    # --- out ---------------------------------------------------------------

    def _emit(self, cd: CertificateData, inv_src: str,
              rank_src: "str | None", tpl: Template) -> CertificateData:
        self.log(f"[vampire] inv: {inv_src}")
        cd.inv = cd.inv_smt = inv_src
        if rank_src is not None:
            self.log(f"[vampire] ranking: {rank_src}")
            cd.ranking = cd.ranking_smt = rank_src
        self._record(inv_src, rank_src, tpl)
        return cd

    def _record(self, inv_src: str, rank_src: "str | None",
                tpl: Template) -> None:
        if self.artifacts is None:
            return
        what = (f"Derived by Vampire's answer literals from a `{tpl.name}` "
                f"template with {len(tpl.all_holes)} holes -- the "
                f"obligations were stated with the certificate open and "
                f"Vampire returned the coefficients -- then checked against "
                f"the module's encoding with cvc5.")
        self.artifacts.put("inv", inv_src, status="proved", language="smt",
                           what=f"The invariant this run found. {what}")
        if rank_src is not None:
            self.artifacts.put(
                "ranking", rank_src, status="proved", language="smt",
                what=f"The ranking function this run found. {what}")

    def _none(self, cd, tried, asker) -> CertificateData:
        what = ("inductive invariant implying the property" if cd.is_safety
                else "invariant and ranking function")
        detail = (
            f"Vampire was asked for one as the coefficients of "
            f"{' and '.join(tried) or 'no'} template(s), over "
            f"{len(self.ob.s)} component(s), in {asker.calls} call(s) taking "
            f"{asker.spent:.1f} s. An answer literal comes back only when a "
            f"refutation pins the coefficients down, so this is not a proof "
            f"that no certificate of this shape exists -- a longer "
            f"--vampire-timeout, or a shape outside the templates, may be "
            f"what is missing. `--infer houdini` searches the same module by "
            f"proposing facts and proving them, which is a different reach."
        )
        if self.artifacts is not None:
            self.artifacts.note(
                f"--infer vampire derived no {what}. {detail}\n",
                status="unknown",
                what=f"A derivation of {what} that did not succeed.",
                why="Vampire answers or times out; nothing was refuted.",
            )
        raise Refused(f"--infer vampire derived no {what}. {detail}")
