"""TA2Magic by theorem proving: candidates the Vampire prover proves.

``--infer vampire``.  Vampire is a first-order theorem prover that reads
SMT-LIB, and what that changes about a search is that its answer is
one-sided.  Handed the *negation* of an obligation it either refutes it --
the obligation holds, and the refutation is the warrant -- or it runs out of
time.  It never says "false" and never hands back a model, so there is no
counterexample to guide anything by and nothing here asks it to *find* a
certificate.  The route proposes, and Vampire decides what stays:

* **the invariant** is Houdini's.  Candidate facts are read off the module
  and off simulated runs of it -- the bound a component stays within, stated
  with a constant the program mentions; a congruence the runs keep; a
  relation between two components; the bounds a component keeps on each
  side of a Bool flag; and under ``--safety`` the property's own conjuncts.
  Each is put to Vampire (it holds at entry; a round preserves it given the
  others) and dropped when not proved, until a pass drops nothing.
* **the ranking function** (``--buchi``) is one of a fixed list of shapes --
  an affine form of one or two components, or ``K*x + y`` for a
  lexicographic pair, shifted to stay positive where the property fails and
  optionally zeroed where it holds -- tried smallest first against
  ``rule_buchi``'s own ``hrank``.
* **the certificate is then cut down to what the proofs used.**  Vampire
  reports an unsat core, which names the invariant facts a refutation
  needed, so the invariant handed on is the closure of those under the
  preservation proofs rather than everything Houdini kept.  Every fact is
  one more `step_inv` implication for Lean to close.

Two things follow from the one-sidedness, both measured.

**A candidate that is false costs a whole time limit**, because nothing
comes back early to say so.  That is why the module is *simulated* first:
a fact some reachable state violates is gone before Vampire is asked, which
is free, and what is left is mostly true.  The runs quantify the inputs the
way the obligations do -- each round's inputs drawn afresh, subject to
``--pre`` -- so a state they reach is one every certificate has to cover,
and under ``--safety`` a reached state where the property fails ends the
route there, with the state.

**The time limit climbs: 2 s, 10 s, 60 s, then what is left of
``--vampire-timeout``.**  Vampire's portfolio divides its time limit among
its strategies, so a short limit is a different schedule rather than a
truncated long one, and a proof one limit finds another may not.  Measured
on the 399 obligations of certificates the benchmark matrix verified, the
`smtcomp` portfolio refuted 369 within 2 s, nearly all in 10-20 ms; the rest
were encodings it cannot read (bitvectors, tuples, reals) or obligations
that are false.  So a whole search runs at one limit, and only a search
that found nothing is repeated at the next -- proofs already found are
kept, and only what failed is asked again.

What the route needs of a module
================================
Scalar ``Int`` and ``Bool`` state and inputs.  Vampire's SMT-LIB front end
has integer and real arithmetic and datatypes but no bitvectors, a Real
component has no integer reading for a ranking function to land in `Nat`
by, and a matrix-shaped one would be a column per element; each is refused
by name before Vampire is started.
"""

from __future__ import annotations

import os
import random
import shutil
import signal
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from .cert import CertificateData
from .common import Refused
from .magic import TA2Magic
from .smt_synth import SynthContext, affine_smt, moduli, program_constants, smt_int

try:
    import cvc5                                      # type: ignore
    from cvc5 import Kind
except ImportError:                                  # pragma: no cover
    cvc5 = None
    Kind = None

# The rungs below the final one, in seconds. The final rung is whatever is
# left of `--vampire-timeout`.
LADDER = (2, 10, 60)
DEFAULT_TIMEOUT = 120
DEFAULT_CORES = 4
ENV = "VAMPIRE"

# How much of the module the simulation may evaluate, and for how long. The
# runs only *filter*: a fact they keep is still proved or dropped by Vampire.
_MAX_ROUNDS = 4000
_SIM_SECONDS = 3.0
_TRIES = 64                     # input draws per round before a run is ended
_OVERFLOW = 10**9               # a run past this is diverging, not informing

# Sampled rounds: how long one batch may take, and how many batches a
# Houdini pass draws before asking Vampire.
_SAMPLE_SECONDS = 1.0
_WIDE = 10**6
_SAMPLE_PASSES = 4
# The most a certificate's minimisation may take, of what is left.
_MINIMISE_SECONDS = 15.0

# A ranking search asks Vampire about at most this many shapes, smallest
# first; fits them to at most this many rounds; branches on at most this many
# conditions, with at most this many forms on each side of one.
_MAX_RANKS = 48
_MAX_OUTSIDE = 1000
_MAX_SPLITS = 24
_MAX_BRANCH_FORMS = 6
# Up to this many components, every {-1, 0, 1} combination is a form; past
# it, one or two components at a time.
_MAX_DENSE = 5
# Constant-derived shifts offered above a fitted one, per form.
_MAX_SHIFTS = 2


# ══════════════════════════════════════════════════════════════════════════
# Finding the binary
# ══════════════════════════════════════════════════════════════════════════


def resolve_vampire(spec: "str | None") -> str:
    """`--vampire` as an executable, else `$VAMPIRE`, else `vampire` on PATH.

    Resolved at parse time, so a run that cannot start the prover says so
    before a project is generated rather than after. A directory is accepted
    when it holds a `vampire` -- the release zip unpacks to one.
    """
    for source, value in (("--vampire", spec), (f"${ENV}", os.environ.get(ENV))):
        if not value:
            continue
        path = Path(value).expanduser()
        if path.is_dir():
            path = path / "vampire"
        if path.is_file():
            if not os.access(path, os.X_OK):
                raise Refused(f"{source}: {path} is not executable")
            return str(path)
        found = shutil.which(value)
        if found:
            return found
        raise Refused(
            f"{source}: no such file: {path}. Pass the Vampire binary, a "
            f"directory holding one, or a name on PATH."
        )
    found = shutil.which("vampire")
    if found:
        return found
    raise Refused(
        "--infer vampire needs the Vampire prover: pass --vampire PATH, set "
        f"${ENV}, or put `vampire` on PATH. Builds for Linux and macOS are at "
        "https://github.com/vprover/vampire/releases"
    )


def ladder(total: float) -> tuple[float, ...]:
    """The time limits a search runs at, in order, ending at `total`."""
    return tuple(t for t in LADDER if t < total) + (float(total),)


# ══════════════════════════════════════════════════════════════════════════
# Asking Vampire
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Answer:
    proved: bool
    secs: float = 0.0
    core: "frozenset[str] | None" = None    # names, when a core was asked for


class Vampire:
    """The prover, on this run's deadline, with what it has already answered.

    A proof is kept whatever limit found it; a failure is kept with the limit
    it failed at, so the next rung asks again only what the last one could
    not answer. Queries are SMT-LIB text, so the text is the key.
    """

    def __init__(self, exe: str, *, seconds: float, cores: int, log=print):
        self.exe = exe
        self.cores = max(1, int(cores))
        self.deadline = time.monotonic() + seconds
        # Portfolio calls side by side. Each spreads its strategies over
        # `cores` processes; more calls than the machine has cores for would
        # only make every limit mean less time.
        self.jobs = max(1, min(4, (os.cpu_count() or 4) // self.cores))
        self.log = log
        self.calls = 0
        self.spent = 0.0
        self._proved: dict[str, Answer] = {}
        self._failed: dict[str, float] = {}

    def left(self) -> float:
        return self.deadline - time.monotonic()

    def prove(self, script: str, limit: float, *, core: bool = False) -> Answer:
        hit = self._proved.get(script)
        if hit is not None and (hit.core is not None or not core):
            return hit
        if self._failed.get(script, 0.0) >= limit:
            return Answer(False)
        budget = min(limit, self.left())
        if budget < 0.5:
            return Answer(False)
        answer = self._run(script, budget, core)
        if answer.proved:
            self._proved[script] = answer
        else:
            self._failed[script] = max(self._failed.get(script, 0.0), limit)
        return answer

    def prove_all(self, scripts: list[str], limit: float) -> list[Answer]:
        if len(scripts) <= 1 or self.jobs == 1:
            return [self.prove(s, limit) for s in scripts]
        with ThreadPoolExecutor(self.jobs) as pool:
            return list(pool.map(lambda s: self.prove(s, limit), scripts))

    def first(self, scripts: list[str], limit: float) -> "int | None":
        """The index of the first script proved, trying `jobs` at a time."""
        for at in range(0, len(scripts), self.jobs):
            if self.left() < 0.5:
                return None
            batch = scripts[at:at + self.jobs]
            for i, answer in enumerate(self.prove_all(batch, limit)):
                if answer.proved:
                    return at + i
        return None

    def _run(self, script: str, budget: float, core: bool) -> Answer:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".smt2", prefix="verith-vampire-", delete=False
        ) as f:
            f.write(script)
            path = f.name
        cmd = [
            self.exe, "--input_syntax", "smtlib2",
            "--mode", "portfolio", "--schedule", "smtcomp",
            "--cores", str(self.cores),
            "--output_mode", "ucore" if core else "smtcomp",
            # Deciseconds: a limit below a second is a real limit.
            "--time_limit", f"{max(1, int(budget * 10))}d",
            path,
        ]
        t0 = time.perf_counter()
        # Its own session: the portfolio forks a process per strategy, and a
        # run killed from outside has to take them all with it.
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
        lines = [ln.strip() for ln in out.splitlines()]
        if "unsat" in lines:
            names = None
            if core:
                at = lines.index("unsat") + 1
                names = frozenset(
                    ln for ln in lines[at:] if ln and ln not in "()"
                    and not ln.startswith("%")
                )
            return Answer(True, dt, names)
        error = next((ln for ln in lines
                      if "User error" in ln or "Parsing Error" in ln), None)
        if error is not None:
            detail = out[out.find(error):].strip().splitlines()
            raise Refused(
                f"Vampire cannot read this module's encoding: "
                f"{' '.join(detail[:2])[:300]}"
            )
        return Answer(False, dt)


def _kill_group(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()


# ══════════════════════════════════════════════════════════════════════════
# The obligations, as SMT-LIB
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Candidate:
    """One invariant fact or ranking function: its source and its term."""

    src: str                # SMT-LIB over `s0..`, what the certificate carries
    term: object            # the same, parsed, over `ctx.state`


class Obligations:
    """One module's obligations, written for Vampire.

    The round is stated once per script as `define-fun`s -- the successor
    state `v_sp*` over the latched state `v_s*` and the inputs -- and every
    fact is read at it, so a query about twenty facts carries the transition
    once rather than twenty times.
    """

    def __init__(self, ctx: SynthContext):
        tm = ctx.tm
        self.ctx = ctx

        def consts(vs, prefix):
            return [tm.mkConst(v.getSort(), f"{prefix}{i}")
                    for i, v in enumerate(vs)]

        self.s = consts(ctx.state, "v_s")
        self.sp = consts(ctx.state, "v_sp")
        self.si = consts(ctx.state, "v_si")
        self.el = consts(ctx.extl_latched, "v_el")
        self.en = consts(ctx.extl_next, "v_en")
        rewriter = cvc5.Solver(tm)

        def readable(t):
            """`t`, with the tuples a matrix-shaped intermediate leaves folded.

            A module with scalar state can still compute through a matrix --
            `m_max` takes the max of a 2-vector -- and cvc5 encodes that as
            `tuple`/`tuple.select`, which Vampire's SMT-LIB front end does not
            have. The rewriter folds a select of a constructor away, so the
            round Vampire is shown is the same function without them. It is
            only called where a tuple is printed: a term that has none is
            handed over exactly as encoded.
            """
            return rewriter.simplify(t) if "tuple" in str(t) else t

        self.next = [readable(t) for t in ctx.msmt.update_state(self.s, self.el, self.en)]
        self.init = [readable(t) for t in ctx.msmt.init_state(self.en)]
        self.update_pre = readable(ctx.with_inputs(ctx.update_pre, self.el, self.en))
        self.init_pre = readable(ctx.with_inputs(ctx.init_pre, self.el, self.en))

    # --- the queries ------------------------------------------------------

    def holds_at_entry(self, facts: list[Candidate]) -> str:
        """`init_pre e -> fact (init e)`, for all of `facts` at once."""
        goal = self._and([self._at(f.term, self.si) for f in facts])
        return self._script([("pre", self.init_pre)], goal,
                            defines=zip(self.si, self.init))

    def preserved(self, hyps: list[Candidate], goals: list[Candidate]) -> str:
        """`hyps s /\\ update_pre e -> goals (update s e)`."""
        named = [(f"h{i}", self._at(h.term, self.s)) for i, h in enumerate(hyps)]
        goal = self._and([self._at(g.term, self.sp) for g in goals])
        return self._script(named + [("pre", self.update_pre)], goal,
                            defines=zip(self.sp, self.next))

    def implies(self, hyps: list[Candidate], goal) -> str:
        """`hyps s -> goal s`: the invariant is a proof of the property."""
        named = [(f"h{i}", self._at(h.term, self.s)) for i, h in enumerate(hyps)]
        return self._script(named, self._at(goal, self.s))

    def drops(self, hyps: list[Candidate], rank: Candidate) -> str:
        """`hrank`: `inv s /\\ ~P s /\\ update_pre e -> V (update s e) < V s`.

        With `V = Int.toNat rank`, exactly as `rule_buchi` states it and
        `smt_query.check_hrank` restates it: a clamp on both sides, so the
        rank may be anything where the property holds.
        """
        tm = self.ctx.tm
        named = [(f"h{i}", self._at(h.term, self.s)) for i, h in enumerate(hyps)]
        named += [
            ("notP", tm.mkTerm(Kind.NOT, self._at(self.ctx.prp, self.s))),
            ("pre", self.update_pre),
        ]
        goal = tm.mkTerm(
            Kind.LT,
            self._clamp(self._at(rank.term, self.sp)),
            self._clamp(self._at(rank.term, self.s)),
        )
        return self._script(named, goal, defines=zip(self.sp, self.next))

    # --- plumbing ---------------------------------------------------------

    def _at(self, term, vs):
        return term.substitute(self.ctx.state, vs)

    def _and(self, terms):
        tm = self.ctx.tm
        if not terms:
            return tm.mkBoolean(True)
        return terms[0] if len(terms) == 1 else tm.mkTerm(Kind.AND, *terms)

    def _clamp(self, t):
        tm = self.ctx.tm
        zero = tm.mkInteger(0)
        return tm.mkTerm(Kind.ITE, tm.mkTerm(Kind.GEQ, t, zero), t, zero)

    def _script(self, hyps, goal, *, defines=()) -> str:
        """Named hypotheses and a goal, as a script whose `unsat` is a proof.

        The goal is asserted negated. Hypotheses are named so that an unsat
        core can say which of them the proof used; the goal is named too, so
        a core is never empty for a reason other than the goal being false.
        """
        from .smt_query import _constants

        defines = list(defines)
        seen: dict = {}
        for _v, body in defines:
            _constants(body, seen)
        for _n, t in hyps:
            _constants(t, seen)
        _constants(goal, seen)
        defined = {str(v) for v, _b in defines}
        lines = ["(set-logic ALL)"]
        lines += [f"(declare-fun {sym} () {seen[sym].getSort()})"
                  for sym in sorted(seen) if sym not in defined]
        lines += [f"(define-fun {v} () {v.getSort()} {body})"
                  for v, body in defines]
        lines += [f"(assert (! {t} :named {n}))" for n, t in hyps]
        lines.append(f"(assert (! (not {goal}) :named goal))")
        lines.append("(check-sat)")
        return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════


class _Unevaluable(Exception):
    """A term with no value here: a division by zero, or an op not covered."""


# The value of a division by zero. SMT-LIB leaves it unspecified, so nothing
# is claimed about a state that needs one: it propagates, and a result that
# depends on it is `_Unevaluable`.
_UNDEF = object()


def _strict(fn):
    def apply(vals):
        return _UNDEF if any(v is _UNDEF for v in vals) else fn(vals)
    return apply


def _ite(vals):
    c = vals[0]
    return _UNDEF if c is _UNDEF else vals[1] if c else vals[2]


def _and(vals):
    if any(v is False for v in vals):
        return False
    return _UNDEF if any(v is _UNDEF for v in vals) else True


def _or(vals):
    if any(v is True for v in vals):
        return True
    return _UNDEF if any(v is _UNDEF for v in vals) else False


def _product(vals):
    out = 1
    for v in vals:
        out *= v
    return out


def _chain(op):
    return _strict(lambda vs: all(op(a, b) for a, b in zip(vs, vs[1:])))


def _division(vals):
    a, b = vals
    if a is _UNDEF or b is _UNDEF or b == 0:
        return _UNDEF
    return (a - a % abs(b)) // b


def _modulus(vals):
    a, b = vals
    if a is _UNDEF or b is _UNDEF or b == 0:
        return _UNDEF
    # SMT-LIB's is Euclidean: the remainder is never negative.
    return a % abs(b)


_OPS = {} if Kind is None else {
    Kind.ITE: _ite,
    Kind.AND: _and,
    Kind.OR: _or,
    Kind.NOT: _strict(lambda vs: not vs[0]),
    Kind.IMPLIES: lambda vs: _or([_strict(lambda v: not v[0])(vs[:1]), vs[1]]),
    Kind.XOR: _strict(lambda vs: vs[0] != vs[1]),
    Kind.ADD: _strict(sum),
    Kind.SUB: _strict(lambda vs: vs[0] - sum(vs[1:])),
    Kind.NEG: _strict(lambda vs: -vs[0]),
    Kind.MULT: _strict(_product),
    Kind.ABS: _strict(lambda vs: abs(vs[0])),
    Kind.EQUAL: _strict(lambda vs: all(v == vs[0] for v in vs[1:])),
    Kind.DISTINCT: _strict(lambda vs: len(set(vs)) == len(vs)),
    Kind.LT: _chain(lambda a, b: a < b),
    Kind.LEQ: _chain(lambda a, b: a <= b),
    Kind.GT: _chain(lambda a, b: a > b),
    Kind.GEQ: _chain(lambda a, b: a >= b),
    Kind.INTS_DIVISION: _division,
    Kind.INTS_MODULUS: _modulus,
}


class Evaluator:
    """cvc5 terms at concrete values, in Python.

    A term is compiled once into a flat list of operations, one per distinct
    subterm, so a transition that shares its subterms -- the `let`s a deep
    network prints with -- is evaluated in time linear in its DAG rather than
    its tree. The kinds a module's integer transition is made of are Python
    arithmetic; anything else is handed to cvc5's rewriter with the values
    substituted in, which evaluates a ground term without deciding anything.
    """

    def __init__(self, tm):
        self.tm = tm
        self._solver = None
        self._programs: dict = {}

    def __call__(self, term, env: dict):
        return self.many([term], env)[0]

    def many(self, terms, env: dict) -> list:
        """Several terms at one assignment, sharing what they share."""
        key = tuple(t.getId() for t in terms)
        program = self._programs.get(key)
        if program is None:
            program = self._programs[key] = self._compile(terms)
        vals: list = []
        for kind, arg in program[0]:
            if kind == 0:
                vals.append(arg)
            elif kind == 1:
                try:
                    vals.append(env[arg])
                except KeyError as e:
                    raise _Unevaluable(f"no value for {arg}") from e
            elif kind == 2:
                vals.append(self._rewrite(arg, env))
            else:
                vals.append(kind([vals[j] for j in arg]))
        out = [vals[i] for i in program[1]]
        if any(v is _UNDEF for v in out):
            raise _Unevaluable("division by zero")
        return out

    def _compile(self, terms):
        slot: dict = {}
        ops: list = []
        stack = [(t, False) for t in reversed(list(terms))]
        while stack:
            t, expanded = stack.pop()
            tid = t.getId()
            if tid in slot:
                continue
            k = t.getKind()
            if k == Kind.CONST_INTEGER:
                op = (0, t.getIntegerValue())
            elif k == Kind.CONST_BOOLEAN:
                op = (0, t.getBooleanValue())
            elif k == Kind.CONSTANT:
                op = (1, t.getSymbol())
            elif k not in _OPS:
                op = (2, t)
            elif not expanded:
                stack.append((t, True))
                stack.extend((c, False) for c in reversed(list(t)))
                continue
            else:
                op = (_OPS[k], tuple(slot[c.getId()] for c in t))
            slot[tid] = len(ops)
            ops.append(op)
        return ops, [slot[t.getId()] for t in terms]

    def _rewrite(self, t, env):
        from .smt_query import _constants

        tm = self.tm
        if self._solver is None:
            self._solver = cvc5.Solver(tm)
        consts = list(_constants(t, {}).values())
        vals = []
        for c in consts:
            v = env.get(c.getSymbol())
            if v is None:
                raise _Unevaluable(f"no value for {c.getSymbol()}")
            vals.append(tm.mkBoolean(v) if isinstance(v, bool)
                        else tm.mkInteger(int(v)))
        r = self._solver.simplify(t.substitute(consts, vals) if consts else t)
        if r.getKind() == Kind.CONST_INTEGER:
            return r.getIntegerValue()
        if r.getKind() == Kind.CONST_BOOLEAN:
            return r.getBooleanValue()
        raise _Unevaluable(f"{t.getKind()} does not evaluate to a literal")


# ══════════════════════════════════════════════════════════════════════════
# Simulation
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class Runs:
    """What the simulation reached: states, and the rounds between them.

    A state is a tuple of component values, in `ctx.state` order.
    """

    states: list
    rounds: list
    # Under `--safety`, a reached state where the property fails, and how
    # many rounds it took.
    violation: "tuple | None" = None
    depth: int = 0


class Draws:
    """Input values, and states near the ones a run reached.

    Integers come from the program's own constants half the time -- the
    value a guard compares against is the one worth hitting -- and from a
    range a little wider than those constants otherwise.
    """

    def __init__(self, ctx: SynthContext, constants, seed: int,
                 wide: bool = False):
        self.rng = random.Random(seed)
        self.pool = sorted(set(constants) | {-1, 0, 1})
        self.reach = min(1000, 2 * max(abs(c) for c in self.pool) + 10)
        # `wide` draws integers far past anything the program mentions. A
        # shape fitted to states within the program's own range can hold
        # only *because* they were small -- `11 - x` is positive on every
        # state with `x <= 10` -- and states from a wider range are what tell
        # that apart from a shape the invariant actually bounds.
        if wide:
            self.reach = _WIDE
        self.fresh = 0.5 if wide else 0.25
        self.sorts = [v.getSort() for v in ctx.state]

    def value(self, sort):
        rng = self.rng
        if sort.isBoolean():
            return rng.random() < 0.5
        if rng.random() < 0.5:
            return rng.choice(self.pool)
        return rng.randint(-self.reach, self.reach)

    def near(self, seeds: list) -> tuple:
        """A state: a reached one with some components moved, or a fresh one."""
        rng = self.rng
        if not seeds or rng.random() < self.fresh:
            return tuple(self.value(srt) for srt in self.sorts)
        s = list(rng.choice(seeds))
        for i in rng.sample(range(len(s)), k=rng.randint(1, len(s))):
            if self.sorts[i].isBoolean():
                s[i] = not s[i]
            elif rng.random() < 0.6:
                s[i] = s[i] + rng.choice((-2, -1, 1, 2))
            else:
                s[i] = self.value(self.sorts[i])
        return tuple(s)


def simulate(ctx: SynthContext, ob: Obligations, ev: Evaluator,
             constants: tuple[int, ...], *, safety: bool,
             seed: int = 0) -> Runs:
    """Run the module from its entry states, drawing inputs as `--pre` allows.

    A run is a run of the module: each round's latched inputs are the values
    the round before awaited, and only the awaited ones are drawn. So every
    state reached is reachable -- a fact it violates is not an invariant, and
    a `--safety` property it violates does not hold.
    """
    draws = Draws(ctx, constants, seed)
    names = ctx.names
    latched = [str(c) for c in ob.el]
    awaited = [(str(c), c.getSort()) for c in ob.en]

    def inputs(pre, base, before):
        """Awaited inputs `pre` allows. At entry nothing is latched yet, so
        the latched ones are drawn too -- `init_inv` quantifies them."""
        for _ in range(_TRIES):
            env = dict(base)
            if before is None:
                env.update({n: draws.value(srt)
                            for n, (_, srt) in zip(latched, awaited)})
            else:
                env.update(zip(latched, before))
            env.update({n: draws.value(srt) for n, srt in awaited})
            try:
                if ev(pre, env):
                    return env
            except _Unevaluable:
                return None
        return None

    def prop_fails(state) -> bool:
        try:
            return not ev(ctx.prp, dict(zip(names, state)))
        except _Unevaluable:
            return False

    states: dict = {}
    rounds: dict = {}
    deterministic = not awaited
    trajectories = 1 if deterministic else 32
    length = _MAX_ROUNDS if deterministic else 128
    t_end = time.monotonic() + _SIM_SECONDS
    for _ in range(trajectories):
        if len(rounds) >= _MAX_ROUNDS or time.monotonic() > t_end:
            break
        env = inputs(ob.init_pre, {}, None)
        if env is None:
            continue
        try:
            s = tuple(ev.many(ob.init, env))
        except _Unevaluable:
            continue
        before = [env[n] for n, _ in awaited]
        seen_here = set()
        for depth in range(length + 1):
            states.setdefault(s, None)
            if safety and prop_fails(s):
                return Runs(list(states), list(rounds), violation=s, depth=depth)
            if deterministic and s in seen_here:
                break
            seen_here.add(s)
            if (len(rounds) >= _MAX_ROUNDS or time.monotonic() > t_end
                    or any(not isinstance(v, bool) and abs(v) > _OVERFLOW
                           for v in s)):
                break
            base = {str(c): v for c, v in zip(ob.s, s)}
            env = inputs(ob.update_pre, base, before)
            if env is None:
                break
            try:
                sp = tuple(ev.many(ob.next, env))
            except _Unevaluable:
                break
            before = [env[n] for n, _ in awaited]
            rounds.setdefault((s, sp), None)
            s = sp
    return Runs(list(states), list(rounds))


def sample_rounds(ctx: SynthContext, ob: Obligations, ev: Evaluator,
                  draws: Draws, seeds: list, keep, *, want: int = 300,
                  seconds: float = _SAMPLE_SECONDS) -> list:
    """Rounds from states `keep` accepts, not necessarily reachable ones.

    What an obligation quantifies over is every state the invariant admits,
    with any inputs `--pre` allows -- latched ones included -- so a round
    from such a state that breaks a fact, or that a ranking function does not
    drop on, is a counterexample to that obligation as Vampire would be asked
    it. Finding one here is free, and asking Vampire would cost a whole time
    limit to learn nothing.
    """
    ins = [(str(c), c.getSort()) for c in ob.el + ob.en]
    names = [str(c) for c in ob.s]
    out: list = []
    t_end = time.monotonic() + seconds
    for _ in range(40 * want):
        if len(out) >= want or time.monotonic() > t_end:
            break
        s = draws.near(seeds)
        try:
            if not keep(s):
                continue
            base = dict(zip(names, s))
            for _ in range(8):
                env = dict(base)
                env.update({n: draws.value(srt) for n, srt in ins})
                if ev(ob.update_pre, env):
                    out.append((s, tuple(ev.many(ob.next, env))))
                    break
        except _Unevaluable:
            continue
    return out


# ══════════════════════════════════════════════════════════════════════════
# Candidates
# ══════════════════════════════════════════════════════════════════════════


def _reading(ctx: SynthContext, i: int) -> str:
    return f"(ite s{i} 1 0)" if ctx.env.state_sorts[i].isBoolean() else f"s{i}"


def _bounds(values, constants) -> tuple["int | None", "int | None"]:
    """The tightest program constants below and above every value seen.

    A constant rather than the extreme value seen: the bound a run happens to
    reach is a fact about the run, and the one the program's guard names is
    usually the fact about the program -- `NNInv` needs `x <= 9`, and `9` is
    only in the guard.
    """
    lo, hi = min(values), max(values)
    below = [c for c in constants if c <= lo]
    above = [c for c in constants if c >= hi]
    return (max(below) if below else None, min(above) if above else None)


def invariant_candidates(ctx: SynthContext, runs: Runs,
                         constants: tuple[int, ...],
                         conjuncts: list[str]) -> list[str]:
    """Facts every simulated state satisfies, as SMT-LIB over `s0..`."""
    consts = tuple(sorted(set(constants) | {0}))
    states = runs.states
    ints = [i for i, srt in enumerate(ctx.env.state_sorts) if srt.isInteger()]
    bools = [i for i, srt in enumerate(ctx.env.state_sorts) if srt.isBoolean()]
    out: list[str] = list(conjuncts)
    if not states:
        return out

    def bounded(form: str, values, cs=consts) -> list[str]:
        lo, hi = _bounds(values, cs)
        facts = []
        if len(set(values)) == 1 and len(states) > 1:
            facts.append(f"(= {form} {smt_int(values[0])})")
        if lo is not None:
            facts.append(f"(<= {smt_int(lo)} {form})")
        if hi is not None:
            facts.append(f"(<= {form} {smt_int(hi)})")
        return facts

    for i in ints:
        values = [s[i] for s in states]
        out += bounded(f"s{i}", values)
        distinct = set(values)
        for k in moduli(consts):
            residues = {v % k for v in values}
            if len(residues) == 1 and len(distinct) > 2:
                out.append(f"(= (mod s{i} {k}) {residues.pop()})")
    for i in bools:
        values = {s[i] for s in states}
        if len(values) == 1:
            out.append(f"s{i}" if values.pop() else f"(not s{i})")
            continue
        for flag, lit in ((True, f"s{i}"), (False, f"(not s{i})")):
            side = [s for s in states if s[i] == flag]
            for j in ints:
                for fact in bounded(f"s{j}", [s[j] for s in side]):
                    if fact not in out:
                        out.append(f"(=> {lit} {fact})")
    for a, i in enumerate(ints):
        for j in ints[a + 1:]:
            for form, values in (
                (f"(- s{i} s{j})", [s[i] - s[j] for s in states]),
                (f"(+ s{i} s{j})", [s[i] + s[j] for s in states]),
            ):
                # Relations against the constants near zero only: a sum
                # bounded by some large literal is rarely what holds a run in.
                out += bounded(form, values,
                               tuple(c for c in consts if abs(c) <= 1))
    return list(dict.fromkeys(out))


def split_conditions(ctx: SynthContext, ob: Obligations) -> list[str]:
    """What a piecewise ranking function may branch on, as SMT-LIB over `s0..`.

    First the comparisons the property and the transition branch on -- where
    a run changes what it does, and so where the quantity that falls changes
    too -- then each Bool component, the sign of each integer one, and the
    order of each pair: `max(y, z) - x` falls on `while (x < y) {x++; y = z}`
    and is `(ite (<= y z) (- z x) (- y x))`, a branch the program never
    spells.
    """
    from .smt_query import _constants

    out: list[str] = []
    state = set(ctx.names)
    roots = [ctx.prp] + [t.substitute(ob.s, ctx.state) for t in ob.next]
    seen: set = set()
    stack = list(roots)
    while stack:
        t = stack.pop()
        if t.getId() in seen:
            continue
        seen.add(t.getId())
        k = t.getKind()
        kids = list(t)
        if k in (Kind.LT, Kind.LEQ, Kind.GT, Kind.GEQ) or (
            k == Kind.EQUAL and kids and kids[0].getSort().isInteger()
        ):
            if set(_constants(t, {})) <= state:
                out.append(str(t))
        stack.extend(kids)
    ints = [i for i, srt in enumerate(ctx.env.state_sorts) if srt.isInteger()]
    for i, srt in enumerate(ctx.env.state_sorts):
        out.append(f"s{i}" if srt.isBoolean() else f"(<= 0 s{i})")
    out += [f"(<= s{i} s{j})" for a, i in enumerate(ints) for j in ints[a + 1:]]
    return list(dict.fromkeys(out))[:_MAX_SPLITS]


class Ranks:
    """Ranking functions fitted to rounds, in a few fixed shapes.

    Each shape is an affine form of the state's integer readings -- one
    component, a sum or difference of two, or `K*x + y` for a lexicographic
    pair -- and the *shift* that makes it at least one on every round that
    has to drop is read off those rounds rather than enumerated. A piecewise
    rank `(ite c f g)` fits a form to each side of `c` the same way, and the
    offset between the sides from the rounds that cross from one to the
    other. Every shape is also offered zeroed where the property holds,
    since `hrank` never constrains those states.
    """

    def __init__(self, ctx: SynthContext, ev: Evaluator, prp_src: str,
                 splits: list[str], constants: tuple[int, ...] = ()):
        self.ctx, self.ev, self.prp_src = ctx, ev, prp_src
        # Shifts a guard's constant suggests: `c` and one past it. A shift
        # fitted to sampled rounds is only as low as the lowest one sampled,
        # and the round that needs the largest shift is usually the corner
        # the guard names -- `y <= 100 && z <= x` needs `101 + x - y - z`,
        # where samples that never hit `y = 100, z = x` exactly fit `95`.
        self.shift_pool = sorted({c + d for c in constants for d in (0, 1)})
        self.names = ctx.names
        n = len(self.names)
        self.readings = [_reading(ctx, i) for i in range(n)]
        if n <= _MAX_DENSE:
            # Every form with coefficients in {-1, 0, 1}: `100 - y + x - z`
            # ranks `ColonSipma-TACAS2001-Fig1`, and no pair of its three
            # components does.
            forms = [v for v in product((1, -1, 0), repeat=n) if any(v)]
        else:
            forms = []
            for i in range(n):
                for sign in (1, -1):
                    forms.append(tuple(sign if k == i else 0 for k in range(n)))
            for i in range(n):
                for j in range(i + 1, n):
                    for si, sj in ((1, -1), (-1, 1), (1, 1), (-1, -1)):
                        forms.append(tuple(si if k == i else sj if k == j else 0
                                           for k in range(n)))
        forms.sort(key=lambda v: sum(map(abs, v)))
        self.forms = forms
        self.splits = []
        for src in splits:
            try:
                term = ctx.env.parse_expr(src)
            except Exception:                        # noqa: BLE001
                continue
            if not term.isNull():
                self.splits.append((src, term))
        self._truth: dict = {}

    def holds(self, state, term=None) -> bool:
        key = (state, None if term is None else term.getId())
        if key not in self._truth:
            try:
                self._truth[key] = bool(self.ev(
                    self.ctx.prp if term is None else term,
                    dict(zip(self.names, state))))
            except _Unevaluable:
                self._truth[key] = term is None
        return self._truth[key]

    def fit(self, rounds: list) -> list[str]:
        """Every shape that drops on all of `rounds`, smallest first."""
        outside = [(s, sp) for s, sp in rounds if not self.holds(s)]
        if len(outside) > _MAX_OUTSIDE:
            outside = random.Random(0).sample(outside, _MAX_OUTSIDE)
        if not outside:
            # Nothing to rank on. A constant is then the first thing worth
            # asking: it proves the property holds wherever the invariant does.
            return ["0"]
        forms = list(self.forms) + self._lex(outside)
        out: list[str] = []
        for form in dict.fromkeys(forms):
            shift = 1 - min(self._at(form, s) for s, _ in outside)
            above = [c for c in self.shift_pool if c > shift][:_MAX_SHIFTS]
            for sh in dict.fromkeys((0, shift, *above)):
                src = affine_smt(sh, form, self.readings)
                out += self._accept(src, lambda st, f=form, sh=sh: self._at(f, st) + sh,
                                    outside)
        for c_src, c_term in self.splits:
            out += self._piecewise(c_src, c_term, outside)
        out = list(dict.fromkeys(out))
        out.sort(key=len)
        return out

    # --- shapes -----------------------------------------------------------

    def _lex(self, outside) -> list:
        """`K*x + y`, with `K` wider than the range `y` takes on these rounds."""
        n = len(self.names)
        out = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ys = [int(s[j]) for s, _ in outside]
                width = max(ys) - min(ys) + 1
                if width > 1000:
                    continue
                K = max(2, width + 1)
                for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    out.append(tuple(si * K if k == i else sj if k == j else 0
                                     for k in range(n)))
        return out

    def _piecewise(self, c_src, c_term, outside) -> list[str]:
        side = {b: [(s, sp) for s, sp in outside
                    if self.holds(s, c_term) == b] for b in (True, False)}
        if not side[True] or not side[False]:
            return []
        zero = tuple(0 for _ in self.names)
        fits = {}
        for b in (True, False):
            stays = [(s, sp) for s, sp in side[b] if self.holds(sp, c_term) == b]
            fits[b] = [f for f in [zero] + self.forms
                       if all(self._at(f, sp) < self._at(f, s) for s, sp in stays)
                       ][:_MAX_BRANCH_FORMS]
        out = []
        for f in fits[True]:
            for g in fits[False]:
                sf = 1 - min(self._at(f, s) for s, _ in side[True])
                sg = 1 - min(self._at(g, s) for s, _ in side[False])
                offsets = self._offsets(c_term, f, g, sf, sg, side)
                if offsets is None:
                    continue
                sf, sg = sf + offsets[0], sg + offsets[1]
                src = (f"(ite {c_src} {affine_smt(sf, f, self.readings)} "
                       f"{affine_smt(sg, g, self.readings)})")

                def rank(st, f=f, g=g, sf=sf, sg=sg):
                    if self.holds(st, c_term):
                        return self._at(f, st) + sf
                    return self._at(g, st) + sg

                out += self._accept(src, rank, outside)
        return out

    def _offsets(self, c_term, f, g, sf, sg, side):
        """How far to lift each side so that crossing to the other drops."""
        of = og = 0
        for _ in range(6):
            moved = False
            for s, sp in side[True]:
                if not self.holds(sp, c_term):
                    need = max(self._at(g, sp) + sg + og, 0) + 1 - (self._at(f, s) + sf)
                    if need > of:
                        of, moved = need, True
            for s, sp in side[False]:
                if self.holds(sp, c_term):
                    need = max(self._at(f, sp) + sf + of, 0) + 1 - (self._at(g, s) + sg)
                    if need > og:
                        og, moved = need, True
            if not moved:
                return of, og
        return None                                  # each side lifts the other

    def _accept(self, src, rank, outside) -> list[str]:
        """`src`, and its zeroed form, for whichever drops on every round."""
        out = []
        before = [(rank(s), sp) for s, sp in outside]
        if all(_drops(r, rank(sp)) for r, sp in before):
            out.append(src)
        if all(_drops(r, 0 if self.holds(sp) else rank(sp)) for r, sp in before):
            out.append(f"(ite {self.prp_src} 0 {src})")
        return out

    @staticmethod
    def _at(form, state) -> int:
        return sum(c * int(v) for c, v in zip(form, state) if c)


def _drops(before: int, after: int) -> bool:
    """`Int.toNat after < Int.toNat before`."""
    return max(after, 0) < max(before, 0)


# ══════════════════════════════════════════════════════════════════════════
# The route
# ══════════════════════════════════════════════════════════════════════════


class TA2MagicVampire(TA2Magic):
    """Houdini and a ranking search, with Vampire as the only judge."""

    def __init__(self, module, *, vampire: str, timeout: float = DEFAULT_TIMEOUT,
                 cores: int = DEFAULT_CORES, artifacts=None, log=print):
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
        self.cores = int(cores)
        self.artifacts = artifacts
        self.log = log

    # --- driver ---------------------------------------------------------

    def infer(self, cd: CertificateData) -> CertificateData:
        ctx = SynthContext.build(self.module, cd, route="vampire")
        self._check_sorts(ctx)
        self.ctx = ctx
        self.ob = ob = Obligations(ctx)
        self.ev = ev = Evaluator(ctx.tm)
        constants = program_constants(ctx)
        self.log("[vampire] columns: "
                 + ", ".join(_reading(ctx, i) for i in range(len(ctx.state))))

        runs = simulate(ctx, ob, ev, constants, safety=cd.is_safety)
        self.log(f"[vampire] simulated {len(runs.rounds)} rounds, "
                 f"{len(runs.states)} distinct states")
        if runs.violation is not None:
            state = ", ".join(f"s{i} = {v}" for i, v in enumerate(runs.violation))
            raise Refused(
                f"--infer vampire: the property does not hold -- a run of "
                f"{runs.depth} round(s), with inputs drawn as `--pre` allows, "
                f"reaches {state}. No invariant implies a property the module "
                f"violates, so there is nothing to put to Vampire."
            )
        self.runs = runs
        self.draws = Draws(ctx, constants, seed=1)
        self.wide = Draws(ctx, constants, seed=2, wide=True)

        conjuncts = self._conjuncts(ctx, cd) if cd.is_safety else []
        facts = self._parse(invariant_candidates(ctx, runs, constants, conjuncts))
        self.log(f"[vampire] {len(facts)} candidate facts hold on the runs")
        ranks = None if cd.is_safety else Ranks(ctx, ev, cd.prp,
                                                split_conditions(ctx, ob),
                                                constants)

        prover = Vampire(self.exe, seconds=self.timeout, cores=self.cores,
                         log=self.log)
        found, reached, tried = None, 0.0, 0
        for limit in ladder(self.timeout):
            if prover.left() < 0.5:
                break
            reached = limit
            self.log(f"[vampire] time limit {limit:g} s "
                     f"({prover.left():.0f} s of the budget left)")
            inv = self._houdini(prover, facts, limit)
            self.log(f"[vampire] Houdini kept {len(inv)} of {len(facts)}: "
                     + " ".join(f.src for f in inv))
            if cd.is_safety:
                need = self._property_rests_on(prover, inv, conjuncts, limit)
                if need is not None:
                    inv = self._shrink(prover, inv, need, limit)
                    found = (self._minimise(prover, inv, conjuncts, None), None)
                    break
                continue
            candidates = self._ranks(ranks, inv)
            tried = max(tried, len(candidates))
            if not candidates:
                continue
            at = prover.first([ob.drops(inv, r) for r in candidates], limit)
            if at is not None:
                rank = candidates[at]
                self.log(f"[vampire] ranking function proved: {rank.src}")
                core = prover.prove(ob.drops(inv, rank), limit, core=True).core
                inv = self._shrink(prover, inv, self._named(inv, core), limit)
                found = (self._minimise(prover, inv, [], (ranks, rank)), rank)
                break
        self.log(f"[vampire] {prover.calls} Vampire calls, {prover.spent:.1f} s")
        if found is None:
            return self._none(cd, len(facts), tried, reached, prover)
        inv, rank = found
        return self._emit(cd, inv, rank)

    # --- Houdini ----------------------------------------------------------

    def _holds(self, fact: Candidate, state) -> bool:
        try:
            return bool(self.ev(fact.term, dict(zip(self.ctx.names, state))))
        except _Unevaluable:
            return True

    def _houdini(self, prover: Vampire, facts: list[Candidate],
                 limit: float) -> list[Candidate]:
        """The largest subset of `facts` Vampire proves inductive at `limit`.

        Each pass first drops what a sampled round refutes -- a state every
        kept fact holds at, whose successor breaks one -- which is Houdini's
        own step taken without a prover, and then asks Vampire about the
        rest.
        """
        ob = self.ob
        if not facts:
            return []
        kept = list(facts)
        if not prover.prove(ob.holds_at_entry(kept), limit).proved:
            answers = prover.prove_all([ob.holds_at_entry([f]) for f in kept], limit)
            kept = [f for f, a in zip(kept, answers) if a.proved]
        while kept and prover.left() >= 0.5:
            kept = self._refuted_by_sampling(kept)
            if not kept:
                break
            if prover.prove(ob.preserved(kept, kept), limit).proved:
                return kept
            answers = prover.prove_all([ob.preserved(kept, [f]) for f in kept], limit)
            survivors = [f for f, a in zip(kept, answers) if a.proved]
            if len(survivors) == len(kept):
                return kept
            kept = survivors
        # Out of budget mid-pass: what is left was not shown inductive.
        return [] if kept and prover.left() < 0.5 else kept

    def _refuted_by_sampling(self, kept: list[Candidate]) -> list[Candidate]:
        for n in range(_SAMPLE_PASSES):
            rounds = sample_rounds(
                self.ctx, self.ob, self.ev,
                self.wide if n % 2 else self.draws, self.runs.states,
                lambda s: all(self._holds(f, s) for f in kept))
            broken = {i for i, f in enumerate(kept)
                      if any(not self._holds(f, sp) for _, sp in rounds)}
            if not broken:
                break
            kept = [f for i, f in enumerate(kept) if i not in broken]
        return kept

    def _ranks(self, ranks: Ranks, inv: list[Candidate]) -> list[Candidate]:
        """Shapes fitted to the rounds `inv` admits, smallest first.

        Fitted to the reached rounds and to sampled ones from states `inv`
        admits where the property fails -- `hrank` quantifies over all of
        those, so a shape that does not drop on a sampled round would only
        cost Vampire a time limit.
        """
        sampled = sample_rounds(
            self.ctx, self.ob, self.ev, self.draws, self.runs.states,
            lambda s: not ranks.holds(s) and all(self._holds(f, s) for f in inv))
        fitted = self._parse(ranks.fit(list(self.runs.rounds) + sampled))
        wide = sample_rounds(
            self.ctx, self.ob, self.ev, self.wide, self.runs.states,
            lambda s: not ranks.holds(s) and all(self._holds(f, s) for f in inv))
        kept = [r for r in fitted if self._drops_on(r, wide)]
        self.log(f"[vampire] {len(fitted)} ranking functions drop on "
                 f"{len(self.runs.rounds)} reached and {len(sampled)} sampled "
                 f"rounds; {len(kept)} also on {len(wide)} rounds from a wider "
                 f"range")
        return kept[:_MAX_RANKS]

    def _drops_on(self, rank: Candidate, rounds: list) -> bool:
        names = self.ctx.names
        for s, sp in rounds:
            try:
                if not _drops(self.ev(rank.term, dict(zip(names, s))),
                              self.ev(rank.term, dict(zip(names, sp)))):
                    return False
            except _Unevaluable:
                continue
        return True

    def _property_rests_on(self, prover, inv, conjuncts,
                           limit) -> "list[Candidate] | None":
        """The facts of `inv` that imply the property, or `None` if it does not.

        Houdini keeps a seeded conjunct exactly when it is inductive with the
        rest, so the usual answer is the conjuncts themselves. When one was
        dropped the rest may still imply it, and then the proof's core says
        which facts that took.
        """
        mine = [f for f in inv if f.src in conjuncts]
        if len(mine) == len(conjuncts):
            return mine
        if not inv:
            return None
        answer = prover.prove(self.ob.implies(inv, self.ctx.prp), limit, core=True)
        return self._named(inv, answer.core) if answer.proved else None

    # --- cutting the certificate down ---------------------------------------

    def _shrink(self, prover: Vampire, inv: list[Candidate],
                need: list[Candidate], limit: float) -> list[Candidate]:
        """The facts `need` rests on, closed under the preservation proofs.

        Each proof of "a round preserves `need`" is asked from all of `inv`,
        and its core names the facts it used; those join `need` until a proof
        uses nothing new. The set that comes out is inductive by those
        proofs, and it is checked once more on its own before it replaces
        `inv` -- a core Vampire over-trims would otherwise surface as a Lean
        failure. Anything that does not come back proved keeps `inv` whole.
        """
        ob = self.ob
        need = list(dict.fromkeys(need))
        while True:
            if not need:
                return need
            answer = prover.prove(ob.preserved(inv, need), limit, core=True)
            if not answer.proved or answer.core is None:
                return inv
            more = [f for f in self._named(inv, answer.core) if f not in need]
            if not more:
                break
            need += more
        if len(need) == len(inv):
            return inv
        if prover.prove(ob.preserved(need, need), limit).proved:
            self.log(f"[vampire] invariant cut to the {len(need)} of "
                     f"{len(inv)} facts its proofs use")
            return need
        return inv

    def _minimise(self, prover: Vampire, inv: list[Candidate],
                  keep: list[str], ranked) -> list[Candidate]:
        """`inv` with each fact the rest do without taken out, one at a time.

        An unsat core is whatever the refutation happened to touch, not the
        least it could have, so what `_shrink` returns can still carry facts
        nothing needs. Dropping one is tried against sampled rounds first --
        a state the rest admit whose successor breaks one of them, or that
        the ranking function does not drop on -- and only a drop no sample
        refutes is put to Vampire, at a short limit and on a small share of
        the budget: a smaller certificate is worth having, and not worth the
        search that found it.
        """
        ob = self.ob
        deadline = time.monotonic() + min(_MINIMISE_SECONDS, max(prover.left(), 0) / 4)
        kept = list(inv)
        for fact in reversed(list(inv)):
            if fact.src in keep or time.monotonic() > deadline:
                continue
            rest = [f for f in kept if f is not fact]
            if self._sampled_counterexample(rest, ranked):
                continue
            scripts = [ob.preserved(rest, rest)]
            if ranked is not None:
                scripts.append(ob.drops(rest, ranked[1]))
            limit = min(LADDER[0], deadline - time.monotonic())
            if limit > 0.5 and all(a.proved for a in prover.prove_all(scripts, limit)):
                kept = rest
        if len(kept) < len(inv):
            self.log(f"[vampire] invariant minimised to {len(kept)} of {len(inv)} facts")
        return kept

    def _sampled_counterexample(self, facts: list[Candidate], ranked) -> bool:
        ranks, rank = ranked if ranked is not None else (None, None)

        def admitted(s):
            if ranks is not None and ranks.holds(s):
                return False
            return all(self._holds(f, s) for f in facts)

        rounds = sample_rounds(self.ctx, self.ob, self.ev, self.draws,
                               self.runs.states, admitted, want=100,
                               seconds=_SAMPLE_SECONDS / 4)
        for s, sp in rounds:
            if any(not self._holds(f, sp) for f in facts):
                return True
            if rank is not None:
                try:
                    before = self.ev(rank.term, dict(zip(self.ctx.names, s)))
                    after = self.ev(rank.term, dict(zip(self.ctx.names, sp)))
                except _Unevaluable:
                    continue
                if not _drops(before, after):
                    return True
        return False

    @staticmethod
    def _named(inv: list[Candidate], core) -> list[Candidate]:
        if core is None:
            return list(inv)
        return [f for i, f in enumerate(inv) if f"h{i}" in core]

    # --- reading the module and the property ------------------------------

    def _check_sorts(self, ctx: SynthContext) -> None:
        bad = [f"s{i} is {s}" for i, s in enumerate(ctx.env.state_sorts)
               if not (s.isInteger() or s.isBoolean())]
        bad += [f"{c} is {c.getSort()}"
                for c in list(ctx.extl_next) + list(ctx.extl_latched)
                if not (c.getSort().isInteger() or c.getSort().isBoolean())]
        if bad:
            raise Refused(
                f"--infer vampire reads Int and Bool state and inputs, and "
                f"this module has {', '.join(bad)}. Vampire's SMT-LIB front "
                f"end has no bitvector theory. `--infer smt-linear` weighs a "
                f"bitvector as its unsigned value."
            )

    def _conjuncts(self, ctx: SynthContext, cd: CertificateData) -> list[str]:
        """The property's top-level conjuncts, seeded as facts of their own.

        Split because Houdini keeps facts, not formulas: `(and a b)` is kept
        whole or dropped whole, where `a` alone may be inductive and `b`
        inductive given it.
        """
        p = ctx.prp
        parts = list(p) if p.getKind() == Kind.AND else [p]
        return [str(t) if len(parts) > 1 else cd.prp.strip() for t in parts]

    def _parse(self, srcs: list[str]) -> list[Candidate]:
        out = []
        for src in dict.fromkeys(srcs):
            try:
                term = self.ctx.env.parse_expr(src)
            except Exception:                        # noqa: BLE001
                continue
            if term.isNull():
                continue
            out.append(Candidate(src, term))
        return out

    # --- out --------------------------------------------------------------

    def _emit(self, cd: CertificateData, inv: list[Candidate],
              rank: "Candidate | None") -> CertificateData:
        srcs = [f.src for f in inv]
        inv_src = ("true" if not srcs else srcs[0] if len(srcs) == 1
                   else "(and " + " ".join(srcs) + ")")
        self.log(f"[vampire] inv: {inv_src}")
        cd.inv = cd.inv_smt = inv_src
        if rank is not None:
            self.log(f"[vampire] ranking: {rank.src}")
            cd.ranking = cd.ranking_smt = rank.src
        self._record(inv_src, rank)
        return cd

    def _record(self, inv_src: str, rank: "Candidate | None") -> None:
        if self.artifacts is None:
            return
        what = ("Houdini over facts read off simulated runs of the module, "
                "each proved by Vampire -- it holds at entry and every round "
                "preserves it -- and cut down to the facts the proofs used.")
        self.artifacts.put("inv", inv_src, status="proved", language="smt",
                           what=f"The invariant this run found. {what}")
        if rank is not None:
            self.artifacts.put(
                "ranking", rank.src, status="proved", language="smt",
                what="The ranking function this run found: Vampire proved "
                     "`hrank` for it over the invariant beside it.",
            )

    def _none(self, cd, n_facts, n_ranks, reached, prover) -> CertificateData:
        if cd.is_safety:
            what, article = "inductive invariant implying the property", "an"
            started = f"Houdini started from {n_facts} facts the simulated runs keep"
        else:
            what, article = "ranking function", "a"
            started = (f"Houdini started from {n_facts} facts the simulated "
                       f"runs keep, and at most {n_ranks} ranking functions "
                       f"dropped on every reached and sampled round")
        detail = (
            f"{started}; Vampire's time limit reached {reached:g} s over "
            f"{prover.calls} calls. Vampire cannot refute a candidate, so "
            f"this is not a proof that none exists: a fact or a shape outside "
            f"the candidates -- a disjunction, a relation of three "
            f"components, a constant the program never mentions -- or a "
            f"longer time limit may be what is missing."
        )
        if self.artifacts is not None:
            self.artifacts.note(
                f"--infer vampire found no {what}. {detail}\n",
                status="unknown",
                what=f"A search for {article} {what} that did not succeed.",
                why="Vampire proves or times out; nothing here was refuted.",
            )
        raise Refused(f"--infer vampire found no {what}. {detail}")
