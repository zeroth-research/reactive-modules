"""TA2Magic by a learned ranking function, certified before it is offered.

``--infer learn``. Where ``--infer ai`` and ``--infer ai-cegar`` ask an LLM for
the certificate, this route learns one. A small ReLU network is trained on
rollouts of the module to drop on the rounds the property does not hold, its
weights are rounded to integers, and the candidate is put to a decision
procedure: Farkas-certified regions over the module's own wires, with CEGAR
finding the regions (:mod:`benchmarks.svcomp._farkas`). Only a candidate the
procedure certifies comes back, so what verith is handed is a certificate that
has already been proved -- ``--pre-check`` and ``lake build`` confirm it rather
than discovering it.

The invariant is inferred the same way and for the same reason: Houdini over a
candidate lattice of sign and pairwise facts, each survivor certified as one
inductive Safety claim. It is what the ranking obligation is allowed to assume,
and it is what verith's ``inv`` becomes, so the two are the same fact.

What each property asks for
===========================
``--buchi P`` -- ``G (F P)``. The rounds the rank must drop on are those where
``P`` does not hold, and that the run leaves them again and again is exactly
recurrence. This is the route's own shape: a rank is what it produces.

``--safety P`` -- ``G P``. There is no rank (``rule_globally`` takes none), so
the whole certificate is the invariant, and the property is seeded as one more
Houdini candidate: it is an invariant when it survives with the rest. That makes
this route an alternative to ``--fbk-proveit`` on a narrower class of programs --
Houdini's lattice is fixed, where ic3ia's interpolants are not -- and it needs
no ic3ia build.

What the route needs of a module
================================
The procedure reads scalar integer wires and a deterministic transition, and
refuses anything else by name -- a matrix-shaped state, a real or boolean
component, a next value reading an awaited input. A property naming a state
component the module never reads latched is refused too: such a component is not
a column of the system the proof quantifies over.
"""

from __future__ import annotations

from types import SimpleNamespace

import z3
from z3.z3util import get_vars

from zrth import Module

from .cert import CertificateData, smt_predicates_to_lean
from .common import Refused
from .magic import TA2Magic


def _engine():
    """The decision procedure and its trainer.

    Imported here rather than at module scope: the packages live beside ``zrth``
    rather than inside it, and a verith run that does not ask for this route
    should not need them present."""
    try:
        from benchmarks.svcomp._bench import Bench
        from benchmarks.svcomp._farkas import (certify, check_supported,
                                               inductive, read_system)
        from benchmarks.svcomp._invariants import as_predicates, infer_invariants
        from benchmarks.svcomp._nodes import Unsupported
        from benchmarks.svcomp._property import Liveness, Safety
        from benchmarks.svcomp._train import learn_ranking
    except ImportError as e:      # pragma: no cover - a broken checkout
        raise Refused(
            f"--infer learn needs the `benchmarks.svcomp` package, which is not "
            f"importable: {e}"
        ) from e
    return SimpleNamespace(
        Bench=Bench, certify=certify, check_supported=check_supported,
        inductive=inductive, read_system=read_system, as_predicates=as_predicates,
        infer_invariants=infer_invariants, Unsupported=Unsupported,
        Liveness=Liveness, Safety=Safety, learn_ranking=learn_ranking,
    )


# ---------------------------------------------------------------------------
# SMT-LIB in and out
# ---------------------------------------------------------------------------

def _smt_int(v) -> str:
    """An integer literal. SMT-LIB has no negative numerals, so one is a negation."""
    v = int(v)
    return str(v) if v >= 0 else f"(- {-v})"


def _smt_affine(coeffs, const, names) -> str:
    """``k + Σ cᵢ·sᵢ`` as an SMT-LIB term, written out in full.

    Printed here rather than through z3's ``sexpr``, which ``let``-binds the
    subterm a ReLU repeats -- and what this string has to survive is being read
    back and rendered as the body of a Lean definition."""
    parts = [] if not int(const) else [_smt_int(const)]
    for c, n in zip(coeffs, names):
        c = int(c)
        if c == 1:
            parts.append(n)
        elif c == -1:
            parts.append(f"(- {n})")
        elif c:
            parts.append(f"(* {_smt_int(c)} {n})")
    if not parts:
        return "0"
    return parts[0] if len(parts) == 1 else "(+ " + " ".join(parts) + ")"


def _smt_ranking(layers, names) -> str:
    """``V`` as an SMT-LIB Int term over ``names``: a ReLU as an ``ite``.

    ``layers`` is ``[(W1, b1), (W2, b2)]`` with integer entries, the same weights
    the composed module computed with, so the term and the wire the procedure
    certified are one function."""
    (W1, b1), (W2, b2) = layers
    parts = [] if not int(b2[0]) else [_smt_int(b2[0])]
    for j in range(W1.shape[0]):
        c = int(W2[0][j])
        if not c:
            continue
        pre = _smt_affine(W1[j], b1[j], names)
        unit = f"(ite (> {pre} 0) {pre} 0)"
        parts.append(unit if c == 1 else f"(* {_smt_int(c)} {unit})")
    if not parts:
        return "0"
    return parts[0] if len(parts) == 1 else "(+ " + " ".join(parts) + ")"


def _smt_conjunction(preds) -> str:
    """``preds`` -- z3 predicates over the columns -- as one SMT-LIB Bool term."""
    if not preds:
        return "true"
    # A seeded candidate the lattice also proposes survives as both, and the
    # same conjunct twice is noise in every file it reaches.
    rendered = list(dict.fromkeys(p.sexpr() for p in preds))
    lets = [r for r in rendered if "(let " in r]
    if lets:
        raise Refused(
            "an inferred invariant prints with a `let` binding, which the "
            f"certificate's definition cannot carry: {lets[0]}"
        )
    if len(rendered) == 1:
        return rendered[0]
    return "(and " + " ".join(rendered) + ")"


def _parse_property(src: str, declared: tuple, columns: tuple) -> object:
    """``src`` as a z3 predicate over ``z3.Int(name)``, refused if it is not one.

    The property reaches this route as SMT-LIB over ``s0..sN-1``, one per ctrl
    variable -- the same names the prompt routes declare -- so all of them are
    declared for the parse and the *columns* are what the result may name. The
    two differ by a component no term reads latched, which is not part of the
    state the proof quantifies over, and naming one has to say that rather than
    read as a typo."""
    decls = "\n".join(f"(declare-const {n} Int)" for n in declared)
    try:
        asserted = z3.parse_smt2_string(f"{decls}\n(assert {src})")
    except z3.Z3Exception as e:
        raise Refused(
            f"--infer learn reads the property as an SMT-LIB expression over "
            f"{', '.join(declared) or 'the state components'}; z3 could not "
            f"parse {src!r}: {e}"
        ) from e
    if len(asserted) != 1:
        raise Refused(f"the property must be one expression; {src!r} is {len(asserted)}")
    term = asserted[0]
    free = {str(v) for v in get_vars(term)} - set(columns)
    if free:
        raise Refused(
            f"the property names {sorted(free)}, which are not columns of this "
            f"module -- a component no term reads latched is not part of the "
            f"state the proof quantifies over"
        )
    return term


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------

class TA2MagicLearn(TA2Magic):
    """Infer the certificate by learning a ranking function and certifying it.

    ``delta`` is how much the rank must drop on a counting round (one, for an
    integer rank, is as weak as the obligation gets). ``seed`` makes the run
    reproducible: the same module and the same seed learn the same rank."""

    def __init__(
        self,
        source: str,
        module: Module,
        *,
        delta: int = 1,
        hidden_dim: int = 7,
        seed: int = 0,
        log=print,
    ):
        super().__init__(source)
        self.module = module
        self.delta = delta
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.log = log

    # --- the system the procedure reads --------------------------------

    def _system(self, eng):
        """The module as a :class:`System`, its columns named as verith names them.

        ``s0..sN-1`` are the module's ctrl variables in order, which is the
        naming the property and the emitted certificate are written in. The
        columns are the subset some term reads latched, so a variable that is
        only written keeps its name and simply is not one."""
        names = {v: f"s{i}" for i, v in enumerate(self.module.ctrl)}
        try:
            system = eng.read_system(self.module, names)
            eng.check_supported(system)
        except eng.Unsupported as e:
            raise Refused(f"--infer learn cannot read this module: {e}") from e
        return system

    def _declared(self) -> tuple:
        """``s0..sN-1``: one name per ctrl variable, which is how verith numbers
        the state and so how a property written against it reads."""
        return tuple(f"s{i}" for i, _ in enumerate(self.module.ctrl))

    def _bench(self, eng, system):
        """``system`` as a :class:`Bench`: what the trainer rolls out.

        The module is built once and handed back on every call -- nothing here
        mutates it, and a fresh copy would be a different set of wires from the
        one the system was read from."""
        ctrl = {f"s{i}": v for i, v in enumerate(self.module.ctrl)}
        extl = {f"e{i}": v for i, v in enumerate(self.module.extl)}
        return eng.Bench(
            name="verith", source="", state=tuple(system.names),
            inputs=tuple(extl), build=lambda: (self.module, ctrl, extl),
        )

    # --- driver ---------------------------------------------------------

    def infer(self, cd: CertificateData) -> CertificateData:
        eng = _engine()
        if not isinstance(cd.prp, str):
            raise Refused("--infer learn needs a property: pass --safety or --buchi")
        system = self._system(eng)
        prp = _parse_property(cd.prp, self._declared(), system.names)
        self.log(f"[learn] columns: {', '.join(system.names)}")

        if cd.is_safety:
            inv_smt = self._safety_invariant(eng, system, prp)
            rank_smt = None
        else:
            inv_smt, rank_smt = self._buchi_certificate(eng, system, prp)

        return self._emit(cd, inv_smt, rank_smt)

    # --- the two kinds --------------------------------------------------

    def _safety_invariant(self, eng, system, prp) -> str:
        """An inductive invariant implying ``prp``, or the reason there is none.

        The property is seeded as a candidate alongside the lattice's own, so
        Houdini keeps it exactly when it is inductive given everything else that
        survived -- and then the invariant implies it by containing it."""
        facts = eng.infer_invariants(
            system, extra=[("P", lambda st: self._at(prp, system, st))])
        self.log(f"[learn] invariant candidates kept: {[lbl for lbl, _ in facts]}")
        claim = eng.Safety(
            lambda W, S: self._at(prp, system, {n: S[n] for n in S.names}))
        proof = eng.certify(system, claim,
                            eng.inductive(eng.as_predicates(facts)))
        if not proof.verified:
            raise Refused(
                f"no inductive invariant implying the property was found "
                f"({proof.status}). Houdini's candidates are sign and pairwise "
                f"facts; for an invariant outside that lattice use "
                f"--fbk-proveit, or --infer ai-cegar."
            )
        self.log("[learn] safety invariant certified")
        # Houdini states a fact as `state_map -> BoolRef`, and the system's own
        # `s_map` is that map over the columns -- so the conjunct printed here
        # is the one the proof carried.
        return _smt_conjunction([f(system.s_map) for _, f in facts])

    def _buchi_certificate(self, eng, system, prp) -> tuple:
        """An invariant and a rank that drops wherever ``prp`` does not hold.

        The claim handed to the procedure is ``Liveness(not P)``: no infinite
        stretch of rounds where ``P`` is false, which is ``G (F P)``. The rank
        drops on exactly those rounds, which is verith's ranking obligation."""
        domain = (lambda W, S:
                  z3.Not(self._at(prp, system, {n: S[n] for n in S.names})))
        bench = self._bench(eng, system)
        self.log("[learn] training a ranking function")
        result = eng.learn_ranking(
            bench, delta=self.delta, hidden_dim=self.hidden_dim, seed=self.seed,
            claim=eng.Liveness(domain),
        )
        self.log(f"[learn] {result.n_pairs} sampled rounds, loss {result.final_loss:.4g}")
        if not result.verified:
            raise Refused(
                f"no ranking function was certified ({result.reason}). The rank "
                f"is learned from rollouts of the rounds where the property "
                f"fails, so a property that never fails on one, or a decrease no "
                f"piecewise-linear rank witnesses, leaves nothing to certify."
            )
        self.log("[learn] ranking function certified")
        invariants = result.system.invariants
        return (_smt_conjunction(list(invariants)),
                _smt_ranking(result.layers, result.system.names))

    # --- rendering ------------------------------------------------------

    @staticmethod
    def _at(term, system, st):
        """``term`` -- parsed over ``z3.Int("s_i")`` -- at the state map ``st``."""
        subs = [(z3.Int(n), st[n]) for n in system.names]
        return z3.substitute(term, *subs) if subs else term

    def _emit(self, cd: CertificateData, inv_smt: str, rank_smt: str | None):
        """Fill ``cd`` with the certificate, as Lean and as the SMT-LIB behind it.

        The project needs Lean; ``--pre-check`` needs cvc5's own input back, and
        nothing parses the Lean rendering into one -- so both are kept, as the
        other inferring routes keep them."""
        self.log(f"[learn] inv: {inv_smt}")
        if rank_smt is not None:
            self.log(f"[learn] ranking: {rank_smt}")
        lean = smt_predicates_to_lean(
            CertificateData(prp=cd.prp, kind=cd.kind, inv=inv_smt, ranking=rank_smt),
            self.module,
        )
        cd.inv, cd.inv_smt = lean.inv, inv_smt
        if rank_smt is not None:
            cd.ranking, cd.ranking_smt = lean.ranking, rank_smt
        return cd
