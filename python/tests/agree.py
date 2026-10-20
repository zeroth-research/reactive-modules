"""Random runs of a module, and what its encodings say about them.

The reference is `zrth.eval`: the interpreter `zrth.gym` runs a module with.
A run draws each round's awaited inputs at random; the rounds are
`execute_init` and then `execute_update`, with the next wires latched in
between, exactly as `zrth.gym` steps a module. An encoding is then asked
about each round *from the reference's own state*, so a disagreement is
reported at the round it happens in and does not compound.

`tests/test_encodings_agree.py` asks the SMT encoding and
`tests/lean/test_lean_agree.py` the Lean one, about the same runs.
"""

from __future__ import annotations

import random
from fractions import Fraction
from pathlib import Path

import torch

from zrth import Bool, BitVec, Int, Real, X
from zrth.eval import execute_init, execute_update
from zrth.lean.lowering.sorts import wire_shape

TESTS = Path(__file__).parent

RUNS = 6            # runs per module
ROUNDS = 5          # rounds per run after the initial one


class Skip(Exception):
    """The module is outside what one side can be asked: the reference
    cannot evaluate it (an uninterpreted or a nondeterministic op), or the
    encoding refuses it. Each says why."""


# ── the corpus ──────────────────────────────────────────────────────────────


def _corpus() -> dict:
    """Name -> a function building the module: loaded the way `verith`
    loads a Python input, so a fixture returning a `Named` module is the
    module it names."""
    from zrth.lean.inputs.python import load_module as _load

    out = {}
    for p in sorted((TESTS / "limits" / "mods").glob("m_*.py")):
        out[f"limits/{p.stem}"] = (lambda p=p: _load(p))
    for p in sorted((TESTS / "fixtures").glob("*.py")):
        if "def module(" in p.read_text():
            out[f"fixtures/{p.stem}"] = (lambda p=p: _load(p))
    try:
        from nuterm.tests.benchmarks.svcomp import discover
    except ImportError:                     # pragma: no cover - no nuterm
        return out
    for b in discover():
        out[f"svcomp/{b.name}"] = (lambda b=b: b.build()[0])
    return out


def _one_op(op, ins: list, out) -> object:
    """A module whose one variable is `op` of fresh inputs, every round: so
    each round asks the op alone, at random arguments."""
    from zrth import Module, Term, Var

    es = [Var(s) for s in ins]
    x = Var(out)
    block = [Term(op, [X(x)], [X(e) for e in es])]
    return Module.sequential([x, *es], block, list(block))


def _ops() -> dict:
    """One-op modules for every op the corpus never reaches but some
    encoding encodes, so that each encoded op is held to the interpreter.
    BV at width 4, where a random draw is negative as often as not."""
    from zrth import BV, LIA, LRA

    bv, bv1 = BitVec(4, [1, 1]), BitVec(1, [1, 1])
    real, boolean = Real([1, 1]), Bool([1, 1])
    out = {}
    for name in ("Add", "Sub", "Mul", "And", "Or", "Xor",
                 "UDiv", "SDiv", "UMod", "SMod"):
        out[f"BV.{name}"] = (getattr(BV, name)(), [bv, bv], bv)
    for name in ("Neg", "Not", "Id"):
        out[f"BV.{name}"] = (getattr(BV, name)(), [bv], bv)
    for name in ("Eq", "Ne", "ULt", "ULe", "UGt", "UGe",
                 "SLt", "SLe", "SGt", "SGe"):
        out[f"BV.{name}"] = (getattr(BV, name)(), [bv, bv], bv1)
    out["BV.BVToBool"] = (BV.BVToBool(), [bv], bv1)
    out["BV.MatMul"] = (BV.MatMul(), [BitVec(4, [1, 2]), BitVec(4, [2, 1])], bv)
    for name in ("And", "Or", "Xor"):
        out[f"LRA.{name}"] = (getattr(LRA, name)(), [boolean, boolean], boolean)
    out["LIA.Xor"] = (LIA.Xor(), [boolean, boolean], boolean)
    out["LRA.Not"] = (LRA.Not(), [boolean], boolean)
    for name in ("Ge", "Le", "Ne"):
        out[f"LRA.{name}"] = (getattr(LRA, name)(), [real, real], boolean)
    for name in ("Max", "Min"):
        out[f"LRA.{name}"] = (getattr(LRA, name)(), [Real([1, 3])], real)
    out["LRA.Linear"] = (LRA.Linear(torch.tensor([[0.5, -2.0]]),
                                    torch.tensor([[0.25]])),
                         [Real([2, 1])], real)
    return {f"ops/{k}": (lambda a=a: _one_op(*a)) for k, a in out.items()}


CORPUS = {**_corpus(), **_ops()}


# ── runs ────────────────────────────────────────────────────────────────────


def _draw(wire, rng: random.Random) -> torch.Tensor:
    """A random value for `wire`, shaped as `zrth.eval` keeps it."""
    dt, shape = wire.dtype, wire_shape(wire)
    if isinstance(dt, Bool):
        vals, kind = [rng.random() < 0.5 for _ in range(shape.total)], torch.bool
    elif isinstance(dt, Int):
        vals, kind = [rng.randint(-8, 8) for _ in range(shape.total)], torch.int64
    elif isinstance(dt, Real):
        vals = [rng.randint(-16, 16) / 4 for _ in range(shape.total)]
        kind = torch.float64
    elif isinstance(dt, BitVec):
        vals = [rng.randrange(1 << dt._0) for _ in range(shape.total)]
        kind = torch.int64
    else:                                   # pragma: no cover
        raise Skip(f"no random values for sort {dt}")
    return torch.tensor(vals, dtype=kind).reshape(shape.m, shape.n)


def runs(module, *, seed: int = 0) -> list:
    """`RUNS` runs of `module`: each a list of rounds `(inputs, state)`,
    where `inputs` maps each input variable to the value awaited that round
    and `state` lists the ctrl values after it, in `module.ctrl` order."""
    if len(list(module.atoms)) != 1:
        raise Skip("more than one atom; the encodings take one")
    rng = random.Random(seed)
    out = []
    for _ in range(RUNS):
        st: dict = {}
        run = []
        for k in range(ROUNDS + 1):
            inputs = {e: _draw(e, rng) for e in module.extl}
            if k:
                for c in module.ctrl:
                    st[c] = st[X(c)]
                for e in module.extl:
                    st[e] = st[X(e)]
            for e in module.extl:
                st[X(e)] = inputs[e]
            try:
                (execute_update if k else execute_init)(st, module.atoms)
            except RuntimeError as e:
                raise Skip(f"the interpreter cannot run it: {e}") from e
            run.append((inputs, [st[X(c)] for c in module.ctrl]))
        out.append(run)
    return out


def flat(wire, tensor) -> list:
    """`tensor` as `wire`'s elements, row-major, as plain Python values."""
    vals = tensor.reshape(-1).tolist()
    dt = wire.dtype
    if isinstance(dt, Bool):
        return [bool(v) for v in vals]
    if isinstance(dt, Real):
        return [Fraction(v) for v in vals]
    return vals


def same(wire, want: list, got: list) -> bool:
    """Whether `got` is `want`, elementwise. A Real is compared to within
    rounding, since the reference computes in floats."""
    if len(want) != len(got):
        return False
    if isinstance(wire.dtype, Real):
        tol = Fraction(1, 10**9)
        return all(abs(Fraction(a) - Fraction(b)) <= tol * (1 + abs(Fraction(a)))
                   for a, b in zip(want, got))
    # `True == 1` in Python; a Bool wire read as a number is a disagreement.
    return all(a == b and isinstance(a, bool) == isinstance(b, bool)
               for a, b in zip(want, got))


# ── the SMT encoding ────────────────────────────────────────────────────────


def _value(term) -> object:
    """A cvc5 value as a Python one; a tuple as its elements."""
    if term.isTupleValue():
        return [_value(x) for x in term.getTupleValue()]
    srt = term.getSort()
    if srt.isBoolean() and term.isBooleanValue():
        return term.getBooleanValue()
    if srt.isInteger() and term.isIntegerValue():
        return term.getIntegerValue()
    if srt.isReal() and term.isRealValue():
        return term.getRealValue()
    if srt.isBitVector() and term.isBitVectorValue():
        return int(term.getBitVectorValue(10))
    raise ValueError(f"not a value: {term}")


def disagreements_smt(module, *, seed: int = 0) -> list[str]:
    """Each round of `runs(module)` where the encoded round, evaluated by
    cvc5's rewriter at the same state and inputs, lands somewhere else."""
    from zrth.lean.smt.encoding import ModuleEncoding
    from zrth.lean.smt.encode import tensor_const

    rounds = runs(module, seed=seed)
    try:
        enc = ModuleEncoding(module)
        enc.init_state(enc.extl_next)
        enc.next_state(enc.state, enc.extl_latched, enc.extl_next)
    except (ValueError, RuntimeError) as e:
        raise Skip(f"no SMT encoding: {e}") from e
    tm = enc.tm

    def lits(wires, values):
        return [tensor_const(tm, w, v) for w, v in zip(wires, values)]

    bad = []
    for r, run in enumerate(rounds):
        before = None
        for k, (inputs, state) in enumerate(run):
            en = lits([X(e) for e in module.extl], [inputs[e] for e in module.extl])
            if k == 0:
                terms = enc.init_state(en)
            else:
                prev_inputs, prev_state = before
                s = lits(module.ctrl, prev_state)
                el = lits(module.extl, [prev_inputs[e] for e in module.extl])
                terms = enc.next_state(s, el, en)
            for c, want_t, term in zip(module.ctrl, state, terms):
                want = flat(c, want_t)
                try:
                    got = _value(enc.simplify(term))
                except ValueError as e:
                    bad.append(f"run {r} round {k}: {c.id} does not evaluate: {e}")
                    continue
                got = got if isinstance(got, list) else [got]
                if not same(c, want, got):
                    bad.append(f"run {r} round {k}: wire {c.id} ({c.dtype}) is "
                               f"{want} by the interpreter and {got} by cvc5")
            before = (inputs, state)
    return bad


# ── the Lean encoding ───────────────────────────────────────────────────────


def _lean_value(wire, tensor) -> str:
    """`tensor` as a Lean `Mat` literal of `wire`'s sort, in the form the
    encoding prints its own constants in
    (`lowering.elements.mat_from_scalars`)."""
    from zrth.lean.lowering.elements import mat_from_scalars
    from zrth.lean.codegen.literals import lean_literal
    from zrth.lean.lowering.sorts import flat_element_type

    elems = [lean_literal(wire.dtype, v) for v in flat(wire, tensor)]
    return mat_from_scalars(elems, wire_shape(wire), flat_element_type(wire))


def _lean_shown(wire, element: str) -> str:
    """`element` of `wire`'s sort, printed as `flat` prints the reference:
    a BitVec as the natural number it holds."""
    if isinstance(wire.dtype, BitVec):
        return f"{{toString ({element}).toNat}}"
    return f"{{toString ({element})}}"


def lean_script(module, rounds: list, ns: str) -> "tuple[str, dict[str, str]]":
    """Lean that `#eval`s `module`'s functional encoding on `rounds`, and
    the line each `#eval` should print.

    Each round is a `def` in namespace `ns` -- `init` at the round's awaited
    inputs, or `update` at the reference's state before it -- and one
    printed line, `<tag>|<wire>|<wire>...`, each wire its elements
    row-major. Raises `Skip` for a module Lean cannot run: a Real is
    noncomputable, and a module with no Lean form has nothing to run.
    """
    from zrth.lean.lowering.elements import product_accessor
    from zrth.lean.codegen.context import LeanContext
    from zrth.lean.codegen.terms import build_tuple, lean_gaps
    from zrth.lean.codegen.encodings import ModuleToLean4

    wires = list(module.ctrl) + list(module.extl)
    if any(isinstance(w.dtype, Real) for w in wires):
        raise Skip("a Real is noncomputable in Lean, so there is nothing to #eval")
    ctx = LeanContext(module)
    gaps = lean_gaps(ctx)
    if gaps:
        raise Skip(f"no Lean form: {gaps[0]}")

    ctrl, extl = list(module.ctrl), list(module.extl)
    lines = [f"namespace {ns}", ModuleToLean4(ctx).to_lean_functional()]
    evals, want = [], {}
    for r, run in enumerate(rounds):
        before = None
        for k, (inputs, state) in enumerate(run):
            en = build_tuple([_lean_value(e, inputs[e]) for e in extl])
            if k == 0:
                call = f"init {en}"
            else:
                prev_inputs, prev_state = before
                s = build_tuple([_lean_value(c, v) for c, v in zip(ctrl, prev_state)])
                el = build_tuple([_lean_value(e, prev_inputs[e]) for e in extl])
                call = f"update {s} {el} {en}"
            tag = f"{ns} r{r} k{k}"
            name = f"r{r}k{k}"
            lines.append(f"def {name} := {call}")
            shown, expected = [], []
            for p, c in enumerate(ctrl):
                acc = product_accessor(p, len(ctrl))
                shape = wire_shape(c)
                shown.append(" ".join(
                    _lean_shown(c, f"{ns}.{name}{acc} {i} {j}")
                    for i in range(shape.m) for j in range(shape.n)))
                expected.append(" ".join(
                    str(v).lower() if isinstance(v, bool) else str(int(v))
                    for v in flat(c, state[p])))
            evals.append(f'#eval IO.println s!"{tag}|{"|".join(shown)}"')
            want[tag] = f"{tag}|{'|'.join(expected)}"
            before = (inputs, state)
    lines.append(f"end {ns}")
    return "\n".join(["import Core.Box", "", *lines, "", *evals, ""]), want
