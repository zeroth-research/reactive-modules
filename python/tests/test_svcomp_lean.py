"""Regression tests for the Lean emitter (``benchmarks.svcomp._lean``).

The structural tests run everywhere. The end-to-end test compiles an emitted
proof with ``lake`` and is skipped when no toolchain is present, so it protects
the pipeline on a machine that has one without breaking one that does not.
"""
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import z3

from tests._fixtures import loop_bench
from benchmarks.svcomp._lean import _render_conjuncts, _trivial, emit_program
from benchmarks.svcomp import discover
from benchmarks.svcomp._farkas import certify, inductive, lex_decrease, read_system
from benchmarks.svcomp._property import terminates
from benchmarks.svcomp._termination import _v_module, system_of
from zrth import Module
from benchmarks.svcomp._property import Safety
from benchmarks.svcomp._termination import build_candidate, farkas_cell
from zrth.sugar import ite, ne

LEAN_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "svcomp" / "lean"


def _decrement_obligation(invariants=()):
    """`while (x > 0) x = x - 1` with V(s) = relu(x), built as a real module: the
    composed system the pipeline reads, plus the result certifying it."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x))
    named = [(f"inv{k}", (lambda st, p=p: p)) for k, p in enumerate(invariants)]
    ob = build_candidate(bench, layers, 1.0, named)
    res = farkas_cell(ob)
    assert res.verified, res.status
    return ob.system, res.certificate


def test_emit_contains_the_proof_skeleton():
    system, res = _decrement_obligation()
    src = emit_program("decrement", system, res)
    for decl in ("def V_0 ", "theorem V_0_nonneg", "def invariants", "def trans",
                 "theorem covered", "def post_state", "def ok", "theorem step_ok",
                 "def Step", "theorem lex_step", "def Init", "theorem initiation",
                 "theorem consecution", "def RawStep", "theorem no_inf_step",
                 "theorem no_infinite_run", "no_infinite_run_lex"):
        assert decl in src, f"missing {decl!r}"
    assert "sorry" not in src


def test_trivial_cell_needs_no_certificate():
    """A cell whose support is a constant infeasibility refutes its disjunct
    unconditionally, so it is emitted with ``omega`` and carries no Farkas
    system."""
    system, res = _decrement_obligation()
    trivial = [i for i, c in enumerate(res.certificates[0].cells) if _trivial(c)]
    assert trivial, "expected an unconditionally refuted cell"
    src = emit_program("decrement", system, res)
    assert "constant infeasibility (no Farkas system)" in src
    assert ":= by\n  omega" in src
    for i in trivial:                         # one disjunct: region i is cell i
        assert f"cell{i}d0_A" not in src
        assert f"cell{i}d0_refute" in src


def test_every_device_collapses_on_its_region():
    """A region fixes the mode of every node, so each named wire has an exact
    collapse there and nothing is bounded: no relaxation lemma is emitted, and a
    region's signs are one definition naming every node."""
    system, res = _decrement_obligation()
    src = emit_program("decrement", system, res)
    assert "theorem c00_0" in src and "theorem c01_0" in src, \
        "both ends of the step should collapse on the region"
    assert "lb0_" not in src, "no relaxation bound should be emitted"
    assert re.search(r"def cell0_signs \(s : Vector 1 Int\) : Prop :=\n  signs_\d+ s\n",
                     src), "a region should be one sign definition"


def test_emit_without_invariants_uses_true():
    """With no inferred invariants the shape is unchanged; `invariants` is
    `True` and its lemmas close by `trivial`."""
    system, res = _decrement_obligation()
    src = emit_program("decrement", system, res)
    assert "def invariants (s : Vector 1 Int) : Prop :=\n  True" in src
    assert "trivial" in src


def test_emit_with_invariants_proves_them():
    """A real invariant is emitted as the shared definition and discharged by
    `initiation` / `consecution` rather than assumed."""
    x = z3.Int("x")
    system, res = _decrement_obligation(invariants=(x >= 0,))
    src = emit_program("decrement", system, res)
    assert "def invariants (s : Vector 1 Int) : Prop :=\n  True" not in src
    assert "theorem initiation" in src and "theorem consecution" in src
    # the termination theorem takes the entry state and its Init proof
    assert "theorem no_infinite_run (s0 : Vector 1 Int) (hinit : Init s0)" in src


def test_conditional_invariant_reaches_the_emitted_file():
    """An implication must reach the emitted `invariants` as a disjunction: the
    cells' Farkas rows rest on it, and `omega` closes them by case-splitting it."""
    x = z3.Int("x")
    inv = z3.Implies(x >= 1, x >= 0)
    system, res = _decrement_obligation(invariants=(inv,))
    src = emit_program("conditional", system, res)
    assert "def invariants (s : Vector 1 Int) : Prop :=\n  True" not in src, \
        "the implication was dropped from `invariants`"
    assert "∨" in src.split("def invariants")[1].split("\n\n")[0], \
        "an implication should render as a disjunction omega can split"


def test_branching_entry_fact_is_case_split_not_dropped():
    """An `ite` in an entry value is not linear, so the renderer must case-split
    its conjunct rather than drop it — else `initiation` loses its premise."""
    x = z3.Int("x")
    # branches must differ, or z3 folds the ite away before it is ever rendered, and
    # the condition must be over the state columns or the conjunct is not affine
    init_lean = _render_conjuncts(z3.And(x == z3.If(x >= 3, 5, 7), x >= 1), [x])
    assert "ite" not in init_lean and "If" not in init_lean, \
        f"ite left unexpanded: {init_lean}"
    assert "∨" in init_lean, f"the branch was dropped instead of case-split: {init_lean}"


def test_emit_multi_path_unions_the_step():
    """A branching body yields one namespace per path and a Step that is their
    union, dispatched in `no_infinite_run`."""
    layers = [(np.array([[1], [-1]]), np.array([0, 0])),
              (np.array([[1, 1]]), np.array([0]))]
    bench = loop_bench(("x",), lambda x: ite(ne(x, 0), ite(x > 0, x - 1, x + 1), x))
    ob = build_candidate(bench, layers, 1.0, [])
    res = farkas_cell(ob)
    assert res.verified, res.status
    assert len(res.certificate.certificates) >= 2, "a branching body should give several paths"
    src = emit_program("branching", ob.system, res.certificate)
    assert "namespace loop0_path0" in src and "namespace loop0_path1" in src
    assert "loop0_path0.Step a b ∨ loop0_path1.Step a b" in src
    assert "rintro a b (h | h)" in src


def test_non_trivial_cell_uses_its_certificate():
    """A cell whose refutation needs the guard carries a Farkas system, and the
    emitted proof reaches it through farkas_sound and refute_bridge."""
    layers = [(np.array([[2]]), np.array([-1])), (np.array([[1]]), np.array([0]))]
    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x))
    ob = build_candidate(bench, layers, 1.0, [])
    res = farkas_cell(ob)
    assert res.verified, res.status
    src = emit_program("nontrivial", ob.system, res.certificate)
    assert "farkas_sound" in src and "refute_bridge" in src


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_emitted_proof_kernel_checks(tmp_path):
    """End-to-end: the emitted file compiles against the vendored substrate."""
    x = z3.Int("x")
    system, res = _decrement_obligation(invariants=(x >= 0,))
    out = LEAN_DIR / "proofs" / "_test_decrement"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "program.lean"
    f.write_text(emit_program("decrement", system, res))
    try:
        r = subprocess.run(["lake", "env", "lean", str(f.relative_to(LEAN_DIR))],
                           cwd=LEAN_DIR, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0 and not r.stdout.strip(), r.stdout + r.stderr
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _always_obligation(pred, inv=None):
    """`while (x > 0) x = x - 1` from x = 0, with the safety property ``pred``
    proved by the invariant ``inv`` (default: ``pred`` itself)."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x))
    ob = build_candidate(bench, layers, 1.0, [])
    prop = Safety(pred)
    rule = inductive((inv or pred,))
    res = certify(ob.system, prop, rule)
    assert res.verified, res.status
    return ob.system, res


def test_always_emits_its_own_theorem():
    """A safety property emits no network and no ranking — the invariant is the
    device — and concludes ``always_holds`` rather than ``no_infinite_run``."""
    system, res = _always_obligation(lambda W, S: S["x"] >= 0)
    src = emit_program("safe", system, res)
    for decl in ("def pred", "def invariants", "theorem initiation",
                 "theorem consecution", "def ok", "theorem step_ok",
                 "theorem always_holds"):
        assert decl in src, f"missing {decl!r}"
    for absent in ("def V_", "nrf_", "lex_step", "no_infinite_run", "sorry"):
        assert absent not in src, f"unexpected {absent!r} in a safety proof"
    assert "(((1 * s 0)) ≥ 0)" in src or "≥ 0" in src.split("def pred")[1].split("\n\n")[0]


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_always_proof_kernel_checks(tmp_path):
    """End-to-end for the second property: the emitted safety proof compiles."""
    system, res = _always_obligation(lambda W, S: S["x"] >= 0)
    out = LEAN_DIR / "proofs" / "_test_always"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "program.lean"
    f.write_text(emit_program("safe", system, res))
    try:
        r = subprocess.run(["lake", "env", "lean", str(f.relative_to(LEAN_DIR))],
                           cwd=LEAN_DIR, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0 and not r.stdout.strip(), r.stdout + r.stderr
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _lex_obligation():
    """The nested loop ``while (i > 0) { if (j > 0) j--; else { i--; j = 3; } }``
    with the ranks ``relu(i)``, ``relu(j)`` each composed in at both ends of a
    step, certified lexicographically. Returns the composed system and the result
    carrying the rule and the ranks' networks."""
    prog = system_of(loop_bench(("i", "j"),
        lambda c: (ite(c[0] > 0, ite(c[1] > 0, c[0], c[0] - 1), c[0]),
                   ite(c[0] > 0, ite(c[1] > 0, c[1] - 1, 3), c[1]))))
    mods, ranks = [], []
    for W in ([[1, 0]], [[0, 1]]):
        layers = [(np.array(W), np.array([0])), (np.array([[1]]), np.array([0]))]
        vs_mod, vs = _v_module(prog.pairs, layers, read_next=False)
        vsp_mod, vsp = _v_module(prog.pairs, layers, read_next=True)
        mods += [vs_mod, vsp_mod]; ranks.append((vs[1], vsp[1]))
    system = read_system(Module.parallel(prog.module, *mods), prog.names)
    prop, rule = terminates(), lex_decrease(tuple(ranks))
    res = certify(system, prop, rule)
    assert res.verified, res.status
    return system, res


def test_lex_emits_one_network_per_rank():
    """Two ranks emit two networks and conclude through ``no_infinite_run_lex`` on
    the list of both, each path's ``lex_step`` proving ``lexDec`` directly."""
    system, res = _lex_obligation()
    src = emit_program("lex", system, res)
    for decl in ("def V_0 ", "def V_1 ", "def R0 ", "def R1 ",
                 "lexDec [R0, R1]", "no_infinite_run_lex [R0, R1]",
                 "theorem no_infinite_run", "cell0d0_refute", "cell0d1_refute"):
        assert decl in src, f"missing {decl!r}"
    assert "sorry" not in src


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_lex_proof_kernel_checks():
    """End-to-end for a lexicographic rank: the emitted proof compiles."""
    system, res = _lex_obligation()
    out = LEAN_DIR / "proofs" / "_test_lex"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "program.lean"
    f.write_text(emit_program("lex", system, res))
    try:
        r = subprocess.run(["lake", "env", "lean", str(f.relative_to(LEAN_DIR))],
                           cwd=LEAN_DIR, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0 and not r.stdout.strip(), r.stdout + r.stderr
    finally:
        shutil.rmtree(out, ignore_errors=True)


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_always_on_a_real_benchmark_kernel_checks():
    """A safety property of an SV-COMP program, end to end: ``ndecr`` enters with
    ``i = n - 1`` and only ever decrements ``i``, so ``i <= n`` always holds. Its
    own predicate is the inductive invariant; the proof compiles."""
    bench = next(b for b in discover() if b.name.endswith("ndecr"))
    system = system_of(bench)
    prop = Safety(lambda W, S: S["i"] <= S["n"])
    res = certify(system, prop, inductive((prop.holds,)))
    assert res.verified, res.status
    out = LEAN_DIR / "proofs" / "_test_always_ndecr"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "program.lean"
    f.write_text(emit_program("ndecr_safe", system, res))
    try:
        r = subprocess.run(["lake", "env", "lean", str(f.relative_to(LEAN_DIR))],
                           cwd=LEAN_DIR, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0 and not r.stdout.strip(), r.stdout + r.stderr
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _compiles(name: str, src: str) -> None:
    """Compile an emitted proof against the substrate and assert it kernel-checks."""
    out = LEAN_DIR / "proofs" / f"_test_{name}"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "program.lean"
    f.write_text(src)
    try:
        r = subprocess.run(["lake", "env", "lean", str(f.relative_to(LEAN_DIR))],
                           cwd=LEAN_DIR, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0 and not r.stdout.strip(), r.stdout + r.stderr
    finally:
        shutil.rmtree(out, ignore_errors=True)


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_disjunctive_invariant_proof_kernel_checks():
    """A disjunctive invariant reaches Lean as it is: ``omega`` case-splits it in
    ``consecution``, and ``step_ok`` closes from the per-disjunct refutations."""
    system, res = _always_obligation(
        lambda W, S: z3.Or(S["x"] >= 0, S["x"] <= -5))
    src = emit_program("disj", system, res)
    assert "∨" in src.split("def invariants")[1].split("\n\n")[0]
    _compiles("disj", src)


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_property_over_a_computed_wire_kernel_checks():
    """``while (i < n) i++`` beside ``d = relu(n - i)``, with the claim
    ``d == n - i`` under the invariant ``i <= n``: the safety proof carries the
    network, regions over ``d``'s pattern, and ``pred`` names ``V_0 s fzero``."""
    prog = system_of(loop_bench(("i", "n"),
                                lambda c: (ite(c[0] < c[1], c[0] + 1, c[0]), c[1]),
                                init=lambda: (0, 5)))
    layers = [(np.array([[-1, 1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    mod, out = _v_module(prog.pairs, layers, read_next=False)
    system = read_system(Module.parallel(prog.module, mod), prog.names)
    d = out[1]
    prop = Safety(lambda W, S: W[d] == S["n"] - S["i"])
    rule = inductive((lambda W, S: S["i"] <= S["n"],))
    res = certify(system, prop, rule)
    assert res.verified, res.status
    src = emit_program("wire", system, res)
    assert "V_0 s fzero" in src.split("def pred")[1].split("\n\n")[0]
    assert "theorem always_holds" in src
    _compiles("wire", src)


def test_the_proof_layer_refuses_a_safety_claim_over_the_step():
    """The engine certifies a safety claim relating a state to its successor, but
    the substrate has one composition for safety, over single states — so the
    proof layer refuses such a claim by name rather than emit a theorem of the
    wrong shape."""
    from benchmarks.svcomp._nodes import Unsupported
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x))
    ob = build_candidate(bench, layers, 1.0, [])
    res = certify(ob.system, Safety(lambda W, S: S.next["x"] <= S["x"]), inductive(()))
    assert res.verified, res.status
    with pytest.raises(Unsupported, match="over the step"):
        emit_program("step", ob.system, res)
