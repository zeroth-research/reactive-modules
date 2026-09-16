"""Regression tests for the Lean emitter (``benchmarks.svcomp._lean``).

The structural tests run everywhere. The end-to-end test compiles an emitted
proof with ``lake`` and is skipped when no toolchain is present, so it protects
the pipeline on a machine that has one without breaking one that does not.
"""
import dataclasses
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import z3

from tests._fixtures import candidate, loop_bench
from benchmarks.svcomp._lean import (_render_conjuncts, _term_str, _trivial,
                                     emit_program)
from benchmarks.svcomp import discover
from benchmarks.svcomp._farkas import (Unsupported, certify, inductive, lex_decrease,
                                       read_system)
from benchmarks.svcomp._termination import terminates
from benchmarks.svcomp._termination import _v_module, compose, system_of
from zrth import Module
from benchmarks.svcomp._property import Safety
from zrth.sugar import ite, ne

LEAN_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "svcomp" / "lean"


def _decrement_obligation(invariants=()):
    """`while (x > 0) x = x - 1` with V(s) = relu(x), built as a real module: the
    composed system the pipeline reads, plus the result certifying it."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x))
    named = [(f"inv{k}", (lambda st, p=p: p)) for k, p in enumerate(invariants)]
    ob = candidate(bench, layers, 1.0, named)
    res = certify(ob.system, ob.claim, ob.witness)
    assert res.verified, res.status
    return ob.system, res


def test_emit_contains_the_proof_skeleton():
    system, res = _decrement_obligation()
    src = emit_program("decrement", system, res)
    for decl in ("def V_0 ", "theorem V_0_nonneg", "def update", "def init_pre", "def RM",
                 "namespace claim0", "def invariants", "def domain", "def trans",
                 "theorem covered", "def post_state", "def ok", "theorem step_ok",
                 "def Step", "theorem lex_step", "theorem refines", "theorem some_path",
                 "theorem hrank", "theorem holds", "rule_buchi_lex"):
        assert decl in src, f"missing {decl!r}"
    assert "sorry" not in src


def test_the_module_is_rendered_exactly():
    """``update`` is the transition as read: one arm per column, the branch an
    ``if`` on the guard's linear condition, each leaf the affine body — so what
    the kernel reasons about is the module, not a summary of it."""
    system, res = _decrement_obligation()
    src = emit_program("decrement", system, res)
    assert ("def update (s : Vector 1 Int) : Vector 1 Int := fun i =>\n"
            "  match i with\n"
            "    | fzero => (if ((-1 * s fzero) < 0) then ((1 * s fzero) + -1) "
            "else (1 * s fzero))") in src, src.split("def update")[1][:300]
    assert ("def RM : ReactiveModule (Vector 1 Int) (Vector 1 Int) :=\n"
            "  { init := fun e => e, update := fun s _ => update s,\n"
            "    init_pre := init_pre, update_pre := fun _ => True }") in src


def test_a_nested_branch_renders_as_a_nested_if():
    """A body with two levels of branching renders as a nested ``if``, and the
    ``refines`` lemma splits as many times as the column has branches."""
    layers = [(np.array([[1], [-1]]), np.array([0, 0])),
              (np.array([[1, 1]]), np.array([0]))]
    bench = loop_bench(("x",), lambda x: ite(ne(x, 0), ite(x > 0, x - 1, x + 1), x))
    ob = candidate(bench, layers)
    res = certify(ob.system, ob.claim, ob.witness)
    assert res.verified, res.status
    src = emit_program("nested", ob.system, res)
    upd = src.split("def update")[1].split("\n\n")[0]
    assert upd.count("if ") == 2, upd
    assert src.count("(try split) <;> ") == 2 * len(res.certificates), \
        "each column's branches must be split in every path's refines"


def test_the_module_is_rendered_exactly_or_not_at_all():
    """A transition this layer cannot render exactly is refused by name: unlike
    a guard or an invariant, a weakened ``update`` would not be the module."""
    x, y = z3.Ints("x y")
    assert _term_str(z3.If(x >= 0, x - 1, x), [x]) == \
        "(if ((1 * s fzero) ≥ 0) then ((1 * s fzero) + -1) else (1 * s fzero))"
    with pytest.raises(Unsupported, match="not affine"):
        _term_str(x * y, [x, y])
    with pytest.raises(Unsupported, match="condition is not linear"):
        _term_str(z3.If(x * y >= 0, x, y), [x, y])


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
    assert ("theorem invariant : RM.StateSet_isInvariant invariants :=\n"
            "  fun _ _ => trivial") in src


def test_emit_with_invariants_proves_them():
    """A real invariant arrives as a proved Safety claim: it is emitted first as
    ``claim0`` with its own ``own_initial`` / ``own_step`` / ``consecution``, and
    the liveness claim cites its ``pred_invariant`` rather than re-deriving it."""
    x = z3.Int("x")
    system, res = _decrement_obligation(invariants=(x >= 0,))
    src = emit_program("decrement", system, res)
    assert "namespace claim0" in src and "namespace claim1" in src
    assert "theorem own_step" in src and "theorem consecution" in src
    assert "claim0.pred_invariant" in src
    assert ("theorem holds : ∀ ss, RM.traces ss →\n"
            "    ss ⊧ G (F (LTLFormula.Not (LTLFormula.APₛ domain)))") in src


def test_conditional_invariant_reaches_the_emitted_file():
    """An implication must reach the emitted `invariants` as a disjunction: the
    cells' Farkas rows rest on it, and `omega` closes them by case-splitting it."""
    x = z3.Int("x")
    inv = z3.Implies(x >= 1, x >= 0)
    system, res = _decrement_obligation(invariants=(inv,))
    src = emit_program("conditional", system, res)
    own = src.split("namespace claim0")[1].split("def own")[1].split("\n\n")[0]
    assert own.strip() != ": Prop :=\n  True", "the implication was dropped from the invariant"
    assert "∨" in own, "an implication should render as a disjunction omega can split"


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


def test_emit_multi_path_covers_the_domain():
    """A branching body yields one namespace per path; ``some_path`` says they
    cover the module's rounds inside the domain, and ``hrank`` dispatches on it."""
    layers = [(np.array([[1], [-1]]), np.array([0, 0])),
              (np.array([[1, 1]]), np.array([0]))]
    bench = loop_bench(("x",), lambda x: ite(ne(x, 0), ite(x > 0, x - 1, x + 1), x))
    ob = candidate(bench, layers)
    res = certify(ob.system, ob.claim, ob.witness)
    assert res.verified, res.status
    assert len(res.certificates) >= 2, "a branching body should give several paths"
    src = emit_program("branching", ob.system, res)
    assert "namespace path0" in src and "namespace path1" in src
    assert "path0.trans s ∨ path1.trans s" in src
    assert "rcases some_path s hI hd with hg | hg" in src


def test_non_trivial_cell_uses_its_certificate():
    """A cell whose refutation needs the guard carries a Farkas system, and the
    emitted proof reaches it through farkas_sound and refute_bridge."""
    layers = [(np.array([[2]]), np.array([-1])), (np.array([[1]]), np.array([0]))]
    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x))
    ob = candidate(bench, layers)
    res = certify(ob.system, ob.claim, ob.witness)
    assert res.verified, res.status
    src = emit_program("nontrivial", ob.system, res)
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
    ob = candidate(bench, layers)
    prop = Safety(pred)
    rule = inductive((inv or pred,))
    res = certify(ob.system, prop, rule)
    assert res.verified, res.status
    return ob.system, res


def test_always_emits_its_own_theorem():
    """A safety property emits no network and no ranking — the invariant is the
    device — and concludes ``G (AP pred)`` through ``rule_globally``."""
    system, res = _always_obligation(lambda W, S: S["x"] >= 0)
    src = emit_program("safe", system, res)
    for decl in ("def pred", "def own", "def assumed", "def invariants",
                 "theorem own_initial", "theorem own_step", "theorem consecution",
                 "def ok", "theorem step_ok", "theorem pred_invariant",
                 "rule_globally pred", "G (LTLFormula.AP pred)", "theorem holds"):
        assert decl in src, f"missing {decl!r}"
    for absent in ("def V_", "nrf_", "lex_step", "rule_buchi", "sorry"):
        assert absent not in src, f"unexpected {absent!r} in a safety proof"
    assert "≥ 0" in src.split("def pred")[1].split("\n\n")[0]


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
    """Two ranks emit two networks and conclude through ``rule_buchi_lex`` on
    the list of both, each path's ``lex_step`` proving ``lexDec`` directly."""
    system, res = _lex_obligation()
    src = emit_program("lex", system, res)
    for decl in ("def V_0 ", "def V_1 ", "def R0 ", "def R1 ",
                 "lexDec [R0, R1]", "rule_buchi_lex domain invariants invariant [R0, R1]",
                 "theorem holds", "cell0d0_refute", "cell0d1_refute"):
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


def test_an_exported_project_carries_everything_it_imports():
    """``export_project`` writes a directory that stands alone: the proof, a copy
    of every substrate library the proof imports, a lakefile naming them, and the
    toolchain file."""
    import tempfile
    from benchmarks.svcomp import _lean_check as lc

    system, res = _decrement_obligation()
    with tempfile.TemporaryDirectory() as tmp:
        out = lc.export_project("decrement", system, res, Path(tmp))
        names = {p.name for p in out.iterdir()}
        assert {"Program.lean", "lakefile.toml", "lean-toolchain"} <= names
        assert {f"{lib}.lean" for lib in lc.SUBSTRATE} <= names
        src = (out / "Program.lean").read_text()
        imports = {l.split()[1] for l in src.splitlines() if l.startswith("import ")}
        assert imports <= set(lc.SUBSTRATE), f"imports nothing carries: {imports}"
        lakefile = (out / "lakefile.toml").read_text()
        for lib in lc.SUBSTRATE + ("Program",):
            assert f'name = "{lib}"' in lakefile, lib


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_an_exported_project_builds_on_its_own(tmp_path):
    """End-to-end: the exported project compiles from scratch, outside this
    package, with nothing but the toolchain."""
    from benchmarks.svcomp import _lean_check as lc

    system, res = _decrement_obligation()
    out = lc.export_project("decrement", system, res, tmp_path)
    outcome, detail = lc.build_project(out)
    assert outcome == "CHECKED", detail


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
    system, res = _computed_wire_obligation()
    _compiles("wire", emit_program("wire", system, res))


def _computed_wire_obligation():
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
    return system, res


def test_a_claim_naming_a_computed_wire_states_it_as_that_network():
    """A claim may name a computed wire, and the statement says so: ``pred``
    renders the wire as the network applied to the state."""
    system, res = _computed_wire_obligation()
    src = emit_program("wire", system, res)
    assert "V_0 s fzero" in src.split("def pred")[1].split("\n\n")[0]
    assert "G (LTLFormula.AP pred)" in src


def _step_claim():
    """``while (x > 0) x--`` with the step claim ``x' <= x``, certified outright."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x))
    ob = candidate(bench, layers)
    res = certify(ob.system, Safety(lambda W, S: S.next["x"] <= S["x"]), inductive(()))
    assert res.verified, res.status
    return ob.system, res


def test_a_safety_claim_over_the_step_is_stated_with_a_step_atom():
    """A safety claim relating a state to its successor is a claim like any
    other: ``pred`` takes both states, the statement is ``G (APₛ pred)``, and
    ``rule_globally_step`` closes it from ``pred_holds`` at ``update s``."""
    system, res = _step_claim()
    src = emit_program("step", system, res)
    assert "def pred (s s' : Vector 1 Int) : Prop" in src
    assert "ss ⊧ G (LTLFormula.APₛ pred)" in src and "rule_globally_step" in src
    assert "pred s (update s)" in src


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_safety_claim_over_the_step_kernel_checks():
    system, res = _step_claim()
    _compiles("step", emit_program("step", system, res))


def test_the_module_is_emitted_once_from_the_system():
    """``update``, ``init_pre`` and ``RM`` come from the system alone: two
    different claims about the same module emit the module block verbatim, and
    each claim's statement is decided by its kind — ``G (F (Not (APₛ domain)))``
    for a Liveness claim, ``G (AP pred)`` for a Safety claim."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x))
    ob = candidate(bench, layers)
    live = certify(ob.system, ob.claim, ob.witness)
    nonneg = lambda W, S: S["x"] >= 0
    safe = certify(ob.system, Safety(nonneg), inductive((nonneg,)))
    assert live.verified and safe.verified
    src_live, src_safe = emit_program("a", ob.system, live), emit_program("a", ob.system, safe)

    def module_block(src):
        return src.split("def update")[1].split("namespace claim0")[0]
    assert module_block(src_live) == module_block(src_safe)
    assert "ss ⊧ G (F (LTLFormula.Not (LTLFormula.APₛ domain)))" in src_live
    assert "ss ⊧ G (LTLFormula.AP pred)" in src_safe


def test_the_label_is_the_clients_and_stays_in_the_header():
    """Nothing in the general layer says "terminates": the word enters only
    through the client's ``label``, and only into the header comment."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x))
    ob = candidate(bench, layers)
    live = certify(ob.system, ob.claim, ob.witness)
    plain = emit_program("a", ob.system, live)
    assert "terminat" not in plain.lower()
    labelled = emit_program("a", ob.system, live, label="terminates via a ranking function")
    header, rest = labelled.split("\n\n", 1)
    assert "terminates via a ranking function" in header
    assert "terminat" not in rest.lower()


def _countdown():
    """``while (x != 0) x--`` from ``x = 5``, and two facts proved about it on
    the bare system: ``x >= 0`` and ``x <= 5``."""
    bench = loop_bench(("x",), lambda x: ite(ne(x, 0), x - 1, x), init=lambda: (5,))
    system = system_of(bench)
    nonneg = lambda W, S: S["x"] >= 0
    atmost = lambda W, S: S["x"] <= 5
    p0 = certify(system, Safety(nonneg), inductive((nonneg,)))
    p1 = certify(system, Safety(atmost), inductive((atmost,)))
    assert p0.verified and p1.verified
    return system, p0, p1


def _both_sides():
    """A claim that assumes a proved claim AND carries an invariant of its own:
    ``x <= 5`` under the assumed ``x >= 0``."""
    system, p0, _ = _countdown()
    known = system.knowing(p0)
    atmost = lambda W, S: S["x"] <= 5
    res = certify(known, Safety(atmost), inductive((atmost,)))
    assert res.verified, res.status
    return known, res


def test_an_own_invariant_is_proved_relative_to_what_is_assumed():
    """Assume-guarantee: the claim's own invariant need only be inductive
    *given* what the assumed claims establish, so ``own`` and ``assumed`` are
    separate definitions and only ``own`` is carried by ``own_step``."""
    known, res = _both_sides()
    src = emit_program("both", known, res)
    main = src.split("end claim0")[-1]
    assert "def own (s : Vector 1 Int) : Prop :=\n  True" not in main, "own should be real"
    assert "def assumed (s : Vector 1 Int) : Prop :=\n  True" not in main, "assumed should be real"
    assert "theorem own_step" in main and "claim0.pred_invariant" in main
    assert ("RM.StateSet_isInvariant_relative assumed own assumed_invariant\n"
            "    own_initial own_step") in main


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_an_own_invariant_relative_to_an_assumed_one_kernel_checks():
    known, res = _both_sides()
    _compiles("both", emit_program("both", known, res))


def _two_assumed():
    """A claim assuming two proved claims and carrying no invariant of its own."""
    system, p0, p1 = _countdown()
    known = system.knowing(p0, p1)
    res = certify(known, Safety(lambda W, S: S["x"] >= -1), inductive(()))
    assert res.verified, res.status
    return known, res


def test_several_assumed_claims_are_each_a_namespace_and_each_cited():
    """Two assumed proofs give two namespaces, and the claim that assumes them
    takes one hypothesis per claim and cites each one's ``pred_invariant``."""
    known, res = _two_assumed()
    src = emit_program("two", known, res)
    for ns in ("claim0", "claim1", "claim2"):
        assert f"namespace {ns}" in src, ns
    main = src.split("end claim1")[-1]
    assert "(h0 : claim0.pred s) (h1 : claim1.pred s)" in main
    assert "(claim0.pred_invariant s hr) (claim1.pred_invariant s hr)" in main


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_several_assumed_claims_kernel_check():
    known, res = _two_assumed()
    _compiles("two", emit_program("two", known, res))


def test_a_rule_per_witness_is_the_whole_seam():
    """What the proof layer knows about a witness is one rule, and the registry
    is where a witness is paired with it: each rule names the claim kind it
    concludes, and the witness for it refuses every other kind itself."""
    from benchmarks.svcomp._lean import RULES
    from benchmarks.svcomp._farkas import Inductive, LexDecrease
    from benchmarks.svcomp._property import Liveness, Safety

    assert set(RULES) == {Inductive, LexDecrease}
    assert RULES[Inductive].discharges is Safety
    assert RULES[LexDecrease].discharges is Liveness
    for witness, rule in RULES.items():
        for hook in ("discharges", "invariant", "prelude", "evidence", "state", "compose"):
            assert hasattr(rule, hook), f"{rule.__name__} has no {hook}"
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x))
    ob = candidate(bench, layers)
    # each witness refuses the claim kind its rule does not conclude
    with pytest.raises(Unsupported):
        certify(ob.system, Safety(lambda W, S: S["x"] >= 0), ob.witness)
    with pytest.raises(Unsupported):
        certify(ob.system, ob.claim, inductive(()))


def test_the_proof_layer_refuses_what_it_has_no_rule_for():
    """The statement comes from the claim and the proof from the witness, so a
    witness this layer has no rule for is refused by name rather than emitted
    under another witness's rule."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x))
    ob = candidate(bench, layers)
    res = certify(ob.system, ob.claim, ob.witness)
    assert res.verified, res.status
    with pytest.raises(Unsupported, match="no rule for a"):
        emit_program("odd", ob.system, dataclasses.replace(res, witness=object()))


def test_the_proof_layer_refuses_an_assumption_it_cannot_state():
    """What a proof assumed is part of what it proved, so the file must state
    the same assumptions the engine certified under: an assumed claim that
    itself assumed something, or a proof certified against a different set, is
    refused by name."""
    system, p0, _ = _countdown()
    known, res = _both_sides()
    # `res` assumed p0; emitting it against the bare system would state its
    # invariant over facts its paths never had
    with pytest.raises(Unsupported, match="the system carries 0"):
        emit_program("bare", system, res)
    # and an assumed claim that itself assumed something has no proof to cite
    chained = system.knowing(p0, res)
    main = certify(chained, Safety(lambda W, S: S["x"] >= -1), inductive(()))
    assert main.verified, main.status
    with pytest.raises(Unsupported, match="itself assumed"):
        emit_program("chained", chained, main)


def test_the_proof_layer_refuses_a_claim_naming_a_device_on_the_next_state():
    """A network read at the successor is emitted per path, not as a definition
    over ``s``, so a claim naming that wire is refused by name rather than
    stated as the network at the wrong state."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    bench = loop_bench(("x",), lambda x: ite(x > 0, x - 1, x))
    ob = candidate(bench, layers)
    v_sp = ob.witness.ranks[0][1]
    res = certify(ob.system, Safety(lambda W, S: W[v_sp] >= 0), inductive(()))
    assert res.verified, res.status
    with pytest.raises(Unsupported, match="reads the next"):
        emit_program("nextdev", ob.system, res)


def _assuming_safety():
    """``x`` counts down from 5 and stops at 0; ``x >= 0`` is proved and assumed,
    and the weaker ``x >= -1`` is then claimed with no invariant of its own."""
    bench = loop_bench(("x",), lambda x: ite(ne(x, 0), x - 1, x), init=lambda: (5,))
    system = system_of(bench)
    nonneg = lambda W, S: S["x"] >= 0
    proof = certify(system, Safety(nonneg), inductive((nonneg,)))
    known = system.knowing(proof)
    weaker = certify(known, Safety(lambda W, S: S["x"] >= -1), inductive(()))
    assert weaker.verified, weaker.status
    return known, weaker


def test_a_safety_claim_assumes_a_proved_safety_claim():
    """A Safety claim may assume earlier Safety proofs like a Liveness claim
    does: its own invariant is proved relative to what is assumed
    (``StateSet_isInvariant_relative``), the assumed part is cited, never
    re-proved."""
    known, weaker = _assuming_safety()
    src = emit_program("assume_safe", known, weaker)
    assert "namespace claim0" in src and "namespace claim1" in src
    main = src.split("end claim0")[-1]
    assert "claim0.pred_invariant" in main and "StateSet_isInvariant_relative" in main
    assert "theorem own_step" in main
    assert "def own (s : Vector 1 Int) : Prop :=\n  True" in main


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_safety_claim_assuming_a_safety_claim_kernel_checks():
    known, weaker = _assuming_safety()
    _compiles("assume_safe", emit_program("assume_safe", known, weaker))


def test_the_liveness_theorem_cites_the_invariant_proof():
    """With an invariant assumed as a proved Safety claim, the file carries both
    claims: a ``claim0`` namespace proving its ``holds`` and ``pred_invariant``,
    and a liveness claim that cites ``pred_invariant`` for the invariant instead
    of re-deriving it — so no ``consecution`` remains outside ``claim0``. And the
    two-claim file kernel-checks."""
    bench = loop_bench(("x",), lambda x: ite(ne(x, 0), x - 1, x), init=lambda: (5,))
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    system = system_of(bench)
    nonneg = lambda W, S: S["x"] >= 0
    proof = certify(system, Safety(nonneg), inductive((nonneg,)))
    composed, witness = compose(system.knowing(proof), layers)
    live = certify(composed, terminates(), witness)
    assert live.verified, live.status
    src = emit_program("assumed", composed, live)
    assert "namespace claim0" in src and "claim0.pred_invariant" in src
    outside = src.split("namespace claim0")[0] + src.split("end claim0")[-1]
    assert "theorem consecution" not in outside
    _compiles("assumed", src)
