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

from benchmarks.svcomp._farkas import Net, certify_decrease
from benchmarks.svcomp._lean import _trivial, emit_program
from benchmarks.svcomp._verify_ranking import Obligation

LEAN_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "svcomp" / "lean"


def _decrement_obligation(invariants=(), init=None):
    """`while (x > 0) x = x - 1` with V(s) = relu(x), as an Obligation plus its
    certified paths."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    x = z3.Int("x")
    s, sp = [x], [z3.If(x > 0, x - 1, x)]
    res = certify_decrease(Net.from_layers(layers), s, sp, x > 0, invariants, 1.0)
    assert res.verified, res.status
    ob = Obligation(("x",), s, sp, None, None, 1.0, x > 0,
                    invariants=invariants, layers=layers, init=init)
    return ob, res.certificates


def test_emit_contains_the_proof_skeleton():
    ob, paths = _decrement_obligation()
    src = emit_program("decrement", ob, paths)
    for decl in ("def V ", "theorem V_nonneg", "def invariants", "def trans",
                 "theorem covered", "def post_state", "def Step", "theorem lex_step",
                 "def Init", "theorem initiation", "theorem consecution",
                 "def RawStep", "theorem no_inf_step", "theorem program_terminates",
                 "no_infinite_run_lex"):
        assert decl in src, f"missing {decl!r}"
    assert "sorry" not in src


def test_trivial_cell_needs_no_certificate():
    """A cell whose support is a constant infeasibility decreases unconditionally,
    so it is emitted with ``omega`` and carries no Farkas system."""
    ob, paths = _decrement_obligation()
    trivial = [i for i, c in enumerate(paths[0].cells) if _trivial(c)]
    assert trivial, "expected an unconditionally decreasing cell"
    src = emit_program("decrement", ob, paths)
    assert "unconditional decrease (no Farkas system)" in src
    assert "by omega" in src
    for i in trivial:
        assert f"cell{i}_A" not in src
        assert f"cell{i}_decrease" in src


def test_pre_bound_needs_no_hypothesis():
    """The pre-state lemma is an inequality taking only the state — no sign
    hypothesis — which is the asymmetry the scheme rests on, and a cell's signs
    name the successor pattern alone."""
    ob, paths = _decrement_obligation()
    src = emit_program("decrement", ob, paths)
    assert re.search(r"theorem pre_lb_\d+ \(s : Vector 1 Int\) :\n", src), \
        "pre_lb should take the state and nothing else"
    assert "pre_c0" not in src, "the pre-state collapse lemma should be gone"
    assert re.search(r"def cell0_signs \(s : Vector 1 Int\) : Prop :=\n  post_signs_\d+ s\n",
                     src), "an un-narrowed cell should constrain the successor only"


def test_emit_without_invariants_uses_true():
    """With no inferred invariants the shape is unchanged; `invariants` is
    `True` and its lemmas close by `trivial`."""
    ob, paths = _decrement_obligation()
    src = emit_program("decrement", ob, paths)
    assert "def invariants (s : Vector 1 Int) : Prop :=\n  True" in src
    assert "trivial" in src


def test_emit_with_invariants_proves_them():
    """A real invariant is emitted as the shared definition and discharged by
    `initiation` / `consecution` rather than assumed."""
    x = z3.Int("x")
    ob, paths = _decrement_obligation(invariants=(x >= 0,), init=(x >= 0))
    src = emit_program("decrement", ob, paths)
    assert "def invariants (s : Vector 1 Int) : Prop :=\n  True" not in src
    assert "theorem initiation" in src and "theorem consecution" in src
    # the termination theorem takes the entry state and its Init proof
    assert "theorem program_terminates (s0 : Vector 1 Int) (hinit : Init s0)" in src


def test_conditional_invariant_reaches_the_emitted_file():
    """An implication must reach the emitted `invariants` as a disjunction: the
    cells' Farkas rows rest on it, and `omega` closes them by case-splitting it."""
    x = z3.Int("x")
    inv = z3.Implies(x >= 1, x >= 0)
    ob, paths = _decrement_obligation(invariants=(inv,), init=(x >= 1))
    src = emit_program("conditional", ob, paths)
    assert "def invariants (s : Vector 1 Int) : Prop :=\n  True" not in src, \
        "the implication was dropped from `invariants`"
    assert "∨" in src.split("def invariants")[1].split("\n\n")[0], \
        "an implication should render as a disjunction omega can split"


def test_branching_init_reaches_the_emitted_file():
    """An `ite` in an entry value is not linear, so its conjunct must be case-split
    into the emitted `Init` rather than dropped, or `initiation` has no premise."""
    x = z3.Int("x")
    # branches must differ, or z3 folds the ite away before it is ever rendered, and
    # the condition must be over the state columns or the conjunct is not affine
    ob, paths = _decrement_obligation(init=z3.And(x == z3.If(x >= 3, 5, 7), x >= 1))
    src = emit_program("branchinit", ob, paths)
    init_block = src.split("def Init")[1].split("\n\n")[0]
    assert "ite" not in init_block and "If" not in init_block, \
        f"ite left unexpanded in Init: {init_block}"
    assert "∨" in init_block, \
        f"the branch was dropped instead of case-split: {init_block}"


def test_emit_multi_path_unions_the_step():
    """A branching body yields one namespace per path and a Step that is their
    union, dispatched in `program_terminates`."""
    layers = [(np.array([[1], [-1]]), np.array([0, 0])),
              (np.array([[1, 1]]), np.array([0]))]
    x = z3.Int("x")
    s, sp = [x], [z3.If(x != 0, z3.If(x > 0, x - 1, x + 1), x)]
    res = certify_decrease(Net.from_layers(layers), s, sp, x != 0, (), 1.0)
    assert res.verified, res.status
    assert len(res.certificates) == 2
    ob = Obligation(("x",), s, sp, None, None, 1.0, x != 0, layers=layers)
    src = emit_program("branching", ob, res.certificates)
    assert "namespace loop0_path0" in src and "namespace loop0_path1" in src
    assert "loop0_path0.Step a b ∨ loop0_path1.Step a b" in src
    assert "rintro a b (h | h)" in src


def test_non_trivial_cell_uses_its_certificate():
    """A cell whose decrease needs the guard carries a Farkas system, and the
    emitted proof reaches it through farkas_sound and decrease_bridge."""
    layers = [(np.array([[2]]), np.array([-1])), (np.array([[1]]), np.array([0]))]
    x = z3.Int("x")
    s, sp = [x], [z3.If(x > 0, x - 1, x)]
    res = certify_decrease(Net.from_layers(layers), s, sp, x > 0, (), 1.0)
    assert res.verified, res.status
    ob = Obligation(("x",), s, sp, None, None, 1.0, x > 0, layers=layers)
    src = emit_program("nontrivial", ob, res.certificates)
    assert "farkas_sound" in src and "decrease_bridge" in src


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_emitted_proof_kernel_checks(tmp_path):
    """End-to-end: the emitted file compiles against the vendored substrate."""
    x = z3.Int("x")
    ob, paths = _decrement_obligation(invariants=(x >= 0,), init=(x >= 0))
    out = LEAN_DIR / "proofs" / "_test_decrement"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "program.lean"
    f.write_text(emit_program("decrement", ob, paths))
    try:
        r = subprocess.run(["lake", "env", "lean", str(f.relative_to(LEAN_DIR))],
                           cwd=LEAN_DIR, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0 and not r.stdout.strip(), r.stdout + r.stderr
    finally:
        shutil.rmtree(out, ignore_errors=True)
