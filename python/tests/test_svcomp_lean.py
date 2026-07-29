"""Regression tests for the Lean emitter (``benchmarks.svcomp._lean``).

The structural tests run everywhere. The end-to-end test compiles an emitted
proof with ``lake`` and is skipped when no toolchain is present, so it protects
the pipeline on a machine that has one without breaking one that does not.
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import z3

from benchmarks.svcomp._farkas import certify_decrease
from benchmarks.svcomp._lean import emit_program
from benchmarks.svcomp._verify_ranking import Obligation

LEAN_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "svcomp" / "lean"


def _decrement_obligation(invariants=(), init=None):
    """`while (x > 0) x = x - 1` with V(s) = relu(x), as an Obligation plus its
    certified paths."""
    layers = [(np.array([[1]]), np.array([0])), (np.array([[1]]), np.array([0]))]
    x = z3.Int("x")
    s, sp = [x], [z3.If(x > 0, x - 1, x)]
    res = certify_decrease(layers, s, sp, x > 0, invariants, 1.0)
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
    (cell,) = paths[0].cells
    src = emit_program("decrement", ob, paths)
    assert "unconditional decrease (no Farkas system)" in src
    assert "cell0_A" not in src
    assert "cell0_decrease" in src and "by omega" in src


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


def test_emit_multi_path_unions_the_step():
    """A branching body yields one namespace per path and a Step that is their
    union, dispatched in `program_terminates`."""
    layers = [(np.array([[1], [-1]]), np.array([0, 0])),
              (np.array([[1, 1]]), np.array([0]))]
    x = z3.Int("x")
    s, sp = [x], [z3.If(x != 0, z3.If(x > 0, x - 1, x + 1), x)]
    res = certify_decrease(layers, s, sp, x != 0, (), 1.0)
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
    res = certify_decrease(layers, s, sp, x > 0, (), 1.0)
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
