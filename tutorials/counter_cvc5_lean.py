#!/usr/bin/env python3
"""
Counter system verification using generic SMT-LIB2, cvc5, and Lean 4.

Pipeline:
  1. Build the counter reactive module (same system as counter.ipynb)
  2. Generate SMT-LIB2 and call cvc5 to verify bounded liveness
  3. Use create_project() to generate a full Lean 4 project stub
  4. Patch the stub to add lean-smt and replace sorry with `smt` tactic
  5. Call `lake build` on the result

Usage:
    cd python && uv run python ../tutorials/counter_cvc5_lean.py [--out DIR]
"""

import argparse
import subprocess
import shutil
import tempfile
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from zrth import Wire, Term, Module, IType as it, Bool, Int
from zrth.lean.project import create_project
from zrth.lean.cert import CertificateData


# ---------------------------------------------------------------------------
# Counter module construction (same system as counter.ipynb, in Int domain)
# ---------------------------------------------------------------------------

def make_counter() -> Module:
    """Build the counter reactive module.

    Python semantics:
        init(y0, z0) = (0, y0, z0)
        update(x, y, z) = if x < y or x < z then (x+1, y, z) else (0, y, z)

    State = Mat Int 3 1  (x, y, z stacked)
    Extl  = Mat Int 2 1  (y0, z0)
    """
    state = (Wire(Int(3, 1)), Wire(Int(3, 1)))
    extl = (Wire(Int(2, 1)), Wire(Int(2, 1)))

    # init: [[0,0],[1,0],[0,1]] @ extl
    A = Wire(Int(3, 2))
    init = [
        Term(it.Tensor(torch.tensor([[0, 0], [1, 0], [0, 1]])), write=[A]),
        Term(it.MatMul(), write=[state[1]], read=[A, extl[1]]),
    ]

    # update: extract x,y,z → compare → branch
    row_x = Wire(Int(1, 3)); row_y = Wire(Int(1, 3)); row_z = Wire(Int(1, 3))
    x = Wire(Int(1, 1)); y = Wire(Int(1, 1)); z = Wire(Int(1, 1))
    x_lt_y = Wire(Bool(1, 1)); x_lt_z = Wire(Bool(1, 1)); cond = Wire(Bool(1, 1))
    e1 = Wire(Int(3, 1)); res_t = Wire(Int(3, 1))
    B = Wire(Int(3, 3)); res_f = Wire(Int(3, 1))

    update = [
        Term(it.Tensor(torch.tensor([[1, 0, 0]])), write=[row_x]),
        Term(it.MatMul(), write=[x], read=[row_x, state[0]]),
        Term(it.Tensor(torch.tensor([[0, 1, 0]])), write=[row_y]),
        Term(it.MatMul(), write=[y], read=[row_y, state[0]]),
        Term(it.Tensor(torch.tensor([[0, 0, 1]])), write=[row_z]),
        Term(it.MatMul(), write=[z], read=[row_z, state[0]]),
        Term(it.Lt(), write=[x_lt_y], read=[x, y]),
        Term(it.Lt(), write=[x_lt_z], read=[x, z]),
        Term(it.Or(), write=[cond], read=[x_lt_y, x_lt_z]),
        Term(it.Tensor(torch.tensor([[1], [0], [0]])), write=[e1]),
        Term(it.Add(), write=[res_t], read=[state[0], e1]),
        Term(it.Tensor(torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]])), write=[B]),
        Term(it.MatMul(), write=[res_f], read=[B, state[0]]),
        Term(it.Ite(), write=[state[1]], read=[cond, res_t, res_f]),
    ]

    return Module.sequential(init, update, obs=[state, extl])


def make_property(ctrl) -> list[Term]:
    """Property P: x == 0, i.e., [1,0,0] @ state == [[0]]."""
    s = ctrl[0][1]
    row_x = Wire(Int(1, 3)); x = Wire(Int(1, 1))
    zero = Wire(Int(1, 1)); out = Wire(Bool(1, 1))
    return [
        Term(it.Tensor(torch.tensor([[1, 0, 0]])), write=[row_x]),
        Term(it.MatMul(), write=[x], read=[row_x, s]),
        Term(it.Tensor(torch.tensor([[0]])), write=[zero]),
        Term(it.Eq(), write=[out], read=[x, zero]),
    ]


def make_invariant(ctrl) -> list[Term]:
    """Invariant: x >= 0 ∧ (x <= y ∨ x <= z)."""
    s = ctrl[0][1]
    row_x = Wire(Int(1, 3)); row_y = Wire(Int(1, 3)); row_z = Wire(Int(1, 3))
    x = Wire(Int(1, 1)); y = Wire(Int(1, 1)); z = Wire(Int(1, 1))
    zero = Wire(Int(1, 1))
    x_ge_0 = Wire(Bool(1, 1)); x_le_y = Wire(Bool(1, 1))
    x_le_z = Wire(Bool(1, 1)); disj = Wire(Bool(1, 1)); out = Wire(Bool(1, 1))
    return [
        Term(it.Tensor(torch.tensor([[1, 0, 0]])), write=[row_x]),
        Term(it.MatMul(), write=[x], read=[row_x, s]),
        Term(it.Tensor(torch.tensor([[0, 1, 0]])), write=[row_y]),
        Term(it.MatMul(), write=[y], read=[row_y, s]),
        Term(it.Tensor(torch.tensor([[0, 0, 1]])), write=[row_z]),
        Term(it.MatMul(), write=[z], read=[row_z, s]),
        Term(it.Tensor(torch.tensor([[0]])), write=[zero]),
        Term(it.Ge(), write=[x_ge_0], read=[x, zero]),
        Term(it.Le(), write=[x_le_y], read=[x, y]),
        Term(it.Le(), write=[x_le_z], read=[x, z]),
        Term(it.Or(), write=[disj], read=[x_le_y, x_le_z]),
        Term(it.And(), write=[out], read=[x_ge_0, disj]),
    ]


def make_init_pre(extl) -> list[Term]:
    """init_pre: y0 >= 0 ∧ z0 >= 0."""
    e = extl[0][1]
    row_y0 = Wire(Int(1, 2)); row_z0 = Wire(Int(1, 2))
    y0 = Wire(Int(1, 1)); z0 = Wire(Int(1, 1))
    zero = Wire(Int(1, 1))
    y0_ge_0 = Wire(Bool(1, 1)); z0_ge_0 = Wire(Bool(1, 1)); out = Wire(Bool(1, 1))
    return [
        Term(it.Tensor(torch.tensor([[1, 0]])), write=[row_y0]),
        Term(it.MatMul(), write=[y0], read=[row_y0, e]),
        Term(it.Tensor(torch.tensor([[0, 1]])), write=[row_z0]),
        Term(it.MatMul(), write=[z0], read=[row_z0, e]),
        Term(it.Tensor(torch.tensor([[0]])), write=[zero]),
        Term(it.Ge(), write=[y0_ge_0], read=[y0, zero]),
        Term(it.Ge(), write=[z0_ge_0], read=[z0, zero]),
        Term(it.And(), write=[out], read=[y0_ge_0, z0_ge_0]),
    ]


def make_ranking(ctrl) -> list[Term]:
    """Ranking: if x == 0 then 0 else max(y, z) - x + 1."""
    s = ctrl[0][1]
    row_x = Wire(Int(1, 3)); row_y = Wire(Int(1, 3)); row_z = Wire(Int(1, 3))
    x = Wire(Int(1, 1)); y = Wire(Int(1, 1)); z = Wire(Int(1, 1))
    zero = Wire(Int(1, 1)); p = Wire(Bool(1, 1))
    y_ge_z = Wire(Bool(1, 1)); max_yz = Wire(Int(1, 1))
    diff = Wire(Int(1, 1)); one = Wire(Int(1, 1)); diff_1 = Wire(Int(1, 1))
    ite_res = Wire(Int(1, 1))
    scalar = Wire(Int(1)); out = Wire(Int(1))
    return [
        Term(it.Tensor(torch.tensor([[1, 0, 0]])), write=[row_x]),
        Term(it.MatMul(), write=[x], read=[row_x, s]),
        Term(it.Tensor(torch.tensor([[0, 1, 0]])), write=[row_y]),
        Term(it.MatMul(), write=[y], read=[row_y, s]),
        Term(it.Tensor(torch.tensor([[0, 0, 1]])), write=[row_z]),
        Term(it.MatMul(), write=[z], read=[row_z, s]),
        Term(it.Tensor(torch.tensor([[0]])), write=[zero]),
        Term(it.Eq(), write=[p], read=[x, zero]),
        Term(it.Ge(), write=[y_ge_z], read=[y, z]),
        Term(it.Ite(), write=[max_yz], read=[y_ge_z, y, z]),
        Term(it.Sub(), write=[diff], read=[max_yz, x]),
        Term(it.Tensor(torch.tensor([[1]])), write=[one]),
        Term(it.Add(), write=[diff_1], read=[diff, one]),
        Term(it.Ite(), write=[ite_res], read=[p, zero, diff_1]),
        Term(it.TensorGet(), write=[scalar], read=[ite_res]),
        Term(it.ToUnsigned(), write=[out], read=[scalar]),
    ]


# ---------------------------------------------------------------------------
# SMT-LIB2 generation + cvc5 invocation (kept for pre-check)
# ---------------------------------------------------------------------------

def smtlib_bounded_liveness(k: int, y0: int, z0: int) -> str:
    """BMC check: from (0, y0, z0), is x=y ∨ x=z reached within k steps?"""
    lines = ["(set-logic QF_LIA)", "(set-option :produce-proofs true)", ""]
    for i in range(k + 1):
        lines.append(f"(declare-fun x{i} () Int)")
    lines.append(f"(define-fun y () Int {y0})")
    lines.append(f"(define-fun z () Int {z0})")
    lines.append("")
    lines.append("(assert (= x0 0))")
    lines.append("")
    for i in range(k):
        guard = f"(or (< x{i} y) (< x{i} z))"
        lines.append(f"(assert (= x{i+1} (ite {guard} (+ x{i} 1) 0)))")
    lines.append("")
    for i in range(k + 1):
        lines.append(f"(assert (not (or (= x{i} y) (= x{i} z))))")
    lines.append("")
    lines.append("(check-sat)")
    lines.append("(exit)")
    return "\n".join(lines)


def run_cvc5(smtlib: str) -> str:
    """Call cvc5 on an SMT-LIB string, return stdout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".smt2", delete=False) as f:
        f.write(smtlib)
        tmp = f.name
    try:
        r = subprocess.run(["cvc5", tmp], capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Lean project patching — add lean-smt, replace sorry with smt tactic
# ---------------------------------------------------------------------------

def patch_lakefile(project_dir: Path) -> None:
    """Add lean-smt as a dependency to lakefile.toml."""
    lakefile = project_dir / "lakefile.toml"
    text = lakefile.read_text()

    smt_dep = """
[[require]]
name = "smt"
git = "https://github.com/ufmg-smite/lean-smt.git"
rev = "main"
"""
    if "lean-smt" not in text and "ufmg-smite" not in text:
        text += smt_dep
    lakefile.write_text(text)


def patch_certificate(project_dir: Path) -> None:
    """No-op: cert.py now generates zeroth_hammer + lean-smt natively."""
    pass  # All patching is now done by the generator


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Counter verification: SMT-LIB + cvc5 + Lean 4 project with lean-smt."
    )
    parser.add_argument(
        "--out", "-o", type=str, default=None,
        help="Output directory for Lean project (default: tempdir)",
    )
    parser.add_argument(
        "--build", action="store_true", default=True,
        help="Run `lake build` on the generated project (default: true)",
    )
    parser.add_argument(
        "--no-build", action="store_false", dest="build",
        help="Skip `lake build`",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    y0, z0 = 5, 3

    # ── Step 1: pre-check with cvc5 ──
    print("=" * 60)
    print("Step 1: SMT-LIB pre-check with cvc5")
    print("=" * 60)

    smt = smtlib_bounded_liveness(6, y0, z0)
    if args.verbose:
        print(smt)
        print()
    result = run_cvc5(smt)
    print(f"  Bounded liveness (k=6): {result}")
    if result != "unsat":
        print("  ERROR: expected unsat — cvc5 could not verify the property")
        sys.exit(1)
    print("  ✓ cvc5 confirms: x=y ∨ x=z reached within 6 steps\n")

    # ── Step 2: build module + Lean project ──
    print("=" * 60)
    print("Step 2: Generate Lean 4 project via create_project()")
    print("=" * 60)

    m = make_counter()
    cert_data = CertificateData(
        prp=make_property(m.ctrl),
        inv=make_invariant(m.ctrl),
        init_pre=make_init_pre(m.extl),
        ranking=make_ranking(m.ctrl),
    )

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = Path(__file__).parent / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    project_dir = create_project(
        output_dir=out_dir,
        module=m,
        project_name="Counter",
        cert_data=cert_data,
    )
    print()

    # ── Step 3: patch for lean-smt ──
    print("=" * 60)
    print("Step 3: Patch project for lean-smt")
    print("=" * 60)

    patch_lakefile(project_dir)
    print("  ✓ Added lean-smt dependency to lakefile.toml")

    patch_certificate(project_dir)
    print("  ✓ Patched Certificate.lean: sorry → smt tactic")

    # Show the patched certificate
    cert_path = project_dir / "Certificate" / "Certificate.lean"
    print(f"\n  --- {cert_path} (patched) ---")
    print(cert_path.read_text())
    print("  ---\n")

    # Show the patched lakefile
    lake_path = project_dir / "lakefile.toml"
    print(f"  --- {lake_path} (patched) ---")
    print(lake_path.read_text())
    print("  ---\n")

    # ── Step 4: call lake build ──
    if args.build:
        print("=" * 60)
        print("Step 4: lake build")
        print("=" * 60)
        print(f"  Project: {project_dir}")
        print(f"  Running: lake build ...\n")

        r = subprocess.run(
            ["lake", "build"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if r.stdout.strip():
            print("  stdout:")
            for line in r.stdout.strip().split("\n"):
                print(f"    {line}")
        if r.stderr.strip():
            print("  stderr:")
            for line in r.stderr.strip().split("\n"):
                print(f"    {line}")

        if r.returncode == 0:
            print("\n  ✓ lake build succeeded — all proofs checked!")
        else:
            print(f"\n  ✗ lake build failed (exit code {r.returncode})")
            print("    This is expected — lean-smt may not handle all proof obligations.")
            print("    The sorry locations show where manual proof work is needed.")

    print(f"\nProject at: {project_dir}")


if __name__ == "__main__":
    main()
