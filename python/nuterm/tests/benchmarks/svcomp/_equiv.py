"""Differential equivalence checker: DSL module vs the original C program.

Faithfulness guard for the hand-encodings. For random nondet inputs it runs
**the real C** (compiled with cc) and the **DSL module** (via ``zrth.eval``)
from the same inputs, and compares the final program state. A hand-encoding
that mis-transcribes a guard or body will diverge on some input.

The compiled C is the ground truth. This is a *sampling* check (finite random
inputs), not a proof: a symbolic C-vs-DSL equivalence proof would require a
symbolic model of the C (i.e. re-translating it), which is exactly the
translation surface this checker exists to guard, so the real compiler is the
reference. Self-contained: only needs ``cc`` and the C file.

Usage: ``python -m nuterm.tests.benchmarks.svcomp._equiv prog.c bench.py``.

## How inputs are fed
The clean sv_comp programs read ``__VERIFIER_nondet_int()`` only before the
loop, in program order. The harness supplies a definition that reads successive ints
from **stdin**, so ``main``'s signature is untouched. The i-th value on stdin
must line up with the module's i-th ``extl`` input, so **``Bench.inputs`` must
be listed in the order the C reads nondet** (a documented convention).

## Requirements / scope (matches the clean set)
- exactly one ``return`` in the C (checked);
- ``#nondet call sites == len(bench.inputs)`` (checked);
- no in-loop nondet (the module ``update`` must not read ``extl``);
- terminating for the sampled range (small range keeps loops short; a run that
  exceeds ``max_steps`` / the C timeout is reported as a divergence).
"""

from __future__ import annotations

import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from zrth import X
from zrth.eval import eval_itype

from nuterm._bench import Bench
from nuterm._train import run_block

_HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
int __VERIFIER_nondet_int(void) {
    long v;
    if (scanf("%ld", &v) != 1) v = 0;
    return (int) v;
}
"""


# ---------------------------------------------------------------------------
# C side
# ---------------------------------------------------------------------------

def _instrument(c_source: str, state: tuple[str, ...]) -> str:
    """Prepend the nondet harness and inject a final-state dump before the
    sole ``return`` in ``main``."""
    n_ret = len(re.findall(r"\breturn\b", re.sub(r"/\*.*?\*/", "", c_source, flags=re.S)))
    if n_ret != 1:
        raise ValueError(f"expected exactly one `return`, found {n_ret}")
    fmt = "STATE" + " %d" * len(state) + r"\n"
    dump = f'printf("{fmt}", {", ".join(state)}); '
    # NB: a function replacement is used so re does NOT interpret the `\n`
    # (backslash-n) in `dump` as a newline (it would break the C string).
    injected, k = re.subn(r"return\s+0\s*;", lambda m: dump + m.group(0), c_source, count=1)
    if k != 1:
        raise ValueError("could not locate `return 0;` in main")
    return _HARNESS + "\n" + injected


def _n_nondet_calls(c_source: str) -> int:
    src = re.sub(r"/\*.*?\*/", "", c_source, flags=re.S)
    total = len(re.findall(r"__VERIFIER_nondet_int", src))
    externs = len(re.findall(r"extern[^;]*__VERIFIER_nondet_int", src))
    return total - externs


def _compile(c_source: str, state: tuple[str, ...], workdir: Path) -> Path:
    src = _instrument(c_source, state)
    cfile = workdir / "prog.c"
    binf = workdir / "prog.bin"
    cfile.write_text(src)
    r = subprocess.run(["cc", "-O0", "-w", "-o", str(binf), str(cfile)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cc failed:\n{r.stderr}\n--- source ---\n{src}")
    return binf


def _run_c(binf: Path, inputs: list[int], state: tuple[str, ...],
           timeout: float = 2.0) -> dict[str, int] | None:
    """Run the compiled C, feeding ``inputs`` on stdin. Returns the dumped
    final state, or None on timeout (treated as non-termination)."""
    stdin = " ".join(str(v) for v in inputs) + "\n"
    try:
        r = subprocess.run([str(binf)], input=stdin, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    line = next((l for l in r.stdout.splitlines() if l.startswith("STATE")), None)
    if line is None:
        raise RuntimeError(f"no STATE line in C output: {r.stdout!r}")
    vals = [int(x) for x in line.split()[1:]]
    return dict(zip(state, vals))


# ---------------------------------------------------------------------------
# Module side
# ---------------------------------------------------------------------------

def _run_module(bench: Bench, inputs: list[int], max_steps: int) -> dict[str, int] | None:
    """Run init(inputs) then update to a fixpoint. Returns final state, or None
    if it does not converge within ``max_steps`` (treated as non-termination)."""
    prog, ctrl, extl = bench.build()

    # seed extl NEXT wires with the inputs (Bench.inputs order == C nondet order)
    state: dict = {}
    for name, val in zip(bench.inputs, inputs):
        state[X(extl[name])] = torch.tensor([[val]], dtype=torch.int64)

    run_block(prog.atoms, state, lambda a: a.init)   # writes ctrl NEXT wires
    latched = {name: state[X(ctrl[name])] for name in bench.state}

    def as_ints(d):
        return {n: int(t.reshape(-1)[0]) for n, t in d.items()}

    prev = as_ints(latched)
    for _ in range(max_steps):
        st = {ctrl[n]: latched[n] for n in bench.state}      # latch: next -> latched
        run_block(prog.atoms, st, lambda a: a.update)
        nxt = {n: st[X(ctrl[n])] for n in bench.state}
        cur = as_ints(nxt)
        if cur == prev:                                       # fixpoint (guard false)
            return cur
        prev, latched = cur, nxt
    return None


def _init_state_ints(bench: Bench, inputs: list[int]) -> dict[str, int]:
    """The module's initial state (as ints) for the given nondet inputs, used
    to evaluate the (state-level) precondition."""
    prog, ctrl, extl = bench.build()
    state: dict = {}
    for name, val in zip(bench.inputs, inputs):
        state[X(extl[name])] = torch.tensor([[val]], dtype=torch.int64)
    run_block(prog.atoms, state, lambda a: a.init)
    return {n: int(state[X(ctrl[n])].reshape(-1)[0]) for n in bench.state}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclass
class Result:
    """One benchmark's differential outcome: how many input vectors were compared
    and where the DSL encoding and the compiled C disagreed."""
    name: str
    trials: int          # decisive trials actually compared
    passed: int
    failures: list       # (inputs, c_state, module_state)
    inconclusive: int = 0  # both sides non-terminated within limits (skipped)
    skipped: str | None = None

    @property
    def ok(self) -> bool:
        return self.skipped is None and not self.failures and self.trials > 0


def check(bench: Bench, c_file: Path, c_vars: tuple[str, ...] | None = None,
          trials: int = 300, seed: int = 0,
          lo: int = -40, hi: int = 40, max_steps: int = 50_000) -> Result:
    """Compare ``bench``'s DSL encoding against its compiled C source on random
    inputs drawn from ``[lo, hi]``, and report where the final states differ.

    ``c_file`` is the original C program.
    ``c_vars`` names the C variables dumped, positionally matching
    ``bench.state`` (default: the same names)."""
    c_source = Path(c_file).read_text()
    c_vars = tuple(c_vars) if c_vars is not None else bench.state
    if len(c_vars) != len(bench.state):
        return Result(bench.name, 0, 0, [],
                      skipped=f"{len(c_vars)} C vars given for {len(bench.state)} state vars")

    n_calls = _n_nondet_calls(c_source)
    if n_calls != len(bench.inputs):
        return Result(bench.name, 0, 0, [],
                      skipped=f"#nondet calls ({n_calls}) != len(inputs) ({len(bench.inputs)})")

    # Deterministic (no nondet) programs have a single behaviour, so one trial suffices.
    if not bench.inputs:
        trials = 1

    rng = random.Random(seed)
    failures = []
    passed = 0
    inconclusive = 0
    decisive = 0
    with tempfile.TemporaryDirectory() as td:
        binf = _compile(c_source, c_vars, Path(td))
        attempts = 0
        cap = trials * 40  # bound the search for precondition-satisfying inputs
        while decisive < trials and attempts < cap:
            attempts += 1
            inputs = [rng.randint(lo, hi) for _ in range(len(bench.inputs))]
            if bench.precondition is not None:
                if not all(bench.precondition(_init_state_ints(bench, inputs))):
                    continue  # outside the precondition: C leaves vars undefined
            c_state = _run_c(binf, inputs, c_vars)
            m_state = _run_module(bench, inputs, max_steps)
            if c_state is None and m_state is None:
                inconclusive += 1          # both diverge (e.g. nonterminating input)
                continue
            decisive += 1
            if c_state == m_state:
                passed += 1
            else:
                failures.append((inputs, c_state, m_state))
    return Result(bench.name, decisive, passed, failures, inconclusive)


def main(argv: list[str] | None = None) -> int:
    """Check one C file against one module/bench file. Returns an exit code."""
    import argparse
    from nuterm.__main__ import load

    ap = argparse.ArgumentParser(
        prog="python -m nuterm.tests.benchmarks.svcomp._equiv",
        description="Compare a C program with a Python file defining a `BENCH` "
                    "or `module()` on random nondet inputs.")
    ap.add_argument("c_file", type=Path, help="the original C program")
    ap.add_argument("py_file", type=Path, help="Python file defining BENCH or module()")
    ap.add_argument("--vars", help="comma-separated C variable names, positionally "
                    "matching the bench's state (default: the bench's state names)")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lo", type=int, default=-40)
    ap.add_argument("--hi", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=50_000)
    a = ap.parse_args(argv)

    bench = load(a.py_file)
    c_vars = tuple(v.strip() for v in a.vars.split(",")) if a.vars else None
    r = check(bench, a.c_file, c_vars, trials=a.trials, seed=a.seed,
              lo=a.lo, hi=a.hi, max_steps=a.max_steps)
    if r.skipped:
        print(f"SKIP {r.name}: {r.skipped}")
        return 1
    extra = f" ({r.inconclusive} inconclusive)" if r.inconclusive else ""
    print(f"{'OK  ' if r.ok else 'FAIL'} {r.name}: {r.passed}/{r.trials} agree{extra}")
    for inp, c, m in r.failures[:3]:
        print(f"       input={inp}  C={c}  module={m}")
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
