"""Differential equivalence checker: DSL module vs the original C program.

Faithfulness guard for the hand-encodings. For random nondet inputs it runs
**the real C** (compiled with cc) and the **DSL module** (via ``zrth.eval``)
from the same inputs, and compares the final program state. A hand-encoding
that mis-transcribes a guard or body will diverge on some input.

The compiled C is the ground truth. This is a *sampling* check (finite random
inputs), not a proof: a symbolic C-vs-DSL equivalence proof would require a
symbolic model of the C (i.e. re-translating it), which is exactly the
translation surface this checker exists to guard — so we rely on the real
compiler instead. Self-contained: only needs ``cc`` and the C files in ``c/``.

## How inputs are fed
The clean sv_comp programs read ``__VERIFIER_nondet_int()`` only before the
loop, in program order. We provide a definition that reads successive ints
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
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from zrth.eval import eval_itype

from ._bench import Bench

# The C sources live in this package (self-contained).
_C_DIR = Path(__file__).parent / "c"

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
           timeout: float = 5.0) -> dict[str, int] | None:
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

def _run_block(atoms, state, get_block):
    for a in atoms:
        for t in get_block(a):
            read = [state[w] for w in t.read]
            out_sort = t.write[0].dtype if len(t.write) else None
            state.update(zip(t.write, eval_itype(t.itype, read, out_sort)))


def _run_module(bench: Bench, inputs: list[int], max_steps: int) -> dict[str, int] | None:
    """Run init(inputs) then update to a fixpoint. Returns final state, or None
    if it does not converge within ``max_steps`` (treated as non-termination)."""
    prog, ctrl, extl = bench.build()

    # seed extl NEXT wires with the inputs (Bench.inputs order == C nondet order)
    state: dict = {}
    for name, val in zip(bench.inputs, inputs):
        _lat, nxt = extl[name]
        state[nxt] = torch.tensor([[val]], dtype=torch.int64)

    _run_block(prog.atoms, state, lambda a: a.init)   # writes ctrl NEXT wires
    latched = {name: state[ctrl[name][1]] for name in bench.state}

    def as_ints(d):
        return {n: int(t.reshape(-1)[0]) for n, t in d.items()}

    prev = as_ints(latched)
    for _ in range(max_steps):
        st = {ctrl[n][0]: latched[n] for n in bench.state}   # latch: next -> latched
        _run_block(prog.atoms, st, lambda a: a.update)
        nxt = {n: st[ctrl[n][1]] for n in bench.state}
        cur = as_ints(nxt)
        if cur == prev:                                       # fixpoint (guard false)
            return cur
        prev, latched = cur, nxt
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclass
class Result:
    name: str
    trials: int
    passed: int
    failures: list  # (inputs, c_state, module_state)
    skipped: str | None = None

    @property
    def ok(self) -> bool:
        return self.skipped is None and not self.failures


def check(bench: Bench, trials: int = 300, seed: int = 0,
         lo: int = -40, hi: int = 40, max_steps: int = 50_000) -> Result:
    c_source = (_C_DIR / bench.source).read_text()

    n_calls = _n_nondet_calls(c_source)
    if n_calls != len(bench.inputs):
        return Result(bench.name, 0, 0, [],
                      skipped=f"#nondet calls ({n_calls}) != len(inputs) ({len(bench.inputs)})")

    # Deterministic (no nondet) programs have a single behaviour — one trial suffices.
    if not bench.inputs:
        trials = 1

    rng = random.Random(seed)
    failures = []
    passed = 0
    with tempfile.TemporaryDirectory() as td:
        binf = _compile(c_source, bench.state, Path(td))
        for _ in range(trials):
            inputs = [rng.randint(lo, hi) for _ in range(len(bench.inputs))]
            c_state = _run_c(binf, inputs, bench.state)
            m_state = _run_module(bench, inputs, max_steps)
            if c_state is None or m_state is None:
                # non-termination on one side within limits -> divergence signal
                if c_state != m_state:
                    failures.append((inputs, c_state, m_state))
                else:
                    passed += 1
                continue
            if c_state == m_state:
                passed += 1
            else:
                failures.append((inputs, c_state, m_state))
    return Result(bench.name, trials, passed, failures)


def check_all(**kw) -> list[Result]:
    from . import discover
    return [check(b, **kw) for b in discover()]


if __name__ == "__main__":
    import sys
    from . import discover

    only = sys.argv[1] if len(sys.argv) > 1 else None
    benches = [b for b in discover() if only is None or only in b.name]
    all_ok = True
    for b in benches:
        r = check(b)
        if r.skipped:
            print(f"SKIP {r.name}: {r.skipped}")
            continue
        print(f"{'OK  ' if r.ok else 'FAIL'} {r.name}: sampling {r.passed}/{r.trials} agree")
        for inp, c, m in r.failures[:3]:
            print(f"       input={inp}  C={c}  module={m}")
        all_ok &= r.ok
    sys.exit(0 if all_ok else 1)
