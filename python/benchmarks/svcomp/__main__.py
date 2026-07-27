"""Run the ranking-function pipeline over the sv_comp corpus.

    python -m benchmarks.svcomp                  # train + verify every benchmark
    python -m benchmarks.svcomp ndecr            # only names containing 'ndecr'
    python -m benchmarks.svcomp --lean           # also emit and kernel-check proofs

Without ``--lean`` each DSL-encoded program gets a trained neural ranking
function verified over the composed module (``smt_oneshot``), summarised as
``verified / total``.

With ``--lean`` the ranking function is verified by the Farkas cell/CEGAR
verifier, its certificates are emitted as a Lean proof under
``lean/proofs/<name>/``, and that proof is compiled against the vendored
substrate. The summary counts ``CHECKED`` — programs whose termination the Lean
kernel verified — and breaks the rest down by outcome (see :mod:`._lean_check`).
Requires ``lake`` on PATH.

Faithfulness of the encodings against the C sources is a separate check::

    python -m benchmarks.svcomp._equiv
"""

from __future__ import annotations

import sys
import time
from collections import Counter

from . import discover
from ._train import learn_ranking
from ._verify_ranking import farkas_cell


def _run_plain(benches) -> int:
    verified = 0
    for b in benches:
        r = learn_ranking(b)
        verified += bool(r.verified)
        tag = "VERIFIED  " if r.verified else "unverified"
        extra = f" ({r.reason})" if r.reason else ""
        print(f"{tag} {r.name}: {r.n_pairs} pairs, loss {r.final_loss:.4g}{extra}",
              flush=True)
    print(f"\n{verified}/{len(benches)} verified")
    return verified


def _run_lean(benches) -> int:
    from . import _lean_check as lc

    if not lc.toolchain_available():
        print("no `lake` on PATH — install the Lean toolchain to use --lean")
        return -1
    print("building the Lean substrate ...", flush=True)
    lc.build_substrate()

    tally: Counter[str] = Counter()
    for b in benches:
        try:
            # the trainer verified the candidate it accepted; emit that evidence
            r = learn_ranking(b, verifier=farkas_cell)
            res = (lc.certify(b.name, r.obligation, r.verification.certificate)
                   if r.verified
                   else lc.CheckResult(b.name, "UNVERIFIED", detail=r.reason or ""))
        except Exception as e:                       # one benchmark must not end the run
            res = lc.CheckResult(b.name, "ERROR", detail=f"{type(e).__name__}: {e}")
        tally[res.outcome] += 1
        if res.outcome == "UNVERIFIED":
            print(f"{'unverified':10s} {res.name}: {res.detail}", flush=True)
        else:
            print(f"{res.outcome:10s} {res.name}: {res.n_paths} paths, "
                  f"{res.n_cells} cells, {res.n_invariants} invariants", flush=True)
            if res.detail:
                print(f"           {res.detail}", flush=True)

    print(f"\n{tally['CHECKED']}/{len(benches)} kernel-checked")
    for outcome, n in sorted(tally.items()):
        if outcome != "CHECKED":
            print(f"  {outcome}: {n}")
    return tally["CHECKED"]


def main(argv: list[str]) -> int:
    flags = {a for a in argv if a.startswith("--")}
    unknown = flags - {"--lean"}
    if unknown:
        print(f"unknown flag(s): {', '.join(sorted(unknown))}")
        return 2
    args = [a for a in argv if not a.startswith("--")]

    only = args[0] if args else None
    benches = [b for b in discover() if only is None or only in b.name]
    if not benches:
        print(f"no benchmarks match {only!r}")
        return 1

    t0 = time.perf_counter()
    n = _run_lean(benches) if "--lean" in flags else _run_plain(benches)
    if n < 0:
        return 1
    print(f"({time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
