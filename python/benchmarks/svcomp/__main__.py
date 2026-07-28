"""Run the ranking-function pipeline over the sv_comp corpus.

    python -m benchmarks.svcomp                  # train + verify every benchmark
    python -m benchmarks.svcomp ndecr            # only names containing 'ndecr'
    python -m benchmarks.svcomp --lean           # also emit and kernel-check proofs
    python -m benchmarks.svcomp --lean --jobs 4  # with four benchmarks in flight

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

import dataclasses
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from . import discover
from ._train import learn_ranking
from ._verify_ranking import farkas_cell

WORKER_FLAG = "--worker"
_WORKER_TIMEOUT = 2400.0        # above _lean_check.CHECK_TIMEOUT plus training


def _default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


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


def certify_one(bench):
    """Train, verify, emit and kernel-check one benchmark in this process."""
    from . import _lean_check as lc

    t0 = time.perf_counter()
    try:
        # the trainer verified the candidate it accepted; emit that evidence
        r = learn_ranking(bench, verifier=farkas_cell)
        t_train = time.perf_counter() - t0
        res = (lc.certify(bench.name, r.obligation, r.verification.certificate)
               if r.verified
               else lc.CheckResult(bench.name, "UNVERIFIED", detail=r.reason or ""))
    except Exception as e:
        t_train = time.perf_counter() - t0
        res = lc.CheckResult(bench.name, "ERROR", detail=f"{type(e).__name__}: {e}")
    res.train_s = t_train
    return res


def _spawn(name: str):
    """Run ``certify_one`` for ``name`` in a fresh process and read back its
    result, so the benchmark sees an empty Z3 context."""
    from . import _lean_check as lc

    cmd = [sys.executable, "-m", "benchmarks.svcomp", WORKER_FLAG, name]
    # the ranking nets have a handful of hidden units, so a worker gains nothing
    # from intra-op threads; left at the default each would claim several cores
    # and the pool would oversubscribe the machine several times over.
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=_WORKER_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return lc.CheckResult(name, "TIMEOUT", detail=f"worker exceeded {_WORKER_TIMEOUT:g}s")
    for line in reversed(p.stdout.strip().splitlines()):
        if line.startswith("{"):
            return lc.CheckResult(**json.loads(line))
    detail = (p.stderr.strip().splitlines() or ["worker produced no result"])[-1]
    return lc.CheckResult(name, "ERROR", detail=detail)


def _run_lean(benches, jobs: int) -> int:
    from . import _lean_check as lc

    if not lc.toolchain_available():
        print("no `lake` on PATH — install the Lean toolchain to use --lean")
        return -1
    print(f"building the Lean substrate ... ({len(benches)} benchmarks, "
          f"{jobs} at a time)", flush=True)
    lc.build_substrate()

    tally: Counter[str] = Counter()
    train_s = check_s = 0.0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        # each future waits on its own fresh process; results are reported in
        # submission order so one run's output can be diffed against another's
        futures = [pool.submit(_spawn, b.name) for b in benches]
        for f in futures:
            res = f.result()
            tally[res.outcome] += 1
            train_s += res.train_s
            check_s += res.check_s
            if res.outcome in ("UNVERIFIED", "ERROR", "TIMEOUT"):
                print(f"{res.outcome.lower():10s} {res.name}: {res.detail} "
                      f"[{res.train_s:.1f}s]", flush=True)
            else:
                print(f"{res.outcome:10s} {res.name}: {res.n_paths} paths, "
                      f"{res.n_cells} cells, {res.n_invariants} invariants "
                      f"[train {res.train_s:.1f}s, check {res.check_s:.1f}s]", flush=True)
                if res.detail:
                    print(f"           {res.detail}", flush=True)

    print(f"\n{tally['CHECKED']}/{len(benches)} kernel-checked")
    for outcome, n in sorted(tally.items()):
        if outcome != "CHECKED":
            print(f"  {outcome}: {n}")
    print(f"cpu: train {train_s:.0f}s, check {check_s:.0f}s")
    return tally["CHECKED"]


def _run_worker(name: str) -> int:
    """One benchmark, one JSON line on stdout. Invoked by :func:`_spawn`."""
    match = [b for b in discover() if b.name == name]
    if not match:
        print(f"no benchmark named {name!r}", file=sys.stderr)
        return 1
    print(json.dumps(dataclasses.asdict(certify_one(match[0]))), flush=True)
    return 0


def main(argv: list[str]) -> int:
    argv = list(argv)
    jobs = _default_jobs()
    if "--jobs" in argv:
        i = argv.index("--jobs")
        try:
            jobs = int(argv[i + 1])
        except (IndexError, ValueError):
            print("--jobs needs a positive integer")
            return 2
        if jobs < 1:
            print("--jobs needs a positive integer")
            return 2
        del argv[i:i + 2]

    flags = {a for a in argv if a.startswith("--")}
    unknown = flags - {"--lean", WORKER_FLAG}
    if unknown:
        print(f"unknown flag(s): {', '.join(sorted(unknown))}")
        return 2
    args = [a for a in argv if not a.startswith("--")]

    if WORKER_FLAG in flags:
        return _run_worker(args[0])

    only = args[0] if args else None
    benches = [b for b in discover() if only is None or only in b.name]
    if not benches:
        print(f"no benchmarks match {only!r}")
        return 1

    t0 = time.perf_counter()
    n = (_run_lean(benches, jobs) if "--lean" in flags else _run_plain(benches))
    if n < 0:
        return 1
    print(f"({time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
