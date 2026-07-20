"""Run the ranking-function pipeline over the sv_comp corpus.

    python -m benchmarks.svcomp            # train + verify every benchmark
    python -m benchmarks.svcomp ndecr      # only benchmarks whose name contains 'ndecr'

For each DSL-encoded program it trains a neural ranking function and verifies it
over the composed module (``smt_oneshot``): one line per benchmark, then a
``verified / total`` summary with elapsed time. Faithfulness of the encodings
against the C sources is a separate check::

    python -m benchmarks.svcomp._equiv
"""

from __future__ import annotations

import sys
import time

from . import discover
from ._train import learn_ranking


def main(argv: list[str]) -> int:
    only = argv[0] if argv else None
    benches = [b for b in discover() if only is None or only in b.name]
    if not benches:
        print(f"no benchmarks match {only!r}")
        return 1

    t0 = time.perf_counter()
    verified = 0
    for b in benches:
        r = learn_ranking(b)
        verified += bool(r.verified)
        tag = "VERIFIED  " if r.verified else "unverified"
        extra = f" ({r.reason})" if r.reason else ""
        print(f"{tag} {r.name}: {r.n_pairs} pairs, loss {r.final_loss:.4g}{extra}")

    dt = time.perf_counter() - t0
    print(f"\n{verified}/{len(benches)} verified  ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
