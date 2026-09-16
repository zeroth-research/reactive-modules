"""One SV-COMP termination benchmark, as a module `verith` can load.

The corpus in `benchmarks/svcomp/dsl/` is written for the Farkas pipeline:
each file exposes a `BENCH` whose `build()` returns
`(module, ctrl_by_name, extl_by_name)`, and the package is imported
relatively, so `verith`'s `load_module_from_file` cannot read one of those
files directly -- it wants a no-argument `module()` in a file it can exec on
its own.

This is that file, for every benchmark at once: which one is named by
`$SVCOMP_BENCH`, so one checked-in adapter serves all 57 and the command
that measures a case is a command that can be pasted into a terminal.

    SVCOMP_BENCH=AliasDarteFeautrierGonnord-SAS2010-easy1 \
        uv run verith tests/bench_matrix/svcomp_mod.py --buchi '(<= 40 s0)' ...

The benchmark's own name is the key (`Bench.name`, matching the `.c` stem),
not the Python module's, because that is what the corpus and the results
table call it.
"""
from __future__ import annotations

import os


def module():
    name = os.environ.get("SVCOMP_BENCH")
    if not name:
        raise SystemExit(
            "SVCOMP_BENCH is not set: it names the benchmark this adapter "
            "should load, e.g. SVCOMP_BENCH=ColonSipma-TACAS2001-Fig1"
        )
    from benchmarks.svcomp import discover

    for bench in discover():
        if bench.name == name:
            return bench.build()[0]
    raise SystemExit(f"SVCOMP_BENCH={name!r}: no such benchmark in benchmarks/svcomp/dsl/")
