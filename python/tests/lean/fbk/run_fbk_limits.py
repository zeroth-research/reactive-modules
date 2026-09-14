#!/usr/bin/env python3
"""Run the `--fbk-proveit` route over the *limit matrix's own invariants*.

    uv run python tests/lean/fbk/run_fbk_limits.py --ltl <dir> [--ic3ia P]

`tests/limits/cases.py` gives every case an `--invariant` that verith's own
route proves inductive.  Handed to `--safety` instead, each becomes a safety
property for ic3ia to rediscover an invariant for -- so this sweep asks the
same 77 cases the limit matrix asks, through the other certification route.

Where `run_fbk.py` runs a curated probe table with an expected verdict per
probe, this one is a *measurement*: it reports what each case's invariant
does on this route, and the interesting number is how many never reach
ic3ia at all.  Distinct `(module, invariant)` pairs only -- 23 countdown
cases share one invariant, and re-running it 23 times measures nothing.

Probes run strictly sequentially, sharing one `lean-ltl-certifying` build
directory, for the reason `run_fbk.py` gives.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "limits"))

from cases import CASES  # noqa: E402
from run_fbk import probe  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ltl", required=True, help="lean-ltl-certifying checkout")
    p.add_argument("--mods", default=str(HERE.parent.parent / "limits" / "mods"),
                   help="limit-probe fixture directory")
    p.add_argument("--ic3ia", default=os.environ.get("IC3IA"),
                   help="ic3ia binary (default: $IC3IA, else `ic3ia` on PATH)")
    p.add_argument("--out", default="/tmp/fbk-limits.noindex",
                   help="where projects go (keep the .noindex suffix on macOS)")
    p.add_argument("--only", help="run only cases whose name matches this regex")
    p.add_argument("--no-check", action="store_true",
                   help="skip the separate Lean check of each certificate")
    p.add_argument("--timeout", type=int, default=100, help="per-command seconds")
    p.add_argument("--json", help="write the full result table here")
    opts = p.parse_args()

    from zrth.lean.fbk_proveit import ProveItError, resolve_ic3ia
    try:
        opts.ic3ia = resolve_ic3ia(opts.ic3ia)
    except ProveItError as e:
        p.error(str(e))

    # Distinct (module, invariant), remembering every case that shares one.
    pairs: dict[tuple[str, str], list[str]] = {}
    for c in CASES:
        if c.get("inv"):
            pairs.setdefault((c["mod"], c["inv"]), []).append(c["name"])
    todo = [
        (names[0], mod, inv, names)
        for (mod, inv), names in pairs.items()
        if not opts.only or re.search(opts.only, names[0])
    ]

    covered = sum(len(n) for _, _, _, n in todo)
    print(f"{len(todo)} distinct (module, invariant) pairs from {covered} cases, "
          f"sequential\n")
    print(f"{'case':<20} {'module':<16} {'verdict':<10} {'':>5}  note")
    results = []
    for name, mod, inv, names in todo:
        r = probe(name, mod, inv, None, opts)
        r["cases"] = names
        results.append(r)
        extra = f" (+{len(names) - 1} more)" if len(names) > 1 else ""
        print(f"{name + extra:<20} {mod:<16} {r['verdict']:<10} "
              f"{r['seconds']:>5.0f}s  {r['detail'][:60]}", flush=True)

    tally = Counter(r["verdict"] for r in results)
    print("\n" + ", ".join(f"{v}: {n}" for v, n in tally.most_common()))
    cert = sum(len(r["cases"]) for r in results if r["verdict"] == "certified")
    print(f"{cert}/{covered} cases have an invariant this route certifies")
    if opts.json:
        Path(opts.json).write_text(json.dumps(results, indent=2))
        print(f"wrote {opts.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
