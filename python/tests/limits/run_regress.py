#!/usr/bin/env python3
"""Regression check over a fixed baseline, writing to its own results file.

`run_limits.py` runs whatever is in `cases.py`, so it cannot be used as a
regression check while someone is adding cases. This runs exactly the case
names recorded in a baseline file, compares each verdict against it, and
flags changes in both directions.

    uv run python tests/limits/run_regress.py              # against baseline.json
    uv run python tests/limits/run_regress.py other.json
"""
import json
import sys
import time
from pathlib import Path

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
import run_limits  # noqa: E402
from run_limits import run_verith, run_lake, verdict  # noqa: E402
from cases import CASES  # noqa: E402

# Isolate from anything else using the harness. Every project symlinks its
# `.lake` to one shared build dir, and the module names (`System.Data`, ...)
# are the same in every project, so two processes building different projects
# into one build dir overwrite each other's oleans -- which shows up as
# heartbeat timeouts and `unknown constant 'hrank'`, not as a clean failure.
# `packages` still points at the one warm Mathlib; that is read-only.
run_limits.SHARED = run_limits.WORK / "shared_lake_regress"
run_limits.PROJECTS = run_limits.WORK / "projects_regress"
PROJECTS = run_limits.PROJECTS


def main() -> None:
    baseline_path = SP / (sys.argv[1] if len(sys.argv) > 1 else "baseline.json")
    baseline = json.loads(baseline_path.read_text())
    by_name = {c["name"]: c for c in CASES}
    todo = [by_name[n] for n in baseline if n in by_name]
    missing = [n for n in baseline if n not in by_name]
    if missing:
        print(f"!! baseline names no longer in cases.py: {missing}")

    PROJECTS.mkdir(parents=True, exist_ok=True)
    run_limits.ensure_shared_lake()
    print(f"{len(todo)} baseline cases\n")
    out, regressions, fixes = {}, [], []
    t0 = time.time()
    for n, case in enumerate(todo, 1):
        name = case["name"]
        print(f"[{n}/{len(todo)}] {name:<20} ", end="", flush=True)
        gen = run_verith(case)
        build = (dict(ok=False, secs=0.0, targets=[],
                      errors=["(not built: generation failed)"], sorries=[])
                 if not gen["ok"] else run_lake(case))
        v = verdict(case, gen, build)
        was = baseline[name]["verdict"]
        flag = ""
        if v != was:
            flag = f"  <-- {was} -> {v}"
            (fixes if v == "VERIFIED" else regressions).append(f"{name}: {was} -> {v}")
        print(f"{v:<12} {build['secs']:6.1f}s{flag}")
        for e in build.get("errors", [])[:2]:
            print(f"      {e}")
        out[name] = dict(case=case, gen=gen, build=build, verdict=v, was=was)

    (run_limits.WORK / "results_regress.json").write_text(json.dumps(out, indent=1))
    print(f"\n{len(todo)} cases in {time.time() - t0:.0f}s")
    print(f"REGRESSIONS ({len(regressions)}): {regressions or 'none'}")
    print(f"NEWLY GREEN ({len(fixes)}): {fixes or 'none'}")


if __name__ == "__main__":
    main()
