#!/usr/bin/env python3
"""Run the `verith --fbk-proveit` probe sweep, one probe at a time.

    uv run python tests/lean/fbk/run_fbk.py --mods <dir> --ltl <dir> [--ic3ia P]

`--mods` is the limit-probe fixture directory (the `m_*.py` modules that
`tests/limits/` measures); `--ltl` is a `lean-ltl-certifying` checkout.
See README.md for the cold start, including how to get both.

Probes run strictly sequentially.  They share one `lean-ltl-certifying`
build directory, and lake takes one writer -- two runners at once serve
each other's oleans and the symptom is not a clean failure.

By default each probe is also *checked*: the route itself deliberately
does not run Lean on the certificate it installs (it imports
`LTLCertifying.*` and `Smt`, which the generated project does not
provide), so `proveit.py -c` is invoked separately on the same model to
get a Lean verdict.  `--no-check` reports only what the route itself
concluded, at about a third of the wall clock.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from probes import PROBES, REJECTED  # noqa: E402


def _run(cmd, timeout):
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return -1, "<timeout>"


def classify(out: str) -> tuple[str, str]:
    """What the route concluded, as (verdict, detail)."""
    if "Installed certificate" in out:
        return "certified", ""
    if "UNSAFE" in out:
        return "unsafe", "counterexample"
    if "did not prove" in out:
        return "unknown", "ic3ia could not decide"
    m = re.search(r"^error: --fbk-proveit: (.*)$", out, re.M)
    if m:
        return "abort", m.group(1)[:100]
    if out == "<timeout>":
        return "timeout", ""
    return "error", (out.strip().splitlines() or [""])[-1][:100]


def module_path(mod: str, mods_dir) -> Path:
    """`TESTS/<name>` is `tests/fixtures/<name>.py`, anything else `--mods`.

    The limit harness spells it the same way (`run_limits.module_path`), so
    a probe can name either set.
    """
    if mod.startswith("TESTS/"):
        return HERE.parent.parent / "fixtures" / f"{mod.split('/', 1)[1]}.py"
    return Path(mods_dir) / f"{mod}.py"


def probe(name, mod, prop, expect, opts) -> dict:
    out_dir = Path(opts.out) / name
    shutil.rmtree(out_dir, ignore_errors=True)
    started = time.monotonic()

    cmd = [
        "uv", "run", "verith", str(module_path(mod, opts.mods)),
        "--safety", prop, "-o", str(out_dir), "-p", name,
        "--fbk-proveit", str(opts.ltl),
    ]
    if opts.ic3ia:
        cmd += ["--ic3ia", opts.ic3ia]
    _, out = _run(cmd, opts.timeout)
    verdict, detail = classify(out)

    # The route installs the certificate without checking it; ask Lean.
    if verdict == "certified" and not opts.no_check:
        model = out_dir / name / "ProveIt" / f"{name}NA.lean"
        chk = [
            sys.executable, str(Path(opts.ltl) / "proveit.py"), str(model),
            "-o", str(out_dir / "checked.lean"), "-c",
        ]
        if opts.ic3ia:
            chk += ["--ic3ia", opts.ic3ia]
        _, cout = _run(chk, opts.timeout)
        if "Successfully certified" not in cout:
            errs = re.findall(r"error: (.*)", cout)
            verdict, detail = "lean-fail", (errs[0][:100] if errs else "")

    return {
        "name": name, "module": mod, "property": prop,
        "expect": expect, "verdict": verdict, "detail": detail,
        "seconds": round(time.monotonic() - started, 1),
    }


def screen(opts) -> int:
    """Which modules the NA encoding refuses, and whether it is still the
    same set.  Needs no Lean, no ic3ia and no lean-ltl-certifying."""
    sys.path.insert(0, str(Path(opts.mods).resolve()))
    from zrth.lean.common import LeanContext
    from zrth.lean.project import load_module_from_file
    from zrth.lean.translate.fbk import NAUnsupported, check_na_supported

    accepted, rejected, surprises = [], {}, []
    for path in sorted(Path(opts.mods).glob("m_*.py")):
        try:
            check_na_supported(LeanContext(load_module_from_file(str(path))))
            accepted.append(path.stem)
        except NAUnsupported as e:
            rejected[path.stem] = str(e)
        except Exception as e:                       # a broken fixture
            rejected[path.stem] = f"[{type(e).__name__}] {e}"

    print(f"accepted: {len(accepted)}   rejected: {len(rejected)}")
    for name in sorted(accepted):
        if name in REJECTED:
            surprises.append(f"{name}: was rejected ({REJECTED[name][0]}), now accepted")
        print(f"  ok      {name}")
    for name, why in sorted(rejected.items()):
        known = REJECTED.get(name)
        mark = "  " if known else "!!"
        if not known:
            surprises.append(f"{name}: newly rejected -- {why[:80]}")
        print(f"{mark}reject  {name:14} {why[:90]}")

    if surprises:
        print("\nSURPRISES (probes.py::REJECTED is out of date):")
        for s in surprises:
            print(f"  {s}")
    return 1 if surprises else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mods", required=True, help="limit-probe fixture directory")
    p.add_argument("--ltl", help="lean-ltl-certifying checkout")
    p.add_argument("--ic3ia", default=os.environ.get("IC3IA"),
                   help="ic3ia binary (default: $IC3IA, else `ic3ia` on PATH)")
    p.add_argument("--out", default="/tmp/fbk-sweep", help="where projects go")
    p.add_argument("--only", help="run only probes whose name matches this regex")
    p.add_argument("--no-check", action="store_true",
                   help="skip the separate Lean check of each certificate")
    p.add_argument("--timeout", type=int, default=900, help="per-command seconds")
    p.add_argument("--json", help="write the full result table here")
    p.add_argument("--screen", action="store_true",
                   help="only report which modules the encoding accepts")
    opts = p.parse_args()

    if opts.screen:
        return screen(opts)
    if not opts.ltl:
        p.error("--ltl is required unless --screen is given")

    # The same resolution `verith` does, so that the separate `proveit.py -c`
    # invocation below accepts a build *directory* too -- otherwise it fails
    # there with no `error:` line and the probe looks like a Lean failure.
    from zrth.lean.fbk_proveit import ProveItError, resolve_ic3ia
    try:
        opts.ic3ia = resolve_ic3ia(opts.ic3ia)
    except ProveItError as e:
        p.error(str(e))

    selected = [c for c in PROBES if not opts.only or re.search(opts.only, c[0])]
    print(f"{len(selected)} probe(s), sequential"
          f"{'' if not opts.no_check else ', route only (no Lean check)'}\n")
    print(f"{'probe':<14} {'module':<12} {'verdict':<10} {'':>5}  note")

    results, surprises = [], []
    for name, mod, prop, expect in selected:
        r = probe(name, mod, prop, expect, opts)
        results.append(r)
        ok = r["verdict"] == expect
        if not ok:
            surprises.append(r)
        print(f"{r['name']:<14} {r['module']:<12} {r['verdict']:<10} "
              f"{r['seconds']:>5.0f}s  {'' if ok else f'EXPECTED {expect}. '}"
              f"{r['detail']}")

    print(f"\n{sum(1 for r in results if r['verdict'] == r['expect'])}/{len(results)} as expected")
    if surprises:
        print("\nSURPRISES:")
        for r in surprises:
            print(f"  {r['name']}: expected {r['expect']}, got {r['verdict']}  {r['detail']}")
    if opts.json:
        Path(opts.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {opts.json}")
    return 1 if surprises else 0


if __name__ == "__main__":
    sys.exit(main())
