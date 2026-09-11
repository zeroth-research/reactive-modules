#!/usr/bin/env python3
"""Run the verith limit probe: one full Lean project per case, shared .lake.

    uv run python run_limits.py [name ...]

Every project gets `.lake` symlinked to a single shared build dir whose
`packages` points at the already-built Mathlib/cslib/lean-smt of
python/tests/lean. Core, LeanAI and ZerothHammer are byte-identical across
projects, so they are built once and hit the cache afterwards; only
System/* and Certificate/* recompile per case.

Verdicts land in $VERITH_LIMITS_WORK/results.json (default
/tmp/verith-limits); a table is printed as it goes.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SP = Path(__file__).parent                  # python/tests/limits
PY = SP.parent.parent                       # python/
REPO = PY.parent
WARM = PY / "tests" / "lean"                # the already-built Mathlib
MODS = SP / "mods"

# Generated projects and their shared build dir are build artifacts, so they
# live outside the tree. Override with VERITH_LIMITS_WORK to put them
# elsewhere -- see the single-writer warning in README.md.
#
# The `.noindex` suffix is load-bearing on macOS: Spotlight will happily
# index tens of thousands of freshly written `.olean` files, and
# `spotlightknowledged` pinning a core turns a 9 s case into a 928 s one.
# A directory whose name ends in `.noindex` is skipped. Keep the suffix on
# any path passed through VERITH_LIMITS_WORK too.
WORK = Path(os.environ.get("VERITH_LIMITS_WORK", "/tmp/verith-limits.noindex"))
PROJECTS = WORK / "projects"
SHARED = WORK / "shared_lake"
MANIFEST = SP / "lake-manifest.json"
WORK.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SP))
from cases import CASES  # noqa: E402

DEFAULT_TIMEOUT = 900


def module_path(mod: str) -> Path:
    if mod.startswith("TESTS/"):
        return PY / "tests" / "fixtures" / f"{mod.split('/', 1)[1]}.py"
    return MODS / f"{mod}.py"


def run_verith(case) -> dict:
    out = PROJECTS / case["name"]
    if out.exists():
        shutil.rmtree(out)
    cmd = ["uv", "run", "verith", str(module_path(case["mod"]))]
    if case.get("P"):
        # Every case here is a Buchi property: `P` comes with a ranking.
        cmd += ["--buchi", case["P"]]
    if case.get("inv"):
        cmd += ["--invariant", case["inv"]]
    if case.get("rank"):
        cmd += ["--ranking", case["rank"]]
    if case.get("pre"):
        cmd += ["--pre", case["pre"]]
    cmd += ["-o", str(out), "-p", "Rea"]
    cmd += os.environ.get("VERITH_EXTRA", "").split()
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=PY, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return dict(ok=False, secs=time.time() - t0, err="verith timed out (600s)")
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()
        err = " | ".join(ln.strip() for ln in tail[-3:])
        return dict(ok=False, secs=time.time() - t0, err=err)
    return dict(ok=True, secs=time.time() - t0, err="")


_ERR = re.compile(r"^(?:error|warning): (?P<file>[^:]+\.lean):(\d+):(\d+): (?P<msg>.*)$")
_FAILED_TGT = re.compile(r"^✖ \[[\d/]+\] Built (?P<t>\S+)")


def ensure_shared_lake() -> None:
    """Point the shared build dir's `packages` at the already-built ones.

    Without this every project tries to fetch and build Mathlib for itself,
    which shows up as `external command 'git' exited with code 128` on every
    case at once. `packages` is only ever read, so several build dirs may
    point at it -- unlike the build dir itself, which takes one writer.
    """
    warm = WARM / ".lake" / "packages"
    if not warm.is_dir():
        raise SystemExit(
            f"error: {warm} does not exist. Build the warm project first:\n"
            f"    cd {WARM} && lake build"
        )
    SHARED.mkdir(parents=True, exist_ok=True)
    link = SHARED / "packages"
    if link.is_symlink():
        if link.readlink() == warm:
            return
        link.unlink()
    elif link.exists():
        raise SystemExit(f"error: {link} exists and is not a symlink")
    link.symlink_to(warm)


def run_lake(case) -> dict:
    proj = PROJECTS / case["name"] / "Rea"
    lake = proj / ".lake"
    if lake.exists() or lake.is_symlink():
        lake.unlink()
    lake.symlink_to(SHARED)
    shutil.copy2(MANIFEST, proj / "lake-manifest.json")

    t0 = time.time()
    try:
        r = subprocess.run(
            ["lake", "build"], cwd=proj, capture_output=True, text=True,
            timeout=case.get("timeout", DEFAULT_TIMEOUT),
        )
    except subprocess.TimeoutExpired:
        return dict(ok=False, secs=time.time() - t0, targets=["<timeout>"],
                    errors=[f"lake build timed out ({case.get('timeout', DEFAULT_TIMEOUT)}s)"],
                    sorries=[])
    out = r.stdout + "\n" + r.stderr
    lines = out.splitlines()

    failed = [m.group("t") for ln in lines if (m := _FAILED_TGT.match(ln))]
    errors, sorries = [], []
    for i, ln in enumerate(lines):
        if ln.startswith("error: ") and ".lean:" in ln:
            m = _ERR.match(ln)
            if m:
                # first message line, plus the next line if the message is empty
                msg = m.group("msg").strip()
                if not msg and i + 1 < len(lines):
                    msg = lines[i + 1].strip()
                errors.append(f"{Path(m.group('file')).name}: {msg}")
            else:
                errors.append(ln[7:])
        elif ln.startswith("error: ") and "Lean exited" not in ln:
            errors.append(ln[7:])
        if "declaration uses 'sorry'" in ln:
            m = _ERR.match(ln)
            sorries.append(
                f"{Path(m.group('file')).name}:{m.group(2)}" if m else ln.strip()
            )
    return dict(ok=r.returncode == 0 and not sorries, raw_ok=r.returncode == 0,
                secs=time.time() - t0, targets=failed,
                errors=errors[:6], sorries=sorries[:8])


def verdict(case, gen, build) -> str:
    if not gen["ok"]:
        return "GEN-FAIL"
    if build.get("sorries") and build.get("raw_ok"):
        return "SORRY"
    if not build["ok"]:
        return "PROOF-FAIL" if not build["targets"] or "Certificate" in " ".join(build["targets"]) else "BUILD-FAIL"
    return "VERIFIED"


def main() -> None:
    only = set(sys.argv[1:])
    PROJECTS.mkdir(parents=True, exist_ok=True)
    ensure_shared_lake()
    results_path = WORK / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    todo = [c for c in CASES if not only or c["name"] in only]
    print(f"{len(todo)} cases\n")
    for n, case in enumerate(todo, 1):
        name = case["name"]
        print(f"[{n}/{len(todo)}] {name:<16} ", end="", flush=True)
        gen = run_verith(case)
        build = (dict(ok=False, secs=0.0, targets=[], errors=["(not built: generation failed)"], sorries=[])
                 if not gen["ok"] else run_lake(case))
        v = verdict(case, gen, build)
        results[name] = dict(case={k: v2 for k, v2 in case.items() if k != "timeout"},
                             gen=gen, build=build, verdict=v)
        results_path.write_text(json.dumps(results, indent=1))
        flag = "" if (v == "VERIFIED") == (case["expect"] == "ok") else "  <-- unexpected"
        print(f"{v:<11} gen {gen['secs']:5.1f}s  build {build['secs']:6.1f}s{flag}")
        if v == "GEN-FAIL":
            print(f"      {gen['err'][:200]}")
        for e in (build["errors"] or [])[:2]:
            print(f"      {e[:150]}")
    print("\nwrote", results_path)


if __name__ == "__main__":
    main()
