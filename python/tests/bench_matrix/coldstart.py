#!/usr/bin/env python3
"""What the matrix pays before it measures anything: the cold start.

    uv run python tests/bench_matrix/coldstart.py

Three numbers, and they answer different questions.

`first_build_s` is the one that matters operationally: the first project
built into an **empty** build dir, which compiles Core, LeanAI and
ZerothHammer against Mathlib and caches the oleans every later project then
hits. It is paid once per build dir, not once per case, and it is why the
first row of a fresh pass looks like a catastrophe next to its neighbours.
Measured into a build dir of its own, so the matrix's stays warm.

`warm_build_s` is the same project again with that cache filled -- the
per-cell cost the table reports.

`noop_build_s` is `lake build` with nothing to do, the floor under every
number here.

Mathlib itself is **not** compiled: `tests/lean/.lake/packages` holds it
already, and it arrives from the mathlib4 olean cache
(`lake exe cache get`, an Azure blob store) rather than from a local
build. That download-and-decompress, not a compile, is the real cold start
for Mathlib, and it is the one thing here that is a property of the network
rather than of this machine -- so it is reported as what the tree documents
rather than re-measured.

Run it on a quiet machine, and not while `run_matrix.py` is running: two
lake builds at once make both numbers meaningless.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SP = Path(__file__).resolve().parent
PY = SP.parent.parent
WARM = PY / "tests" / "lean"
MANIFEST = PY / "tests" / "limits" / "lake-manifest.json"
WORK = Path(os.environ.get("VERITH_BENCH_WORK", "/tmp/verith-bench.noindex"))
RESULTS = WORK / "results.json"

# Its own build dir: filling this one must not warm the matrix's.
COLD = WORK / "coldstart.noindex"

# `../limits/README.md` records this same measurement at 654 s. It does not
# reproduce: measured here the build dir held nothing but the `packages`
# symlink beforehand and exactly the targets below afterwards. The older
# figure was most likely a pass whose `packages` was not yet populated, or
# one indexed by Spotlight -- the same README documents a 9 s case that
# Spotlight turned into 928 s. Stated rather than silently replaced, because
# the two numbers were taken on different days and only this one has its
# target list attached.
FIRST_BUILD_NOTE = (
    "{n} targets &mdash; <code>Core.*</code>, <code>System.*</code>, "
    "<code>ZerothHammer</code>, <code>Certificate.Data</code> &mdash; into a "
    "build dir holding nothing but the <code>packages</code> symlink. This is "
    "the shared cache filling, paid once per build dir. Routes that declare a "
    "library of their own pay a further one-time cost on top: "
    "<code>LeanAI/</code> for <code>ai</code> and <code>ai-cegar</code>, "
    "<code>ProveIt/</code> for <code>fbk-proveit</code>. "
    "<code>../limits/README.md</code> records 654 s for this measurement; it "
    "did not reproduce here, and that figure has no target list attached.")

MATHLIB_NOTE = (
    "not compiled here &mdash; <code>tests/lean/.lake/packages</code> is already "
    "populated, and Mathlib arrives from the mathlib4 olean cache "
    "(<code>lake exe cache get</code>, an Azure blob store) rather than from a "
    "local build. <code>tests/BENCHMARKS.md</code> records that first "
    "<code>(cd tests/lean &amp;&amp; lake build)</code> at <b>~1 hour</b> cold, which is "
    "mostly download and decompression of ~8,000 oleans; the "
    "<code>lean-ltl-certifying</code> checkout that <code>fbk-proveit</code> needs "
    "adds <b>~80 s</b> on top, because it pins the same package set and so shares "
    "the very same <code>packages</code> directory."
)

EXPLAIN = (
    "Read the first two rows together: the pass compiles Core, LeanAI and "
    "ZerothHammer once, and every project after that hits the cache, which is "
    "the whole reason one shared build dir is worth the serialisation it forces. "
    "The <code>--infer fbk-proveit</code> column has a second such cost of its "
    "own &mdash; <code>LTLCertifying.*</code> and <code>lean2vmt</code>, built "
    "into the checkout's own <code>.lake/build</code> the first time the route "
    "runs, and shared by every probe after."
)


def run(cmd, cwd, timeout=3600):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return time.time() - t0, r


def main() -> None:
    if not (WARM / ".lake" / "packages").is_dir():
        raise SystemExit(f"error: {WARM / '.lake/packages'} is not populated")

    proj_root = WORK / "coldstart_project"
    shutil.rmtree(proj_root, ignore_errors=True)
    shutil.rmtree(COLD, ignore_errors=True)

    print("generating one project (m_countdown, --infer nuterm) ...")
    _, r = run(["uv", "run", "verith", "tests/limits/mods/m_countdown.py",
                "--buchi", "(= s0 0)", "--infer", "nuterm",
                "--artifacts", "reset", "-o", str(proj_root), "-p", "Rea"], PY, 600)
    if r.returncode != 0:
        raise SystemExit(f"generation failed:\n{r.stderr[-2000:]}")

    proj = proj_root / "Rea"
    COLD.mkdir(parents=True, exist_ok=True)
    (COLD / "packages").symlink_to(WARM / ".lake" / "packages")
    (proj / ".lake").unlink(missing_ok=True)
    (proj / ".lake").symlink_to(COLD)
    shutil.copy2(MANIFEST, proj / "lake-manifest.json")

    print(f"cold build into an empty build dir ({COLD}) ... this is the slow one")
    cold_s, r = run(["lake", "build"], proj)
    # What it compiled, not just how long: the number is only meaningful
    # against the target list, and the list is route-dependent -- `LeanAI/`
    # and `ProveIt/` are copied into every project but build only for the
    # routes that declare them as a `lean_lib`.
    targets = re.findall(r"\] Built (\S+)", r.stdout + r.stderr)
    print(f"  {cold_s:.1f}s  rc={r.returncode}")
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])

    print("no-op rebuild ...")
    noop_s, _ = run(["lake", "build"], proj)
    print(f"  {noop_s:.1f}s")

    print("warm build: the same project regenerated, cache already filled ...")
    shutil.rmtree(proj_root, ignore_errors=True)
    run(["uv", "run", "verith", "tests/limits/mods/m_countdown.py",
         "--buchi", "(= s0 0)", "--infer", "nuterm", "--artifacts", "reset",
         "-o", str(proj_root), "-p", "Rea"], PY, 600)
    (proj / ".lake").unlink(missing_ok=True)
    (proj / ".lake").symlink_to(COLD)
    shutil.copy2(MANIFEST, proj / "lake-manifest.json")
    warm_s, _ = run(["lake", "build"], proj)
    print(f"  {warm_s:.1f}s")

    out = dict(first_build_s=cold_s, noop_build_s=noop_s, warm_build_s=warm_s,
               first_build_targets=sorted(targets),
               mathlib_note=MATHLIB_NOTE, explain=EXPLAIN,
               first_build_note=FIRST_BUILD_NOTE.format(n=len(targets)),
               measured_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"meta": {}}
    data.setdefault("meta", {})["coldstart"] = out
    RESULTS.write_text(json.dumps(data, indent=1))
    shutil.rmtree(proj_root, ignore_errors=True)
    shutil.rmtree(COLD, ignore_errors=True)
    print(f"\nwrote meta.coldstart to {RESULTS}")
    print(json.dumps({k: v for k, v in out.items()
                      if k.endswith("_s") or k == "measured_at"}, indent=1))


if __name__ == "__main__":
    main()
