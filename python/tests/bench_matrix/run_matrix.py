#!/usr/bin/env python3
"""Run every `--infer` route over every (benchmark, property) and record it.

    uv run python tests/bench_matrix/run_matrix.py                  # everything
    uv run python tests/bench_matrix/run_matrix.py --suites limits tests
    uv run python tests/bench_matrix/run_matrix.py --routes nuterm sygus
    uv run python tests/bench_matrix/run_matrix.py --only Countdown

One `(row, route)` pair is one `uv run verith` followed by one `lake build`,
and both are timed -- the two numbers the table reports. Results land in
`$VERITH_BENCH_WORK/results.json` as they are produced, and a pair already
in that file is skipped, so an interrupted pass resumes and a single route
can be re-measured without disturbing its neighbours (`--redo` forces one).

**This runs strictly serially and must.** Every generated project symlinks
its `.lake` to one shared build dir so Mathlib is never rebuilt, and module
names (`Certificate.Data`, ...) are identical across projects: two
concurrent runs overwrite each other's oleans and the symptom is
`unknown constant 'hrank'`, which reads exactly like a codegen bug. See
`../limits/README.md`, which lost an hour to it.

Timings are wall clock and so are only as quiet as the machine. Another Lean
build anywhere -- or Spotlight indexing the tens of thousands of `.olean`
files a pass writes -- inflates them wildly and unevenly. Hence `.noindex`
on the work directory, which is the documented way to make Spotlight skip
one; keep the suffix on anything passed through `VERITH_BENCH_WORK`.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

SP = Path(__file__).resolve().parent
PY = SP.parent.parent
REPO = PY.parent
WARM = PY / "tests" / "lean"                       # the already-built Mathlib
MANIFEST = PY / "tests" / "limits" / "lake-manifest.json"

sys.path.insert(0, str(PY))
sys.path.insert(0, str(SP))
from suites import all_rows, pairs, routes  # noqa: E402

WORK = Path(os.environ.get("VERITH_BENCH_WORK", "/tmp/verith-bench.noindex"))
PROJECTS = WORK / "projects"
SHARED = WORK / "shared_lake"
RESULTS = WORK / "results.json"

GEN_TIMEOUT = 300
BUILD_TIMEOUT = 420

# The two paths `--infer fbk-proveit` needs, and the MathSAT bindings its
# `vmt2lean.py` imports. Overridable, but defaulted so a plain run measures
# all seven columns rather than silently six.
PROVEIT_DIR = os.environ.get("VERITH_PROVEIT_DIR",
                             str(Path.home() / "zeroth/proof-prototyping/lean-ltl-certifying"))
IC3IA = os.environ.get("VERITH_IC3IA", str(Path.home() / "zeroth/fbk/ic3ia/build"))
MATHSAT_PY = os.environ.get("VERITH_MATHSAT_PY", str(Path.home() / "zeroth/fbk/mathsat/python"))
KEY_FILE = REPO / "CLAUDE_KEY.txt"
MODEL = os.environ.get("VERITH_MODEL", "")     # "" = verith's own default


# ══════════════════════════════════════════════════════════════════════════
# The shared build dir
# ══════════════════════════════════════════════════════════════════════════

def ensure_shared_lake() -> None:
    """Point the shared build dir's `packages` at the already-built ones.

    Without this every project resolves Mathlib for itself, which shows up
    as `external command 'git' exited with code 128` on every row at once.
    `packages` is only ever read, so several build dirs may point at it --
    unlike the build dir itself, which takes one writer."""
    warm = WARM / ".lake" / "packages"
    if not warm.is_dir():
        raise SystemExit(f"error: {warm} does not exist. Build it first:\n"
                         f"    (cd {WARM} && lake build)")
    SHARED.mkdir(parents=True, exist_ok=True)
    link = SHARED / "packages"
    if link.is_symlink():
        if link.readlink() == warm:
            return
        link.unlink()
    elif link.exists():
        raise SystemExit(f"error: {link} exists and is not a symlink")
    link.symlink_to(warm)


def write_manifest(proj: Path, route_name: str) -> None:
    """The manifest the project builds against.

    `--infer fbk-proveit` writes a lakefile with a *path* require on the
    `lean-ltl-certifying` checkout, and lake refuses a dependency the
    manifest does not list. It needs no fetch -- it is a local directory --
    so the entry is added here rather than by a `lake update`, which would
    re-resolve the other thirteen packages."""
    manifest = json.loads(MANIFEST.read_text())
    if route_name == "fbk-proveit":
        manifest["packages"].append({
            "type": "path", "scope": "", "name": "LTL_Certifying",
            "manifestFile": "lake-manifest.json", "inherited": False,
            "dir": PROVEIT_DIR, "configFile": "lakefile.toml",
        })
    (proj / "lake-manifest.json").write_text(json.dumps(manifest, indent=2))


# ══════════════════════════════════════════════════════════════════════════
# One pair
# ══════════════════════════════════════════════════════════════════════════

def command(row, route, out: Path) -> list[str]:
    """The `uv run verith` this pair is, as a list. Also what the page shows."""
    cmd = ["uv", "run", "verith", rel(row.module),
           f"--{row.kind}", row.prop, *route.args,
           "--artifacts", "reset", "-o", str(out), "-p", "Rea"]
    if route.needs_llm and MODEL:
        cmd += ["--model", MODEL]
    return cmd


def rel(p: Path) -> str:
    """A module path as it is written from `python/`, which is where a run is."""
    try:
        return str(Path(p).resolve().relative_to(PY))
    except ValueError:
        return str(p)


def shown(row, route, out: Path) -> str:
    """The command as a person would paste it: env prefixes, quoted property."""
    import shlex
    pre = [f"{k}={v}" for k, v in sorted(row.env.items())]
    if route.needs_llm:
        pre.append("ANTHROPIC_API_KEY=$(cat CLAUDE_KEY.txt)")
    if "PYTHONPATH" in route.needs_env:
        pre.append(f"PYTHONPATH={MATHSAT_PY}")
    return " ".join(pre + [shlex.quote(c) for c in command(row, route, out)])


def child_env(row, route) -> dict:
    env = dict(os.environ, **row.env)
    env["PYTHONWARNINGS"] = "ignore"
    if route.needs_llm:
        if not env.get("ANTHROPIC_API_KEY") and KEY_FILE.exists():
            env["ANTHROPIC_API_KEY"] = KEY_FILE.read_text().strip()
    if "PYTHONPATH" in route.needs_env:
        env["PYTHONPATH"] = MATHSAT_PY + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def run_capped(cmd, *, cwd, env=None, timeout: int):
    """`cmd` under a wall-clock cap that takes its whole subtree with it.

    `subprocess.run(timeout=)` signals only the process it started, and here
    that process is a wrapper: `uv run` for generation, `lake` for the build.
    On a timeout the wrapper died and the real worker -- verith itself, or
    the `lean` processes under lake -- was reparented to init and kept
    running. One orphaned `--infer ai-cegar` cell held a core for 2h38m
    beside the pass that had already written it off as TIMEOUT, so every
    timing taken next to it is a timing of a busier machine than the page
    claims. The child therefore leads its own process group and the cap
    kills the group.

    Returns `(completed, timed_out)`; on a timeout the partial output is
    still returned, since a refusal is often printed before the hang.
    """
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err), False
    except subprocess.TimeoutExpired:
        out, err = _kill_group(proc)
        return subprocess.CompletedProcess(cmd, proc.returncode or -9, out, err), True


def _kill_group(proc):
    """SIGTERM `proc`'s group, SIGKILL whatever ignored that, and drain."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        try:
            proc.wait(timeout=10)
            break
        except subprocess.TimeoutExpired:
            continue
    try:
        # A survivor holding the pipe open would hang this, so it is capped
        # too; losing the tail of a killed cell's output costs nothing.
        return proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        return "", ""


def generate(row, route, out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    t0 = time.time()
    r, timed_out = run_capped(command(row, route, out), cwd=PY,
                              env=child_env(row, route), timeout=GEN_TIMEOUT)
    if timed_out:
        return dict(ok=False, secs=time.time() - t0,
                    err=f"verith timed out ({GEN_TIMEOUT}s)",
                    err_full=((r.stderr or r.stdout) or "").strip()[-4000:])
    if r.returncode != 0:
        # The diagnosis is often the *first* line of a refusal and the lines
        # after it are the hint, so a 3-line tail threw away the reason.
        # `err_full` keeps the whole thing for the page's detail panel.
        blob = (r.stderr or r.stdout).strip()
        tail = [ln.strip() for ln in blob.splitlines() if ln.strip()]
        return dict(ok=False, secs=time.time() - t0,
                    err=" | ".join(tail[-8:]), err_full=blob[-4000:])
    return dict(ok=True, secs=time.time() - t0, err="",
                inferred=inferred_from(r.stdout))


_PRED = re.compile(r"^\s*(?:\[[\w-]+\]\s*)?(inv|ranking)\s*:\s*(.+?)\s*$")


def inferred_from(stdout: str) -> dict:
    """What the route said it found -- the invariant and ranking it printed.

    Worth keeping beside the timing: two routes that both certify a row did
    not necessarily find the same certificate, and the table's `VERIFIED`
    cannot show that."""
    found: dict = {}
    for line in stdout.splitlines():
        m = _PRED.match(line)
        if m and m.group(1) not in found:
            found[m.group(1)] = m.group(2).strip("`")
    return found


# lake prints `declaration uses \`sorry\`` -- backticks, not the straight
# quotes an older harness looked for. And the shared `.lake` carries one
# upstream sorry of its own (lean-smt's `Smt/Reconstruct/BitVec/Bitblast`),
# so a sorry only counts when it is in a file this project owns.
_ERR = re.compile(r"^(?:error|warning): (?P<file>[^:]+\.lean):(\d+):(\d+):\s*(?P<msg>.*)$")
# A target that fails is reported as `✖ [n/m] Building X`; a finished one as
# `Built X`. Matching only the second recorded no failed target, ever, so
# `BUILD-FAIL` could not be told from `PROOF-FAIL`.
_FAILED = re.compile(r"^✖ \[[\d/]+\] Buil(?:t|ding) (?P<t>\S+)")
_SORRY = re.compile(r"declaration uses [`']sorry[`']")
OURS = ("Certificate/", "System/", "Core/", "LeanAI/", "ZerothHammer/", "ProveIt/")


def build(row, route, out: Path) -> dict:
    proj = out / "Rea"
    lake = proj / ".lake"
    if lake.exists() or lake.is_symlink():
        lake.unlink()
    lake.symlink_to(SHARED)
    write_manifest(proj, route.name)

    t0 = time.time()
    r, timed_out = run_capped(["lake", "build"], cwd=proj, timeout=BUILD_TIMEOUT)
    if timed_out:
        return dict(ok=False, raw_ok=False, secs=time.time() - t0,
                    targets=["<timeout>"], sorries=[],
                    errors=[f"lake build timed out ({BUILD_TIMEOUT}s)"])
    lines = (r.stdout + "\n" + r.stderr).splitlines()
    failed = [m.group("t") for ln in lines if (m := _FAILED.match(ln))]
    errors, sorries = [], []
    for i, ln in enumerate(lines):
        m = _ERR.match(ln)
        if _SORRY.search(ln):
            if m and any(s in m.group("file") for s in OURS):
                sorries.append(f"{Path(m.group('file')).name}:{m.group(2)}")
            continue
        if ln.startswith("error: "):
            if m:
                msg = m.group("msg").strip() or (lines[i + 1].strip()
                                                 if i + 1 < len(lines) else "")
                errors.append(f"{m.group('file')}: {msg}")
            elif "Lean exited" not in ln:
                errors.append(ln[7:])
    return dict(ok=r.returncode == 0 and not sorries, raw_ok=r.returncode == 0,
                secs=time.time() - t0, targets=failed,
                errors=errors[:6], sorries=sorries[:8])


# A route that produced no project did so for one of three reasons, and they
# are not the same measurement. `main` wraps most refusals under `error:
# --infer:`, but a route whose row sets `errors_self_named` names its own
# flag instead (`--fbk-proveit:`), so the prefix alone is not enough.
_REFUTED = re.compile(r"found a counterexample|\bUNSAFE\b|property does not hold")
_NO_CERT = re.compile(
    r"error: --infer|CEGAR failed after|obligation violated|found no ranking"
    r"|found no invariant|could not decide|cannot decide|--fbk-proveit:|--ic3ia:")


# What a certificate is made of. Everything else a project builds -- the five
# encodings under `System/`, `Core`, the hammer -- is the module, and a failure
# there is not the certificate's.
_CERT_FILES = {"Certificate.lean", "Data.lean", "Equivalence.lean"}


def failed_outside_certificate(bld) -> bool:
    """Whether something other than the certificate failed to build.

    A failed encoding takes the certificate down with it -- `Certificate`
    imports `System` -- so a failed `Certificate.*` target says nothing on its
    own; any failed target *outside* it does. Runs recorded before the target
    list was captured fall back on the file each error names."""
    targets = [t for t in bld.get("targets", []) if t != "<timeout>"]
    if targets:
        return any(not t.startswith("Certificate") for t in targets)
    files = [e.split(":", 1)[0] for e in bld.get("errors", [])]
    return any(Path(f).name not in _CERT_FILES for f in files if f.endswith(".lean"))


def obligations_only(bld) -> bool:
    """Whether every error is in `Certificate/Certificate.lean`: the
    obligations, with the predicates in `Data.lean` elaborated and the module
    built."""
    targets = [t for t in bld.get("targets", []) if t != "<timeout>"]
    files = [e.split(":", 1)[0] for e in bld.get("errors", [])]
    files = [f for f in files if f.endswith(".lean")]
    return (bool(files)
            and all(Path(f).name == "Certificate.lean" for f in files)
            and all(t == "Certificate.Certificate" for t in targets))


def verdict(gen, bld, route: str = "") -> str:
    """What this pair showed.

    A route that searched a shape and found it empty is the measurement, not
    a defect, so `NO-CERT` is kept apart from `GEN-FAIL`; and a route that
    *disproved* the property is a third thing again, which the deliberately
    false controls are there to produce. `SORRY` is what the `none` control
    is for -- the project is well-formed and the obligations are simply
    open."""
    if not gen["ok"]:
        err = gen["err"] + " " + gen.get("err_full", "")
        if "timed out" in gen["err"]:
            return "TIMEOUT"
        if _REFUTED.search(err):
            return "REFUTED"
        return "NO-CERT" if _NO_CERT.search(err) else "GEN-FAIL"
    if "<timeout>" in bld.get("targets", []):
        return "TIMEOUT"
    # Before the sorries: a module that does not build is the headline, and
    # a `sorry` in the certificate beside it is not what stopped the build.
    if not bld.get("raw_ok") and failed_outside_certificate(bld):
        return "BUILD-FAIL"
    # `none` supplies no predicate, and its obligations are still handed to
    # the tactics, which cannot close one over a `ranking := sorry` or an
    # `inv := True` the property does not follow from. A build that stops
    # there and nowhere else is that control's clean reading -- the module
    # generated, the project elaborated, the obligations open -- not a
    # failure of anything the column measures.
    if route == "none" and not bld.get("raw_ok") and obligations_only(bld):
        return "SORRY"
    if bld.get("sorries"):
        return "SORRY" if bld.get("raw_ok") else "SORRY+FAIL"
    if not bld["ok"]:
        return "PROOF-FAIL"
    return "VERIFIED"


# ══════════════════════════════════════════════════════════════════════════
# The pass
# ══════════════════════════════════════════════════════════════════════════

def machine() -> dict:
    """What ran this, so a timing can be read against it."""
    def sysctl(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:
            return ""

    def out(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:
            return ""

    mem = sysctl("hw.memsize")
    return dict(
        cpu=sysctl("machdep.cpu.brand_string") or platform.processor(),
        cores=sysctl("hw.ncpu") or str(os.cpu_count()),
        perf_cores=sysctl("hw.perflevel0.physicalcpu"),
        eff_cores=sysctl("hw.perflevel1.physicalcpu"),
        memory_gb=round(int(mem) / 1024**3) if mem.isdigit() else None,
        os=f"{platform.system()} {platform.release()}",
        os_product=out(["sw_vers", "-productVersion"]),
        os_build=out(["sw_vers", "-buildVersion"]),
        python=platform.python_version(),
        lake=out(["lake", "--version"]),
        toolchain=(WARM / "lean-toolchain").read_text().strip(),
        lake_packages_gb=round(int(out(["du", "-sk", str(WARM / ".lake")]).split()[0])
                               / 1024**2, 1) if out(["du", "-sk", str(WARM / ".lake")]) else None,
        packages={p["name"]: (p.get("inputRev") or p.get("rev", ""))[:40]
                  for p in json.loads(MANIFEST.read_text())["packages"]},
    )


def main() -> None:
    global RESULTS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", nargs="*", default=[], help="limits fbk tests svcomp")
    ap.add_argument("--routes", nargs="*", default=[], help="route names to measure")
    ap.add_argument("--only", nargs="*", default=[], help="substring match on benchmark")
    ap.add_argument("--redo", action="store_true", help="re-measure pairs already recorded")
    ap.add_argument("--no-build", action="store_true", help="generate only, skip lake")
    # A repeat measurement is a *new sample*, not an amendment to the old
    # one: it goes in its own file and `render.py` averages the files it is
    # given. Resuming an interrupted pass means pointing `--results` back at
    # that pass's file, which is what the default does for the first one.
    ap.add_argument("--results", default=str(RESULTS), metavar="PATH",
                    help="results file to write (and resume from)")
    ap.add_argument("--label", default="", metavar="NAME",
                    help="what to call this sample on the page (default: filename stem)")
    ap.add_argument("--reverdict", action="store_true",
                    help="recompute every recorded verdict from its stored gen/build "
                         "data and exit -- what to run after correcting `verdict()`, "
                         "so a rule change does not cost a re-measurement")
    ap.add_argument("--prune", action="store_true",
                    help="drop recorded rows and runs the suites no longer produce, "
                         "and exit -- what to run after `suites.py` stops emitting a row")
    args = ap.parse_args()
    RESULTS = Path(args.results)

    if args.prune:
        data = json.loads(RESULTS.read_text())
        keep = {r.key for r in all_rows()}
        gone = sorted(k for k in data["rows"] if k not in keep)
        runs = [p for p in data["runs"] if p.split("::")[0] not in keep]
        for k in gone:
            del data["rows"][k]
        for p in runs:
            del data["runs"][p]
        RESULTS.write_text(json.dumps(data, indent=1))
        for k in gone:
            print(f"  dropped {k}")
        print(f"{len(gone)} rows and {len(runs)} runs dropped; "
              f"{len(data['rows'])} rows and {len(data['runs'])} runs kept")
        return

    if args.reverdict:
        data = json.loads(RESULTS.read_text())
        changed = 0
        for pair, run in data["runs"].items():
            was, now = run["verdict"], verdict(run["gen"], run["build"], run["route"])
            if was != now:
                run["verdict"] = now
                changed += 1
                print(f"  {was:<11} -> {now:<11} {pair}")
        RESULTS.write_text(json.dumps(data, indent=1))
        print(f"{changed} of {len(data['runs'])} verdicts changed")
        return

    PROJECTS.mkdir(parents=True, exist_ok=True)
    ensure_shared_lake()
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results.setdefault("meta", {})
    results.setdefault("rows", {})
    results.setdefault("runs", {})
    results["meta"]["label"] = args.label or RESULTS.stem
    results["meta"]["machine"] = machine()
    results["meta"]["model"] = MODEL or "claude-sonnet-4-6"
    results["meta"]["gen_timeout"] = GEN_TIMEOUT
    results["meta"]["build_timeout"] = BUILD_TIMEOUT
    results["meta"]["proveit_dir"] = PROVEIT_DIR
    results["meta"]["ic3ia"] = IC3IA

    rows = all_rows(args.suites)
    if args.only:
        rows = [r for r in rows if any(o.lower() in r.bench.lower() for o in args.only)]
    rts = routes(proveit_dir=PROVEIT_DIR, ic3ia=IC3IA)
    if args.routes:
        rts = [r for r in rts if r.name in args.routes]
    todo = pairs(rows, rts)

    for row in rows:
        results["rows"][row.key] = dict(
            suite=row.suite, bench=row.bench, kind=row.kind, prop=row.prop,
            prop_label=row.prop_label, module=rel(row.module), note=row.note,
            env=row.env, shared=list(row.shared))

    pending = [(r, rt) for r, rt in todo
               if args.redo or f"{r.key}::{rt.name}" not in results["runs"]]
    print(f"{len(todo)} pairs, {len(pending)} to run "
          f"({len(todo) - len(pending)} already recorded)\n")

    t_start = time.time()
    for n, (row, route) in enumerate(pending, 1):
        pair = f"{row.key}::{route.name}"
        out = PROJECTS / re.sub(r"[^A-Za-z0-9_.-]", "_", pair)
        label = f"{row.suite}/{row.bench}"
        print(f"[{n}/{len(pending)}] {label[:44]:<44} {row.prop_label[:18]:<18} "
              f"{route.name:<11} ", end="", flush=True)
        gen = generate(row, route, out)
        if not gen["ok"] or args.no_build:
            bld = dict(ok=False, raw_ok=False, secs=0.0, targets=[], sorries=[],
                       errors=["(not built: generation failed)"] if not gen["ok"]
                       else ["(not built: --no-build)"])
        else:
            bld = build(row, route, out)
        v = verdict(gen, bld, route.name) if not args.no_build or not gen["ok"] else "GEN-OK"
        # When, not just how long: a timing is only as good as what else
        # the machine was doing, and without this the only way to place a
        # cell in time is to count lines in the log.
        results["runs"][pair] = dict(route=route.name, verdict=v, gen=gen, build=bld,
                                     when=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                     command=shown(row, route,
                                                   Path("/tmp/verith-out.noindex") / out.name))
        RESULTS.write_text(json.dumps(results, indent=1))
        print(f"{v:<11} gen {gen['secs']:6.1f}s  build {bld['secs']:6.1f}s")
        if v in ("GEN-FAIL", "NO-CERT"):
            print(f"      {gen['err'][:160]}")
        for e in (bld["errors"] or [])[:1]:
            if not e.startswith("(not built"):
                print(f"      {e[:160]}")
        shutil.rmtree(out, ignore_errors=True)     # the .olean cache is shared

    print(f"\nwrote {RESULTS}  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
