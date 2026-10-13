"""Prove termination of reactive modules by a learned ranking function.

    python -m nuterm FILE.py                 # train and verify
    python -m nuterm FILE.py --lean          # also emit and kernel-check the proof
    python -m nuterm FILE.py --export out/   # write the proof as its own Lean project

``FILE.py`` defines ``module()``, returning a ``zrth.Module``, or ``BENCH``, a
:class:`nuterm._bench.Bench`. The exit code is 0 when every file is verified
(and kernel-checked, with ``--lean``).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

from ._bench import Bench, bench_of_module
from ._train import learn_ranking


def _import(path: Path):
    """Import ``path``. A file inside a package is imported under its dotted name,
    so its relative imports work; any other file is loaded on its own."""
    path = path.resolve()
    root, parts = path.parent, [path.stem]
    while (root / "__init__.py").exists():
        parts.insert(0, root.name)
        root = root.parent
    sys.path.insert(0, str(root))
    try:
        if len(parts) > 1:
            return importlib.import_module(".".join(parts))
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


def load(path: Path) -> Bench:
    """The :class:`Bench` that the Python file ``path`` defines."""
    mod = _import(path)
    bench = getattr(mod, "BENCH", None)
    # duck-typed: the file may have imported `Bench` through another package path
    if bench is not None and all(hasattr(bench, a) for a in ("name", "state", "inputs", "build")):
        return bench
    if callable(getattr(mod, "module", None)):
        return bench_of_module(mod.module(), path.stem, path.name)
    raise SystemExit(f"{path}: defines neither `BENCH` nor `module()`")


def run_plain(benches) -> int:
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
        r = learn_ranking(bench)
        t_train = time.perf_counter() - t0
        res = (lc.certify(bench.name, r.system, r.proof,
                          label="terminates via a ranking function")
               if r.verified
               else lc.CheckResult(bench.name, "UNVERIFIED", detail=r.reason or ""))
    except Exception as e:
        t_train = time.perf_counter() - t0
        res = lc.CheckResult(bench.name, "ERROR", detail=f"{type(e).__name__}: {e}")
    res.train_s = t_train
    return res


def run_export(benches, out_dir: Path) -> int:
    """Train and verify each benchmark, then write its proof as a standalone
    project under ``out_dir``. Nothing is compiled here."""
    from . import _lean_check as lc

    written = 0
    for bench in benches:
        r = learn_ranking(bench)
        if not r.verified:
            print(f"unverified {bench.name}: {r.reason or ''}", flush=True)
            continue
        path = lc.export_project(bench.name, r.system, r.proof, out_dir,
                                 label="terminates via a ranking function")
        written += 1
        print(f"exported   {bench.name} -> {path}", flush=True)
    print(f"\n{written}/{len(benches)} exported")
    return written


def run_lean(benches) -> int:
    """Kernel-check each benchmark's proof; returns how many were checked."""
    from . import _lean_check as lc

    if not lc.toolchain_available():
        print("no `lake` on PATH: install the Lean toolchain to use --lean")
        return -1
    lc.build_substrate()
    checked = 0
    for b in benches:
        res = certify_one(b)
        checked += res.outcome == "CHECKED"
        print(f"{res.outcome:10s} {res.name} {res.detail}".rstrip(), flush=True)
    print(f"\n{checked}/{len(benches)} kernel-checked")
    return checked


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nuterm", description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", type=Path, help="Python files defining `module()` or `BENCH`")
    ap.add_argument("--lean", action="store_true", help="emit and kernel-check the proof")
    ap.add_argument("--export", type=Path, metavar="DIR", help="write each proof as a Lean project")
    args = ap.parse_args(argv)

    benches = [load(f) for f in args.files]
    t0 = time.perf_counter()
    if args.export is not None:
        n = run_export(benches, args.export)
    elif args.lean:
        n = run_lean(benches)
    else:
        n = run_plain(benches)
    if n < 0:
        return 1
    print(f"({time.perf_counter() - t0:.1f}s)")
    return 0 if n == len(benches) else 1


if __name__ == "__main__":
    raise SystemExit(main())
