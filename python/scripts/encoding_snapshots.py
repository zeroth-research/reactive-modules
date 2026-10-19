"""Every Lean text verith generates for a module, over every fixture module.

The route transcripts (`route_transcripts.py`) and the goldens
(`tests/test_pipeline_golden.py`) cover the certificate files, not the
`System/*.lean` encodings, the `--cert-file` certificate of a bare module, or
the fbk-proveit NA model and bridge. This writes those, so a codegen refactor
is checked the same way: record at the commit before and after, then
`diff -r`. Run from `python/`:

    uv run python scripts/encoding_snapshots.py OUTDIR

OUTDIR/<fixture>/ gets one file per `project.ENCODINGS` row (`System.lean`,
`SystemCirc.lean`, ...; printed by one `ModuleToLean4`, in table order, as
`create_project` does), each row again from a fresh `ModuleToLean4`
(`alone/`, which is what shows state carried between producers), `Cert.lean`
(the standalone certificate with no claim), and `NA.lean` / `Bridge.lean`
where `check_na_supported` takes the module (`property_lean` "true"). A stage
that refuses writes `<stage>.error` with the refusal instead. Every file
name ends in `.txt`, which the repository ignores.
"""

import sys
import traceback
from pathlib import Path

PY = Path(__file__).resolve().parents[1]
FIXTURES = [
    *sorted((PY / "tests" / "fixtures").glob("*.py")),
    *sorted((PY / "tests" / "limits" / "mods").glob("m_*.py")),
    *sorted((PY / "tests" / "smv_fixtures").glob("*.smv")),
]


def _write(out: Path, name: str, produce) -> None:
    # `.txt` on everything: the repository ignores it, so a snapshot kept
    # beside the checkout stays out of `git status`.
    try:
        text = produce()
    except Exception as e:                  # a refusal is part of the record
        (out / f"{name}.error.txt").write_text(
            f"{type(e).__name__}: {e}\n" if not isinstance(e, AssertionError)
            else traceback.format_exc())
        return
    (out / f"{name}.txt").write_text(text)


def snapshot(path: Path, out: Path) -> None:
    from zrth.lean import inputs
    from zrth.lean.context import LeanContext
    from zrth.lean.native import lean_gaps
    from zrth.lean.project import ENCODINGS, generate_standalone_cert_lean
    from zrth.lean.translate import ModuleToLean4
    from zrth.lean.translate.fbk import atom_to_lean_na, check_na_supported
    from zrth.lean.translate.fbk_bridge import atom_to_lean_fbk_bridge

    out.mkdir(parents=True, exist_ok=True)
    try:
        module = inputs.load(path).module
        ctx = LeanContext(module)
        gaps = lean_gaps(ctx)
    except Exception as e:
        (out / "load.error.txt").write_text(f"{type(e).__name__}: {e}\n")
        return
    if gaps:
        (out / "load.error.txt").write_text("no Lean form: " + "; ".join(gaps) + "\n")
        return

    shared = ModuleToLean4(ctx)
    (out / "alone").mkdir(exist_ok=True)
    for enc in ENCODINGS:
        name = f"System{enc.suffix}.lean"
        _write(out, name, getattr(shared, enc.producer))
        _write(out / "alone", name,
               lambda enc=enc: getattr(ModuleToLean4(LeanContext(module)),
                                       enc.producer)())
    _write(out, "Cert.lean", lambda: generate_standalone_cert_lean(module))

    try:
        bodies = check_na_supported(ctx)
    except Exception as e:
        (out / "NA.error.txt").write_text(f"{type(e).__name__}: {e}\n")
        return
    _write(out, "NA.lean", lambda: atom_to_lean_na(
        ctx, "true", module_name="NA", bodies=bodies))
    _write(out, "Bridge.lean", lambda: atom_to_lean_fbk_bridge(
        ctx, na_module="NA", bodies=bodies))


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    for path in FIXTURES:
        snapshot(path, root / f"{path.parent.name}.{path.stem}")
        print(path.name, file=sys.stderr)


if __name__ == "__main__":
    main()
