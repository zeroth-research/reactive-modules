"""
Lean compilation tests for the tests/lean/ lake project.

Fast tests (default):
    just pytest tests/lean/            # checks generated files exist

Slow tests (require lake + Mathlib cache):
    just pytest tests/lean/ -m slow    # runs lake build

Targets verified by the slow tests:
  ZerothHammerTests  — individual tactic examples (manually written)
  ZerothHammer       — standalone zeroth_hammer tactic (generated)
  Certs              — self-contained module certificates (generated)
  ManualTests        — zeroth_hammer on standalone goal shapes (manually written)
"""
import subprocess
from pathlib import Path

import pytest

_LEAN_DIR = Path(__file__).parent


# ──────────────────────────────────────────────────────────────
# Fast tests — just verify file presence after fixture generation
# ──────────────────────────────────────────────────────────────


def test_core_files_present(sync_core_templates):
    """Core/ template files were copied successfully."""
    from zrth.lean.project import CORE_FILES

    for name in CORE_FILES:
        assert (_LEAN_DIR / "Core" / name).exists(), f"Core/{name} missing after sync"


def test_generated_files_present(generate_lean_files):
    """ZerothHammer.lean and Certs/*.lean exist after generation fixture runs."""
    assert (_LEAN_DIR / "ZerothHammer.lean").exists(), "ZerothHammer.lean not generated"
    for name in ("Countdown", "CountdownSafe", "TwoVars", "Collatz",
                 "ArgmaxScalar", "BridgeCountdown", "BridgeTwoVars"):
        path = _LEAN_DIR / "Certs" / f"{name}.lean"
        assert path.exists(), f"Certs/{name}.lean not generated"


def test_the_learned_certificate_is_generated(generate_lean_files):
    """`--infer learn` writes one too, so the `Certs` build elaborates a
    ranking function of the shape the learner produces. It needs cvc5 to
    render its SMT-LIB as Lean, so its absence is a skip, not a failure."""
    pytest.importorskip("cvc5")
    path = _LEAN_DIR / "Certs" / "LearnedCountdown.lean"
    assert path.exists(), "Certs/LearnedCountdown.lean not generated"
    src = path.read_text()
    assert "sorry" not in src, "the learned certificate left an obligation open"
    assert "if " in src, "the ranking function should carry the net's ReLU units"


def test_generated_argmax_scalar_mirrors_matrix_form(generate_lean_files):
    """The emitted scalar Argmax must seed from s0 and compare strictly.

    `argmax1d_scalar_n_eq` proves the unrolled form equal to
    `Core.Mat.argmax_1d` by unfolding both, so a neutral seed or a
    non-strict comparison here would leave that proof passing against a
    definition that disagrees with `torch.argmax`.
    """
    src = (_LEAN_DIR / "Certs" / "ArgmaxScalar.lean").read_text()
    assert "let b0 : Nat × Int := (0, s0)" in src, "scalar Argmax is not seeded with element 0"
    assert "(0, (0 : Int))" not in src, "scalar Argmax still seeds a neutral value"
    assert ".2 < s1" in src, "scalar Argmax does not use a strict comparison"
    assert ".2 ≤ s" not in src, "scalar Argmax still uses a non-strict comparison"


# ──────────────────────────────────────────────────────────────
# Slow tests — full lake build
# ──────────────────────────────────────────────────────────────


def _lake_build(*targets: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["lake", "build", *targets],
        cwd=_LEAN_DIR,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.slow
def test_zeroth_hammer_proofs(generate_lean_files):
    """ZerothHammerTests.lean compiles — every example in the file type-checks."""
    r = _lake_build("ZerothHammerTests")
    assert r.returncode == 0, (
        f"lake build ZerothHammerTests failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    assert "sorry" not in r.stdout and "warning: declaration uses 'sorry'" not in r.stderr, (
        "Build succeeded but some proof used sorry."
    )


@pytest.mark.slow
def test_manual_tests_build(generate_lean_files):
    """ManualTests/Basic.lean compiles — zeroth_hammer closes each standalone example."""
    r = _lake_build("ManualTests")
    assert r.returncode == 0, (
        f"lake build ManualTests failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "ManualTests" in l]
    assert not sorry_lines, "ManualTests proof used sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_argmax_scalar_equiv_build(generate_lean_files):
    """Certs/ArgmaxScalar.lean compiles — each `argmax1d_scalar_n_eq` proof
    goes through, which is what ties the generated unrolled form to
    `Core.Mat.argmax_1d`."""
    r = _lake_build("Certs.ArgmaxScalar")
    assert r.returncode == 0, (
        f"lake build Certs.ArgmaxScalar failed.\n"
        f"stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-800:]}"
    )
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert not sorry_lines, "ArgmaxScalar proof used sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_the_learned_certificate_closes_in_lean(generate_lean_files):
    """`Certs/LearnedCountdown.lean` compiles.

    The certificate `--infer learn` proved for itself is one `zeroth_hammer`
    closes too -- `hrank` included, over a ranking function that is a sum of
    ReLU units rather than the single `ite` every hand-written certificate in
    this suite uses."""
    pytest.importorskip("cvc5")
    r = _lake_build("Certs.LearnedCountdown")
    assert r.returncode == 0, (
        f"lake build Certs.LearnedCountdown failed.\n"
        f"stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-800:]}"
    )
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert not sorry_lines, "learned certificate used sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_playground_build(generate_lean_files):
    """Playground.lean compiles — proof-pattern experiments and regression tests."""
    r = _lake_build("Playground")
    assert r.returncode == 0, (
        f"lake build Playground failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Playground" in l]
    assert not sorry_lines, "Playground proof used sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_hugecounter_build(generate_lean_files):
    """Certs/HugeCounter.lean: a 32-wide state whose transitions are 32-wide
    contractions. Pre-contraction keeps this tractable where a dense matrix
    literal + symbolic sum-expansion blows the heartbeat budget."""
    r = _lake_build("Certs.HugeCounter")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.HugeCounter failed.\n"
        f"stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-800:]}"
    )
    assert not sorry_lines, "HugeCounter certificate has sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_bigcounter_build(generate_lean_files):
    """Certs/BigCounter.lean: a 6-vector state whose every transition is a 6-wide
    MatMul contraction — exercises Fin.sum_univ_succ expansion at scale."""
    r = _lake_build("Certs.BigCounter")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.BigCounter failed.\n"
        f"stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-800:]}"
    )
    assert not sorry_lines, "BigCounter certificate has sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_counter_build(generate_lean_files):
    """Certs/Counter.lean: a 3×1 vector-state module with LIA.Linear transitions
    and tuple-select (s[i][j]) predicates — zeroth_hammer closes all obligations."""
    r = _lake_build("Certs.Counter")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.Counter failed.\n"
        f"stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-800:]}"
    )
    assert not sorry_lines, "Counter certificate has sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_countdown_build(generate_lean_files):
    """Certs/Countdown.lean: zeroth_hammer closes all proof obligations."""
    r = _lake_build("Certs.Countdown")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.Countdown failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    assert not sorry_lines, "Countdown certificate has sorry:\n" + "\n".join(sorry_lines)


def test_generated_safety_cert_has_no_ranking(generate_lean_files):
    """`Certs/CountdownSafe.lean` is the `--safety` shape: `rule_globally`
    over an invariant that implies `P`, with no ranking function defined or
    named anywhere -- a `def ranking := sorry` would be a `sorry` in a file
    whose point is that it has none."""
    src = (_LEAN_DIR / "Certs" / "CountdownSafe.lean").read_text()
    assert "rule_globally" in src and "theorem inv_imp_P" in src
    assert "rule_buchi" not in src
    assert "ranking" not in src and "sorry" not in src


@pytest.mark.slow
@pytest.mark.parametrize("name", ["Countdown", "TwoVars"])
def test_fbk_bridge_builds(generate_lean_files, name):
    """`Certs/Bridge*.lean`: the proof that the `--fbk-proveit` model is the
    module.

    The certificate that route installs is about the *model*; this is the
    file that carries it back. Two independent translators are on the two
    sides of `bridge_k` -- `smt_encode` + cvc5 against
    `_translate_terms_scalar` -- so nothing here closes by `rfl` alone, and
    a translation defect in either shows up as an unclosed goal.
    """
    r = _lake_build(f"Certs.Bridge{name}")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.Bridge{name} failed.\n"
        f"stdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-800:]}"
    )
    assert not sorry_lines, f"Bridge{name} used sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_countdownsafe_build(generate_lean_files):
    """Certs/CountdownSafe.lean: the safety certificate elaborates, which is
    what says `rule_globally lts P hP` is applied to the right things --
    `hP` transfers the invariant to `P` through `inv_imp_P`."""
    r = _lake_build("Certs.CountdownSafe")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.CountdownSafe failed.\n"
        f"stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-800:]}"
    )
    assert not sorry_lines, "CountdownSafe certificate has sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_twovars_build(generate_lean_files):
    """Certs/TwoVars.lean: zeroth_hammer closes all proof obligations."""
    r = _lake_build("Certs.TwoVars")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.TwoVars failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    assert not sorry_lines, "TwoVars certificate has sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
def test_cert_collatz_build(generate_lean_files):
    """Certs/Collatz.lean: zeroth_hammer closes all proof obligations."""
    r = _lake_build("Certs.Collatz")
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert r.returncode == 0, (
        f"lake build Certs.Collatz failed.\n"
        f"stdout:\n{r.stdout[-1000:]}\nstderr:\n{r.stderr[-1000:]}"
    )
    assert not sorry_lines, "Collatz certificate has sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
@pytest.mark.parametrize(
    "fixture,stdin,expected",
    [
        # LIA: x starts at 0, increments each step, resets at 10
        ("counter", "x\nx\nx\n", ["0", "1", "2"]),
        # BV: two-bit counter with enable=1 counts 00, 10, 01, 11
        ("twobit", "1\n1\n1\n", ["0x0#1", "0x0#1", "0x1#1", "0x0#1", "0x0#1", "0x1#1"]),
    ],
)
def test_generated_executable_builds_and_runs(
    generate_lean_files, fixture, stdin, expected
):
    """`verith -x` end to end: generate a project, build `main`, run it.

    The only coverage the executable path has. Its three generated pieces
    have to agree — `update`'s curried signature, the component order in
    `showCtrl`/`parseExtl`, and each wire's element type — and none of that
    is visible without elaborating the result.

    Reuses this project's already-built `.lake/packages` so the run does not
    refetch Mathlib; skipped if they are not built yet.
    """
    import shutil
    import sys
    import tempfile

    packages = _LEAN_DIR / ".lake" / "packages"
    manifest = _LEAN_DIR / "lake-manifest.json"
    if not packages.is_dir() or not manifest.is_file():
        pytest.skip("tests/lean packages not built; run the other slow tests first")

    import importlib

    sys.path.insert(0, str(_LEAN_DIR.parent / "fixtures"))
    try:
        module_def = importlib.import_module(fixture).module()
    finally:
        sys.path.pop(0)
    from zrth.lean.project import create_project

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        project = create_project(out, module_def, "Rea", executable=True)

        (project / ".lake").mkdir(exist_ok=True)
        (project / ".lake" / "packages").symlink_to(packages)
        shutil.copy2(manifest, project / "lake-manifest.json")

        build = subprocess.run(
            ["lake", "build", "main"],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert build.returncode == 0, (
            "lake build main failed.\n"
            f"stdout:\n{build.stdout[-2000:]}\nstderr:\n{build.stderr[-800:]}"
        )

        run = subprocess.run(
            [str(project / ".lake" / "build" / "bin" / "main")],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert run.returncode == 0, f"executable failed: {run.stderr[-500:]}"
        assert run.stdout.split() == expected, (
            f"unexpected trace: {run.stdout.split()}"
        )


@pytest.mark.slow
@pytest.mark.parametrize("name", ["Scalar", "Vec6", "Counter", "Vec32"])
def test_scalar_encoding_builds(generate_lean_files, name):
    """The scalar encoding elaborates alongside the functional one.

    `_product_type_scalar` flattens a multi-element wire into one component
    per element, and the body and the `_scalar_eq` proofs have to follow. No
    certificate in Certs/ carries a Scalar section, so without this the
    encoding was never compiled at all — it had four type errors per
    matrix-state module.
    """
    r = _lake_build(f"Certs.ScalarEnc{name}")
    assert r.returncode == 0, (
        f"lake build Certs.ScalarEnc{name} failed.\n"
        f"stdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-800:]}"
    )
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert not sorry_lines, f"ScalarEnc{name} used sorry:\n" + "\n".join(sorry_lines)


@pytest.mark.slow
@pytest.mark.parametrize(
    "name", ["Scalar", "Vec6", "Counter", "Vec32", "TwoVars", "Mixed"]
)
def test_relational_encoding_builds(generate_lean_files, name):
    """ScalarRel elaborates on top of the scalar encoding.

    `effect_i` is per ctrl *wire*; the state tuple it is related to is per
    *element*. Projecting component `i` of that tuple therefore put a
    `Mat Int 6 1` against an `Int x ... x Int` for any multi-element wire, and
    `Mixed` is the case where the offsets differ too. The chain that has teeth
    here is `TransRel_scalar_eq` into `TransRel_func_eq`: it ties the sliced
    relation back to the functional `update` through `pack`/`unpack`, so a
    slice at the wrong offset does not typecheck rather than proving a wrong
    equation. No certificate in Certs/ carries a ScalarRel section, so
    without this it was not compiled at all.

    The encodings are separate modules (see `conftest.py`), as they are in a
    generated project. Concatenated into one file they elaborate more than
    they do apart -- a `match` gets one auxiliary matcher per module -- and
    that difference alone was enough to hide a broken `effect_i_eq` from
    this test.
    """
    r = _lake_build(f"Certs.RelEnc{name}")
    assert r.returncode == 0, (
        f"lake build Certs.RelEnc{name} failed.\n"
        f"stdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-800:]}"
    )
    sorry_lines = [l for l in r.stdout.splitlines() if "sorry" in l and "Certs/" in l]
    assert not sorry_lines, f"RelEnc{name} used sorry:\n" + "\n".join(sorry_lines)
