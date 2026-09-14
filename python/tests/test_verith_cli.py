"""Tests for the verith CLI (zrth.lean.main)."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
COUNTER_MODULE = FIXTURE_DIR / "counter.py"
# Root of the python package (where pyproject.toml lives)
PKG_ROOT = Path(__file__).parent.parent


def _verith(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["uv", "run", "verith", *args],
        capture_output=True,
        text=True,
        cwd=PKG_ROOT,
    )


def _ollama_available() -> bool:
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

    try:
        import openai
    except ImportError:
        return False


# ── Basic invocation ────────────────────────────────────────────────────────


def test_verith_no_property():
    """Without a property the certificate uses sorry placeholders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        r = _verith(str(COUNTER_MODULE), "-o", tmpdir, "-p", "CounterBasic")
        assert r.returncode == 0, r.stderr
        assert (Path(tmpdir) / "CounterBasic").exists()
        data = (
            Path(tmpdir) / "CounterBasic" / "System" / "Data.lean"
        ).read_text()
        assert "sorry" in data


def test_a_predicate_of_the_wrong_sort_is_refused():
    """The fields are declared: a property, an invariant and a precondition
    become the body of a `Prop`, a ranking function the body of a `Nat`. An
    Int property printed as a `Prop` and a Bool ranking printed as
    `((… = 0) : Int).toNat` are both only Lean's problem otherwise, and
    `--fbk-proveit` has always checked its own."""
    with tempfile.TemporaryDirectory() as tmpdir:
        refused = [
            ("--safety", "s0"),                              # Int as a Prop
            ("--pre", "(+ 1 1)"),                            # Int as a Prop
            ("--buchi", "(= s0 0)", "--ranking", "(= s0 0)"),  # Bool as a Nat
        ]
        for args in refused:
            r = _verith(str(COUNTER_MODULE), *args, "-o", tmpdir, "-p", "Sorts")
            assert r.returncode != 0, f"{args} was accepted"
            assert "must have sort" in r.stderr, r.stderr

        # and the shapes that are right still pass
        r = _verith(
            str(COUNTER_MODULE), "--buchi", "(= s0 0)", "--ranking", "s0",
            "-o", tmpdir, "-p", "Sorts",
        )
        assert r.returncode == 0, r.stderr


def test_a_predicate_cvc5_cannot_read_is_refused_not_pasted():
    """The fields become the bodies of `def P`, `def inv`, `def ranking`.
    A source that does not parse used to be interpolated into them verbatim
    -- `def P : … → Prop := (= s0` -- and the run still said `Project ready`,
    leaving a Lean parse error in generated code as the first sign."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cases = [
            (("--safety", "(= s0"), "--safety"),          # unbalanced
            (("--safety", "(= s9 0)"), "--safety"),       # no such state var
            (("--buchi", "(and s0"), "--buchi"),
            (("--safety", "true", "--invariant", "(bogus s0)"), "--invariant"),
        ]
        for args, flag in cases:
            r = _verith(str(COUNTER_MODULE), *args, "-o", tmpdir, "-p", "Unparsed")
            assert r.returncode != 0, f"{args} was accepted"
            assert "Traceback" not in r.stderr, r.stderr
            assert flag in r.stderr, f"{args} did not name the flag:\n{r.stderr}"
            data = Path(tmpdir) / "Unparsed" / "System" / "Data.lean"
            if data.exists():
                assert args[1] not in data.read_text(), "the SMT source was pasted"


def test_a_refusal_is_an_error_line_not_a_traceback():
    """Every one of these is a decision the generator makes about its input,
    and each already carries a message saying why. A traceback buries that
    message under frames the user cannot act on."""
    real_module = FIXTURE_DIR / "simple_env.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        cases = [
            # `Real` is noncomputable in Lean, so `-x` cannot print it
            (str(real_module), "-x"),
            # a path that is not there
            ("/nonexistent/module.py",),
            # a `-d` the file does not define
            (str(COUNTER_MODULE), "-d", "no_such_function"),
            # a state variable in a precondition, which reads inputs
            (str(COUNTER_MODULE), "--pre", "(= s0 0)"),
        ]
        for case in cases:
            r = _verith(*case, "-o", tmpdir, "-p", "Refused")
            assert r.returncode != 0, f"{case} was accepted"
            assert "Traceback" not in r.stderr, f"{case} crashed:\n{r.stderr}"
            assert "error:" in r.stderr, f"{case} said nothing:\n{r.stderr}"


def test_a_stale_bridge_is_not_left_for_lake_to_build():
    """`Certificate/` is globbed by the lakefile, so every file in it is
    built. Only `--fbk-proveit` writes `Equivalence.lean`, and it is about
    the module of the run that wrote it -- a copy left in an `-o` reused by
    another module would be built against the wrong `System/`."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cert_dir = Path(tmpdir) / "Stale" / "Certificate"
        cert_dir.mkdir(parents=True)
        (cert_dir / "Equivalence.lean").write_text("import NoSuchModule\n")

        r = _verith(str(COUNTER_MODULE), "-o", tmpdir, "-p", "Stale")
        assert r.returncode == 0, r.stderr
        assert not (cert_dir / "Equivalence.lean").exists()
        assert (cert_dir / "Certificate.lean").exists()


def test_verith_with_property():
    """A property without --infer writes prp as sorry (string not compiled to Terms)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        r = _verith(
            str(COUNTER_MODULE), "--buchi", "s0 == 0", "-o", tmpdir, "-p", "CounterProp"
        )
        assert r.returncode == 0, r.stderr
        assert (Path(tmpdir) / "CounterProp").exists()


def test_verith_infer_requires_property():
    """--infer without --safety or --buchi exits with a non-zero status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        r = _verith(str(COUNTER_MODULE), "--infer", "-o", tmpdir)
        assert r.returncode != 0


def test_verith_infer_rejects_cert_file():
    """The standalone certificate is written before inference could reach it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        r = _verith(
            str(COUNTER_MODULE),
            "--buchi",
            "(= s0 0)",
            "--infer",
            "--cert-file",
            str(Path(tmpdir) / "Cert.lean"),
            "-o",
            tmpdir,
        )
        assert r.returncode != 0
        assert "--cert-file" in r.stderr


# ── the pre-check and inference, in that order ──────────────────────────────


# `--infer ai-cegar` with both predicates fixed makes no LLM call: the CEGAR
# loop just verifies them once. That is the whole inference path, offline.
CEGAR_FIXED = (
    "--buchi", "(= s0 0)",
    "--invariant", "(and (>= s0 0) (<= s0 9))",
    "--ranking", "(ite (= s0 0) 0 (- 10 s0))",
    "--infer", "ai-cegar",
)


def test_pre_check_checks_the_inferred_certificate():
    """`--pre-check` has to wait for `--infer`, or it checks nothing."""
    pytest.importorskip("anthropic")  # the CEGAR route builds a client either way
    with tempfile.TemporaryDirectory() as tmpdir:
        r = _verith(
            str(COUNTER_MODULE), *CEGAR_FIXED, "--pre-check", "cvc5",
            "-o", tmpdir, "-p", "CounterPreCheck",
        )
        assert r.returncode == 0, r.stderr
        out = r.stdout
        assert "init_inv" in out and "step_inv" in out and "hrank" in out
        assert "nothing to check" not in out
        # After inference, not before: the certificate it reports on is the
        # one the project was written with.
        assert out.index("TA2Magic") < out.index("SMT pre-check")


# ── safety and Buchi are different certificates ─────────────────────────────


def _cert(tmpdir, name) -> tuple[str, str]:
    """(Data.lean, Certificate.lean) of a generated project."""
    root = Path(tmpdir) / name
    return (
        (root / "System" / "Data.lean").read_text(),
        (root / "Certificate" / "Certificate.lean").read_text(),
    )


def test_buchi_generates_a_ranking_certificate():
    with tempfile.TemporaryDirectory() as tmpdir:
        r = _verith(
            str(COUNTER_MODULE), "--buchi", "(= s0 0)",
            "--invariant", "(and (>= s0 0) (<= s0 9))",
            "--ranking", "(ite (= s0 0) 0 (- 10 s0))",
            "-o", tmpdir, "-p", "CounterBuchi",
        )
        assert r.returncode == 0, r.stderr
        data, cert = _cert(tmpdir, "CounterBuchi")
        assert "def ranking" in data
        assert "rule_buchi" in cert and "theorem hrank" in cert


def test_safety_generates_a_rule_globally_certificate():
    """`G P` is proved by an invariant that implies `P`. No ranking function
    is defined, stated or mentioned -- `rule_globally` has nowhere to put
    one, and a `def ranking := sorry` would be a `sorry` in a file whose
    point is that it has none."""
    with tempfile.TemporaryDirectory() as tmpdir:
        r = _verith(
            str(COUNTER_MODULE), "--safety", "(<= s0 9)",
            "--invariant", "(and (>= s0 0) (<= s0 9))",
            "-o", tmpdir, "-p", "CounterSafety",
        )
        assert r.returncode == 0, r.stderr
        data, cert = _cert(tmpdir, "CounterSafety")
        assert "ranking" not in data and "ranking" not in cert
        assert "rule_globally" in cert and "def safety" in cert
        assert "theorem inv_imp_P" in cert
        assert "rule_buchi" not in cert


def test_safety_and_buchi_are_mutually_exclusive(tmp_path):
    r = _verith(
        str(COUNTER_MODULE), "--safety", "(<= s0 9)", "--buchi", "(= s0 0)",
        "-o", str(tmp_path), "-p", "P",
    )
    assert r.returncode != 0
    assert "mutually exclusive" in r.stderr


def test_safety_rejects_a_ranking(tmp_path):
    """Taking a ranking function and dropping it would look like it was used."""
    r = _verith(
        str(COUNTER_MODULE), "--safety", "(<= s0 9)", "--ranking", "s0",
        "-o", str(tmp_path), "-p", "P",
    )
    assert r.returncode != 0
    assert "--ranking is meaningless with --safety" in r.stderr


def test_safety_rejects_the_unchecked_inference_route(tmp_path):
    """`--infer ai` infers a ranking function and checks nothing with cvc5."""
    r = _verith(
        str(COUNTER_MODULE), "--safety", "(<= s0 9)", "--infer", "ai",
        "-o", str(tmp_path), "-p", "P",
    )
    assert r.returncode != 0
    assert "--safety needs --infer ai-cegar" in r.stderr


def test_cegar_infers_a_safety_certificate():
    """The CEGAR loop with the invariant fixed makes no LLM call: it states
    the safety obligations (init, inductive, `inv -> P`) and verifies them,
    which is the whole route minus the prompting."""
    pytest.importorskip("anthropic")
    with tempfile.TemporaryDirectory() as tmpdir:
        r = _verith(
            str(COUNTER_MODULE), "--safety", "(<= s0 9)",
            "--invariant", "(and (>= s0 0) (<= s0 9))",
            "--infer", "ai-cegar", "--pre-check", "cvc5",
            "-o", tmpdir, "-p", "CounterSafetyInfer",
        )
        assert r.returncode == 0, r.stderr
        assert "all obligations UNSAT" in r.stdout
        # The pre-check states the safety obligation, not `hrank`.
        assert "inv_imp_P" in r.stdout and "hrank" not in r.stdout
        data, cert = _cert(tmpdir, "CounterSafetyInfer")
        assert "ranking" not in data and "rule_globally" in cert


# ── AI inference ────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_verith_infer_claude():
    """--infer with Claude API produces a certificate with inv, P, and ranking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        r = _verith(
            str(COUNTER_MODULE),
            "--buchi",
            "s0 == 0",
            "--infer",
            "-o",
            tmpdir,
            "-p",
            "CounterInferClaude",
        )
        assert r.returncode == 0, r.stderr
        cert = (
            Path(tmpdir) / "CounterInferClaude" / "Certificate" / "Certificate.lean"
        ).read_text()
        assert "def inv" in cert
        assert "def P" in cert
        assert "def ranking" in cert
        assert "sorry" not in cert.split("hrank")[1]  # hrank proof may still have sorry


@pytest.mark.skipif(
    not _ollama_available(), reason="Ollama or openai package not available"
)
def test_verith_infer_ollama():
    """--infer with local Ollama LLM produces a certificate with inv, P, and ranking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        r = _verith(
            str(COUNTER_MODULE),
            "--buchi",
            "s0 == 0",
            "--infer",
            "--model",
            "qwen3-coder",
            "--base-url",
            "http://localhost:11434/v1",
            "-o",
            tmpdir,
            "-p",
            "CounterInferOllama",
        )
        assert r.returncode == 0, r.stderr
        cert = (
            Path(tmpdir) / "CounterInferOllama" / "Certificate" / "Certificate.lean"
        ).read_text()
        assert "def inv" in cert
        assert "def P" in cert
        assert "def ranking" in cert


# ── --build-cert ────────────────────────────────────────────────────────────


def test_build_cert_needs_a_certificate_to_build(tmp_path):
    """A bare project's obligations are all `sorry`. Lake compiles that and
    proves nothing, so asking for the build is a mistake, not a no-op."""
    r = _verith(
        str(COUNTER_MODULE), "--buchi", "(= s0 0)",
        "-o", str(tmp_path), "-p", "P", "--build-cert",
    )
    assert r.returncode != 0
    assert "--build-cert needs a certificate to build" in r.stderr
    # An invariant without a ranking is still a `sorry`, so it is not enough.
    r = _verith(
        str(COUNTER_MODULE), "--buchi", "(= s0 0)", "--invariant", "(<= s0 100)",
        "-o", str(tmp_path), "-p", "P", "--build-cert",
    )
    assert r.returncode != 0
    assert "--build-cert needs a certificate to build" in r.stderr


def _fake_ltl_checkout(root: Path) -> Path:
    """The four files `resolve_project` looks for, and nothing else."""
    (root / "LTLCertifying").mkdir(parents=True)
    (root / "LTLCertifying" / "lean2vmt.lean").write_text("")
    for name in ("proveit.py", "vmt2lean.py", "lakefile.toml"):
        (root / name).write_text("")
    return root


@pytest.mark.parametrize(
    "route",
    [
        ["--buchi", "(= s0 0)", "--invariant", "(<= s0 100)", "--ranking", "s0"],
        # A safety certificate is complete without a ranking function.
        ["--safety", "(<= s0 100)", "--invariant", "(<= s0 100)"],
        ["--buchi", "(= s0 0)", "--infer"],
        ["--safety", "(= s0 0)", "--fbk-proveit", "<checkout>"],
    ],
)
def test_build_cert_is_accepted_by_every_route_that_fills_the_certificate(
    route, tmp_path
):
    """Every way a certificate acquires predicates passes the gate.

    The module file does not exist, so each run stops at loading it -- after
    the gate and before any project, LLM or lake. What this pins is that the
    failure is never the gate.
    """
    route = [
        str(_fake_ltl_checkout(tmp_path / "ltl")) if a == "<checkout>" else a
        for a in route
    ]
    r = _verith(
        str(tmp_path / "no-such-module.py"),
        "-o", str(tmp_path), "-p", "P", "--build-cert", *route,
    )
    assert r.returncode != 0
    assert "--build-cert" not in r.stderr


def test_build_cert_rejects_the_routes_that_generate_no_project(tmp_path):
    for flag, value in (
        ("--cert-file", str(tmp_path / "C.lean")),
        ("--hammer-file", str(tmp_path / "H.lean")),
    ):
        r = _verith(
            str(COUNTER_MODULE), "--buchi", "(= s0 0)", "--invariant", "(<= s0 100)",
            "--ranking", "s0", "-o", str(tmp_path), "-p", "P",
            "--build-cert", flag, value,
        )
        assert r.returncode != 0
        assert f"--build-cert is incompatible with {flag}" in r.stderr


def test_build_cert_checks_the_predicates_it_was_promised(monkeypatch):
    """The parse-time gate cannot see what `--infer` came back with: an LLM
    that answers nothing usable leaves `inv` or `ranking` unset, and building
    that would report a proof of `sorry`."""
    from types import SimpleNamespace

    from zrth.lean import main as m
    from zrth.lean.cert import CertificateData

    built: list[Path] = []
    monkeypatch.setattr(m, "build_certificate", built.append)
    on = SimpleNamespace(build_cert=True)

    m._build_cert(on, Path("/complete"), CertificateData(
        prp="p", inv="i", ranking="r"))
    assert built == [Path("/complete")]

    with pytest.raises(SystemExit, match="no ranking"):
        m._build_cert(on, Path("/p"), CertificateData(prp="p", inv="i"))
    with pytest.raises(SystemExit, match="no invariant, ranking"):
        m._build_cert(on, Path("/p"), CertificateData(prp="p"))

    # --fbk-proveit installs ic3ia's certificate, which `cert_data` does not
    # describe, so there is nothing to be complete about.
    m._build_cert(on, Path("/proveit"), None)
    assert built == [Path("/complete"), Path("/proveit")]

    m._build_cert(SimpleNamespace(build_cert=False), Path("/off"), None)
    assert len(built) == 2


def test_a_certificate_that_compiles_with_sorry_is_not_a_proof(
    monkeypatch, tmp_path
):
    """`zeroth_hammer` leaves a `sorry` where it cannot close an obligation,
    and lake reports that as a warning and exits 0 — so a plain exit code
    would make "built" mean nothing."""
    from zrth.lean import project

    logs: dict[str, tuple[int, str]] = {}
    monkeypatch.setattr(
        project, "stream", lambda cmd, *, cwd: logs[cmd[1]]
    )

    logs["update"] = (0, "")
    logs["build"] = (
        0,
        "warning: Certificate/Certificate.lean:42:0: declaration uses 'sorry'\n"
        "Build completed successfully (3 jobs).\n",
    )
    with pytest.raises(project.LakeBuildError, match="not proved"):
        project.build_certificate(tmp_path)

    # A `sorry` in a dependency is not ours: lean-smt ships one.
    logs["build"] = (
        0,
        "warning: Smt/Reconstruct/BitVec/Bitblast.lean:36:4: declaration uses 'sorry'\n"
        "Build completed successfully (3 jobs).\n",
    )
    project.build_certificate(tmp_path)

    logs["build"] = (1, "error: Certificate/Certificate.lean:9:2: omega failed\n")
    with pytest.raises(project.LakeBuildError, match="omega failed"):
        project.build_certificate(tmp_path)
