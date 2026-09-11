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
    """Without --property the certificate uses sorry placeholders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        r = _verith(str(COUNTER_MODULE), "-o", tmpdir, "-p", "CounterBasic")
        assert r.returncode == 0, r.stderr
        assert (Path(tmpdir) / "CounterBasic").exists()
        data = (
            Path(tmpdir) / "CounterBasic" / "System" / "Data.lean"
        ).read_text()
        assert "sorry" in data


def test_verith_with_property():
    """--property without --infer writes prp as sorry (string not compiled to Terms)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        r = _verith(
            str(COUNTER_MODULE), "-P", "s0 == 0", "-o", tmpdir, "-p", "CounterProp"
        )
        assert r.returncode == 0, r.stderr
        assert (Path(tmpdir) / "CounterProp").exists()


def test_verith_infer_requires_property():
    """--infer without --property exits with a non-zero status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        r = _verith(str(COUNTER_MODULE), "--infer", "-o", tmpdir)
        assert r.returncode != 0


def test_verith_infer_rejects_cert_file():
    """The standalone certificate is written before inference could reach it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        r = _verith(
            str(COUNTER_MODULE),
            "-P",
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
    "-P", "(= s0 0)",
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
            "-P",
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
            "-P",
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
        str(COUNTER_MODULE), "-P", "(= s0 0)",
        "-o", str(tmp_path), "-p", "P", "--build-cert",
    )
    assert r.returncode != 0
    assert "--build-cert needs a certificate to build" in r.stderr
    # An invariant without a ranking is still a `sorry`, so it is not enough.
    r = _verith(
        str(COUNTER_MODULE), "-P", "(= s0 0)", "--invariant", "(<= s0 100)",
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
        ["--invariant", "(<= s0 100)", "--ranking", "s0"],
        ["--infer"],
        ["--fbk-proveit", "<checkout>"],
    ],
)
def test_build_cert_is_accepted_by_every_route_that_fills_the_certificate(
    route, tmp_path
):
    """The three ways a certificate acquires predicates all pass the gate.

    The module file does not exist, so each run stops at loading it -- after
    the gate and before any project, LLM or lake. What this pins is that the
    failure is never the gate.
    """
    route = [
        str(_fake_ltl_checkout(tmp_path / "ltl")) if a == "<checkout>" else a
        for a in route
    ]
    r = _verith(
        str(tmp_path / "no-such-module.py"), "-P", "(= s0 0)",
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
            str(COUNTER_MODULE), "-P", "(= s0 0)", "--invariant", "(<= s0 100)",
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
