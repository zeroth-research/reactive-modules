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
            Path(tmpdir) / "CounterBasic" / "CounterBasic" / "CounterBasicData.lean"
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
