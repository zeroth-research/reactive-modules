"""Tests for AI-based invariant/ranking inference."""

import os
import subprocess
import pytest

from zrth.lean.cert import CertificateData
from zrth.lean.magic_ai import TA2MagicAI

COUNTER_SOURCE = """\
def init():
    return 0

def update(old_x):
    x = old_x + 1
    if x == 10:
        return 0
    return x
"""


def _assert_result(result: CertificateData):
    assert result.inv is not None
    assert result.ranking is not None
    print(f"invariant: {result.inv}")
    print(f"ranking:   {result.ranking}")


# --- parsing the reply (no model) ---

def test_an_answer_in_backticks_is_the_answer():
    """A model quotes Lean the way the prompt does; Lean would read the quote."""
    inv, ranking = TA2MagicAI._parse_generate_response(
        "Reasoning first.\n\n"
        "INVARIANT: `fun s => 0 ≤ s.1 0 0`\n"
        "RANKING: `fun s => (s.1 0 0).toNat`\n"
    )
    assert inv == "fun s => 0 ≤ s.1 0 0"
    assert ranking == "fun s => (s.1 0 0).toNat"


def test_an_unquoted_answer_is_left_alone():
    inv, ranking = TA2MagicAI._parse_generate_response(
        "INVARIANT: fun s => 0 ≤ s.1 0 0\nRANKING: fun s => (s.1 0 0).toNat"
    )
    assert inv == "fun s => 0 ≤ s.1 0 0"
    assert ranking == "fun s => (s.1 0 0).toNat"


def test_a_backtick_inside_the_expression_is_not_a_quote():
    inv, _ = TA2MagicAI._parse_generate_response(
        "INVARIANT: fun s => s.1 0 0 = `x.name\nRANKING: fun s => 0"
    )
    assert inv == "fun s => s.1 0 0 = `x.name"

# --- Claude API tests ---

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_counter_claude():
    from zrth.lean.magic_ai import TA2MagicAI

    magic = TA2MagicAI(COUNTER_SOURCE)
    _assert_result(magic.infer(CertificateData(prp="x == 0")))


# --- Local LLM tests (Ollama) ---

def _ollama_available() -> bool:
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not available")
def test_counter_ollama_qwen3_coder():
    from zrth.lean.magic_ai import TA2MagicAI

    magic = TA2MagicAI(
        COUNTER_SOURCE,
        model="qwen3-coder",
        base_url="http://localhost:11434/v1",
    )
    _assert_result(magic.infer(CertificateData(prp="x == 0")))
