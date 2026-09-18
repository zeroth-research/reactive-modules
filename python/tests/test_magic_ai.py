"""Tests for AI-based invariant/ranking inference."""

import os
import subprocess
import pytest

from zrth.lean.cert import CertificateData
from zrth.lean.magic.ai import TA2MagicAI

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

# --- what the model is told about the state (no model) ---

ONE_WIRE_INIT = "@[simp] def init (extl_n: Unit) : (Mat Int 1 1) :=\n  x0\n"
THREE_WIRE_INIT = ("@[simp] def init (extl_n: (Mat Int 1 1) × (Mat Int 1 1)) : "
                   "(Mat Int 1 1) × (Mat Int 1 1) × (Mat Int 1 1) :=\n  x0\n")


def test_a_one_component_state_is_named_as_the_matrix_itself():
    """`s.1` on a single `Mat` is a projection out of a function, which Lean
    refuses; the message has to say this state has no `.1`."""
    from zrth.lean.magic.ai import _state_type_line
    line = _state_type_line(ONE_WIRE_INIT)
    assert "`(Mat Int 1 1)`" in line
    assert "`s 0 0`" in line


def test_a_tuple_state_is_named_with_its_whole_type():
    from zrth.lean.magic.ai import _state_type_line
    line = _state_type_line(THREE_WIRE_INIT)
    assert "`(Mat Int 1 1) × (Mat Int 1 1) × (Mat Int 1 1)`" in line
    assert "tuple" in line


def test_no_prompt_says_every_state_is_read_through_dot_one():
    from zrth.lean.magic.ai import GENERATE_SYSTEM, VERIFY_SYSTEM
    for prompt in (GENERATE_SYSTEM, VERIFY_SYSTEM):
        assert "there is no `.1`" in prompt
        assert "accessed via `.1`, `.2.1`, `.2.2.1`" not in prompt

# --- Claude API tests ---

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_counter_claude():
    from zrth.lean.magic.ai import TA2MagicAI

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
    from zrth.lean.magic.ai import TA2MagicAI

    magic = TA2MagicAI(
        COUNTER_SOURCE,
        model="qwen3-coder",
        base_url="http://localhost:11434/v1",
    )
    _assert_result(magic.infer(CertificateData(prp="x == 0")))
