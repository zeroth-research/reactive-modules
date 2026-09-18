"""What a failed run's output is reduced to, for the log and for the page.

A cell that fails is read twice: once as a line in the harness log, and once
as the panel the page opens. Both came from the same reduction -- the last
eight lines of whatever the process printed -- which is right for a refusal
`verith` wrote and wrong for a Python traceback, where the last eight lines
are the frames and the exception's own message is what they lead to.

44 of the matrix's cells are tracebacks, and on the LLM routes the whole
output opens with an SDK warning about credentials that has nothing to do
with the failure -- so the page's panel led with it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_matrix import diagnosis                               # noqa: E402


TRACEBACK = '''ANTHROPIC_API_KEY is set and takes precedence over the SDK's profile.
Traceback (most recent call last):
  File "/x/bin/verith", line 10, in <module>
    sys.exit(main())
  File "/x/zrth/lean/magic_cegar.py", line 177, in infer
    raise RuntimeError(
        f"CEGAR failed after {self.max_attempts} attempts."
    )
RuntimeError: CEGAR failed after 5 attempts. Last feedback:
inv_imp_P: obligation violated. Counterexample:
  s[0] = 51'''

REFUSAL = """error: --infer: --infer sygus found no invariant. No inductive
invariant implying the property is a conjunction of at most 3 atoms.
`--infer ai-cegis` searches no fixed space and is what to try next."""


def test_a_traceback_is_reduced_to_what_was_raised():
    out = diagnosis(TRACEBACK)
    assert out.startswith("RuntimeError: CEGAR failed after 5 attempts")
    assert "File " not in out and "raise RuntimeError" not in out


def test_the_exception_keeps_the_lines_its_own_message_runs_on_to():
    # The counterexample is the useful half and it is indented, so a rule
    # that stopped at the first line would throw away the answer.
    out = diagnosis(TRACEBACK)
    assert "inv_imp_P: obligation violated" in out
    assert "s[0] = 51" in out


def test_a_warning_printed_before_the_failure_is_not_the_failure():
    assert "ANTHROPIC_API_KEY" not in diagnosis(TRACEBACK)


def test_a_refusal_is_left_as_it_was():
    """Prose `verith` wrote is already the diagnosis, hint and all."""
    out = diagnosis(REFUSAL)
    assert out == " | ".join(ln.strip() for ln in REFUSAL.splitlines() if ln.strip())


def test_a_traceback_with_no_exception_line_still_says_something():
    assert diagnosis('Traceback (most recent call last):\n  File "x", line 1') != ""
