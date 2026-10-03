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


# ── a refusal is not a search ────────────────────────────────────────────

from run_matrix import gave_up, verdict                        # noqa: E402

NOT_BUILT = dict(ok=False, raw_ok=False, secs=0.0, targets=[], sorries=[],
                 errors=["(not built: generation failed)"])


def _gen(err, gave_up=""):
    return dict(ok=False, secs=0.5, err=err, err_full=err, gave_up=gave_up)


def test_a_declined_module_is_unsupported_not_no_cert():
    """Nothing was searched, so it belongs in no route's denominator."""
    err = "error: --infer: --infer nuterm cannot read this module: wire 0 has sort Real([1,1])"
    assert verdict(_gen(err, "declined"), NOT_BUILT) == "UNSUPPORTED"
    assert verdict(_gen(err, "unknown"), NOT_BUILT) == "NO-CERT"


def test_a_space_proved_empty_stays_no_cert():
    err = "error: --infer: --infer smt-linear found no ranking function."
    assert verdict(_gen(err, "no_solution"), NOT_BUILT) == "NO-CERT"


def test_a_refusal_before_any_project_exists_is_a_decline(tmp_path):
    """`fbk-proveit`'s precheck refuses before a project is written, so there
    is no `artifacts/` for a status -- and a precheck searches nothing."""
    out = tmp_path / "cell"
    blob = "error: --fbk-proveit: state element type(s) Real unsupported"
    assert gave_up(out, blob) == "declined"


def test_a_usage_error_is_not_a_decline(tmp_path):
    blob = "usage: verith [-h] ...\nverith: error: unrecognized arguments: --nope"
    assert gave_up(tmp_path / "cell", blob) == ""


def test_a_project_that_exists_is_read_for_its_own_note(tmp_path):
    """Once there is a project, the route's note is the answer, not the
    absence of one."""
    (tmp_path / "cell" / "Rea").mkdir(parents=True)
    assert gave_up(tmp_path / "cell", "error: --infer: no luck") == ""
