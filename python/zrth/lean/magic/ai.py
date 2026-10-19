"""AI-based inference of invariants and ranking functions.

The model is reached through :mod:`.llm`.
"""

import re

from ..certificate import Claim, LeanAnswer
from ..common import Refused
from . import TA2Magic
from .llm import DEFAULT_MODEL, describe_preconditions, make_client

GENERATE_SYSTEM = """\
You are a formal verification expert. Given Lean4 source code of a reactive module \
and a property, your task is to find an inductive invariant and a ranking function \
that together prove that the property holds infinitely often (G(F(prp))).

The Lean4 module has:
- `init (e : ExtlNative) : CtrlNative` — computes the initial state from external inputs
- `update (s : CtrlNative × ExtlNative) : CtrlNative` — computes the next state

This encodes an infinite loop: initialize state, then repeatedly update.
`CtrlNative` is the type `init` returns in the source. With one state component it *is* that component, a matrix `Mat t 1 1 := Fin 1 → Fin 1 → t`, and its value is `s 0 0` — there is no `.1`. With several it is a tuple of them nested to the right, `A × (B × C)`: `s.1 0 0`, `s.2.1 0 0`, and the last one without a trailing `.1`, `s.2.2 0 0`.

The module may have PRECONDITIONS on its inputs:
- `init_pre`: constraint on the inputs to `init`
- `update_pre`: constraint on the inputs to `update` (external inputs, not state)

These preconditions are assumed to hold — you may rely on them when proving \
the invariant and ranking function.

An INDUCTIVE INVARIANT `inv : CtrlNative → Prop` must satisfy:
1. For all inputs satisfying init_pre: inv (init inputs) holds
2. For all states s and inputs satisfying update_pre: \
if inv s then inv (update (s, inputs)) holds

A RANKING FUNCTION `ranking : CtrlNative → ℕ` must satisfy:
- For all states s and inputs satisfying update_pre: \
if inv s ∧ ¬ prp s, then ranking (update (s, inputs)) < ranking s
- ranking s ≥ 0 always (guaranteed since it returns ℕ)

This proves that the system cannot stay in non-prp states forever, \
so prp must hold infinitely often.

Reply with EXACTLY this format (NO OTHER TEXT):
INVARIANT: <Lean4 expression of type CtrlNative → Prop>
RANKING: <Lean4 expression of type CtrlNative → ℕ>

Use Lean4 syntax, and access the state as described above. \
Use `∧`, `∨`, `¬` for logical connectives and `≤`, `<`, `=` for comparisons.

IMPORTANT — Lean 4 lambda syntax: write `fun s => expr` (NOT the Lean 3 form \
`λ s, expr` or `fun s, expr`). Do not use a comma after the binder.

Every state component is a matrix `Mat t 1 1`, so a scalar is always read with \
`0 0` after it: `s 0 0` for a one-component state, `s.1 0 0` for the first of \
several. The ranking function *MUST* return `Nat`.
"""

VERIFY_SYSTEM = """\
You are a formal verification auditor. You will be given:
1. Lean4 source code of a reactive module (init + update functions)
2. A property `prp` that should hold infinitely often
3. Preconditions on inputs (init_pre, update_pre) — assume these always hold
4. A proposed invariant and ranking function

`CtrlNative` is the type `init` returns in the source. With one state component it *is* that component, a matrix `Mat t 1 1 := Fin 1 → Fin 1 → t`, and its value is `s 0 0` — there is no `.1`. With several it is a tuple of them nested to the right, `A × (B × C)`: `s.1 0 0`, `s.2.1 0 0`, and the last one without a trailing `.1`, `s.2.2 0 0`.

Your job is to rigorously check whether the invariant and ranking function \
are correct. Specifically, check ALL of the following:

1. INIT: For all inputs satisfying init_pre, does inv (init inputs) hold?
2. INDUCTIVE: For all states s and inputs satisfying update_pre, \
if inv s holds, does inv (update (s, inputs)) hold?
3. RANKING DECREASE: For all states s and inputs satisfying update_pre, \
if inv s ∧ ¬ prp s, does ranking (update (s, inputs)) < ranking s?
4. RANKING NON-NEGATIVE: Is ranking s ≥ 0 always? (trivially true for ℕ)

Think through edge cases. Be thorough.

Reply with EXACTLY one of:
- CORRECT (if all checks pass)
- WRONG: <specific explanation of which condition fails and a concrete counterexample>

Put NO OTHER TEXT in the response.
"""


def _state_type_line(source: str) -> str:
    """The state type, named outright, as a line of the user message.

    The system prompt says how a state is read in general; this says which
    case this module is. Without it a model reached for `s.1 0 0` on a
    one-component state -- a projection out of a function, which Lean
    refuses -- on 16 of the bench matrix's single-wire modules.

    `init` may carry modifiers between the attribute and `def`: a
    Real-valued module's is `noncomputable`."""
    m = re.search(
        r"^@\[simp\]\s+(?:[a-z]+\s+)*def init\b[^\n]*?\)\s*:\s*(.+?)\s*:=\s*$",
        source, re.M,
    )
    if not m:
        return ""
    ty = m.group(1)
    shape = ("a tuple of components" if "×" in ty
             else "a single component, so its value is `s 0 0`")
    return f"CtrlNative, the state type, is `{ty}`: {shape}.\n\n"


def _unquote(value: str) -> str:
    """``value`` without the one pair of backticks a model wraps it in.

    The prompt writes Lean as inline code -- `` `fun s => expr` `` -- and a
    model answers in kind. Lean reads a backtick as the start of a name
    literal, so a quoted answer reaches `Data.lean` as
    ``def inv : T → Prop := `fun s => ...` `` and fails with `Function
    expected ... this term has type Lean.Name`, which says nothing about
    quoting. Only an enclosing pair is removed: a backtick inside the
    expression is Lean's and stays."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == "`":
        value = value[1:-1].strip()
    return value

# `s 0 0.toNat`. Lean reads `0.toNat` as a decimal literal whose fractional
# part is an identifier and says so -- "unexpected identifier after decimal
# point; consider parenthesizing the number" -- which is the repair, and
# there is only one parenthesisation that type-checks: the projection is of
# the *state element*, so the application closes before the dot. Measured on
# `m_max/OpMax`, where the model wrote `fun s => s 0 0.toNat` and meant
# `fun s => (s 0 0).toNat`; it writes the parenthesised form on nine other
# cells, so this is a slip rather than a misunderstanding.
_NUMERAL_FIELD = re.compile(
    r"(?P<head>\bs(?:\.\d+)*)\s+(?P<i>\d+)\s+(?P<j>\d+)\.(?P<field>[A-Za-z_][\w']*)"
)

# Names a model reaches for that Lean does not have. Unlike the one above
# there is no rewriting these: `Real.toNat` could mean a floor or a ceiling
# or a truncation, and picking one would be inventing the certificate rather
# than reading it. They go back to the model as feedback instead, which is
# the loop this route already has for a candidate that is wrong.
_ABSENT = {
    "Real.toNat": (
        "`Real.toNat` does not exist. A `Nat`-valued rank over a `Real` "
        "goes through the floor: `\u230ax\u230b.toNat`"
    ),
}


def _parenthesise_elements(value: str) -> str:
    """``s 0 0.toNat`` -> ``(s 0 0).toNat``, which is the only reading."""
    return _NUMERAL_FIELD.sub(r"(\g<head> \g<i> \g<j>).\g<field>", value)


def _absent_names(*values: str) -> str:
    """What Lean will refuse to *find* in these, before a build says so.

    Returned as the feedback the next attempt is given, because a name that
    is not there is a fact about Lean rather than about this module, and
    saying which name goes further than a parse error copied back.
    """
    said = [why for name, why in _ABSENT.items()
            if any(re.search(rf"\b{re.escape(name)}\b", v) for v in values)]
    return " ".join(said)


class TA2MagicAI(TA2Magic):
    """Infers invariants and ranking functions using an LLM.

    By default uses Claude via the Anthropic API (requires ANTHROPIC_API_KEY).
    Pass `base_url` to use a local LLM via any OpenAI-compatible API instead,
    e.g. base_url="http://localhost:11434/v1" for Ollama.
    """

    def __init__(
        self,
        source: str,
        model: str = DEFAULT_MODEL,
        max_attempts: int = 5,
        base_url: str | None = None,
        log=print,
    ):
        self.source = source
        self.max_attempts = max_attempts
        self.log = log
        self._chat = make_client(base_url, model)

    def infer(self, claim: Claim) -> LeanAnswer | None:
        """The certificate of `claim`, as Lean the model wrote and the
        auditor accepted; `None` when that reply holds neither predicate."""
        feedback = None
        for attempt in range(self.max_attempts):
            self.log(f"Generating invariant and ranking function (attempt {attempt})")
            inv, ranking = self._generate(claim, feedback)
            self.log("Candidates:")
            self.log(f"  inv: {inv}")
            self.log(f"  ranking: {ranking}")
            missing = _absent_names(inv, ranking)
            if missing:
                # Lean would reject this before any obligation is stated, so
                # there is nothing for `_verify` to be right or wrong about.
                self.log(f"  not Lean: {missing}")
                feedback = f"WRONG: that does not elaborate. {missing}"
                continue
            self.log("Cross-checking...")
            ok, feedback = self._verify(claim, inv, ranking)
            if ok:
                return LeanAnswer(inv, ranking) if inv or ranking else None
        # `Refused`, not `RuntimeError`: a route that asked the model
        # `max_attempts` times and did not get a certificate has given up, the
        # way every other route gives up, and that is a decision about the
        # module rather than a defect with a traceback to read.
        raise Refused(
            f"Failed to find valid invariant/ranking after {self.max_attempts} "
            f"attempts. Last feedback: {feedback}",
            searched=True,
        )

    def _generate(self, claim: Claim, feedback: str | None) -> tuple[str, str]:
        preconds = describe_preconditions(claim)
        user_msg = (
            f"Source code:\n```lean\n{self.source}\n```\n\n"
            f"{_state_type_line(self.source)}"
            f"Property (prp): {claim.prp}\n\n"
            f"{preconds}\n"
        )
        if feedback:
            user_msg += f"\nPrevious attempt was wrong. Feedback:\n{feedback}\n"
            user_msg += (
                "\nPlease try again with a corrected invariant and ranking function."
            )

        text = self._chat(GENERATE_SYSTEM, user_msg)
        return self._parse_generate_response(text)

    def _verify(
        self, claim: Claim, inv: str, ranking: str
    ) -> tuple[bool, str | None]:
        preconds = describe_preconditions(claim)
        user_msg = (
            f"Source code:\n```lean\n{self.source}\n```\n\n"
            f"{_state_type_line(self.source)}"
            f"Property (prp): {claim.prp}\n\n"
            f"{preconds}\n\n"
            f"Proposed invariant: {inv}\n"
            f"Proposed ranking function: {ranking}\n"
        )

        text = self._chat(VERIFY_SYSTEM, user_msg).strip()
        if text.startswith("CORRECT"):
            return True, None
        elif text.startswith("WRONG"):
            return False, text
        else:
            # The format of the answer was wrong. We could be more clever here,
            # but try to let the AI correct itself.
            user_msg = (
                f"You were checking invariants and ranking functions and answered\n\n{text}."
                "\n\nYou were required to answer either 'CORRECT' (a single word) or 'WRONG: <reason>' (a single line). Do it."
            )
            text = self._chat(VERIFY_SYSTEM, user_msg).strip()
            if text.startswith("CORRECT"):
                return True, None
            return False, text

    @staticmethod
    def _parse_generate_response(text: str) -> tuple[str, str]:
        inv = None
        ranking = None
        for line in text.strip().splitlines():
            line = line.strip()
            if line.startswith("INVARIANT:"):
                inv = _parenthesise_elements(_unquote(line[len("INVARIANT:"):]))
            elif line.startswith("RANKING:"):
                ranking = _parenthesise_elements(
                    _unquote(line[len("RANKING:"):]))
        if inv is None or ranking is None:
            raise ValueError(f"Failed to parse AI response:\n{text}")
        return inv, ranking
