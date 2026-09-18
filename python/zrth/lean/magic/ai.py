"""AI-based inference of invariants and ranking functions.

For Claude API: pip install zrth[ai]
For local LLMs (Ollama, vLLM, etc.) and OpenAI-compatible providers
(OpenRouter, ...): pip install zrth[ai-local]
"""

import os
import re

from ..cert import CertificateData
from ..common import Refused
from . import TA2Magic

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

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


# Room for the answer *after* the reasoning. The generate prompt asks for two
# lines, but a model reasons in prose first and the lines come last, so a cap
# sized for the answer cuts the reply before it: at 1024 tokens 56 of the
# bench matrix's `--infer ai` cells ended mid-sentence, some inside
# `INVARIANT: fun s => s`, and failed to parse. Kept below the size at which
# the Anthropic SDK insists on streaming.
MAX_TOKENS = 8192

def _state_type_line(source: str) -> str:
    """The state type, named outright, as a line of the user message.

    The system prompt says how a state is read in general; this says which
    case this module is. Without it a model reached for `s.1 0 0` on a
    one-component state -- a projection out of a function, which Lean
    refuses -- on 16 of the bench matrix's single-wire modules."""
    m = re.search(r"^@\[simp\] def init\b[^\n]*?\)\s*:\s*(.+?)\s*:=\s*$", source, re.M)
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

def _describe_preconditions(cd: CertificateData) -> str:
    parts = []
    if cd.init_pre is not None:
        parts.append(f"Precondition on init inputs (init_pre): {cd.init_pre}")
    if cd.update_pre is not None:
        parts.append(f"Precondition on update inputs (update_pre): {cd.update_pre}")
    if not parts:
        return "No preconditions on inputs."
    return "\n".join(parts)


def _make_client(base_url: str | None, model: str):
    """Return a chat callable: (system, user) -> str."""
    if base_url is not None:
        if openai is None:
            raise ImportError(
                "openai package is required for local LLM support. "
                "Install with: pip install zrth[ai-local]"
            )
        # Hosted OpenAI-compatible providers (e.g. OpenRouter) need a real
        # key; local servers (Ollama, vLLM) accept any placeholder.
        api_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "unused"
        )
        client = openai.OpenAI(base_url=base_url, api_key=api_key)

        def chat(system: str, user: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content

        return chat
    else:
        if anthropic is None:
            raise ImportError(
                "anthropic package is required for TA2MagicAI. "
                "Install with: pip install zrth[ai]"
            )
        client = anthropic.Anthropic()

        def chat(system: str, user: str) -> str:
            # Checked here rather than around the constructor: a CEGAR run
            # with the invariant fixed builds a client and never calls it,
            # and that run needs no key. Without a key the request itself
            # fails with a TypeError about header resolution, which says
            # nothing about the flag that needs one.
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise Refused(
                    "ANTHROPIC_API_KEY is not set, and the default model is "
                    "reached through the Anthropic API. Set it, or point "
                    "--base-url at an OpenAI-compatible server (a local one "
                    "needs no key)."
                )
            resp = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text

        return chat


class TA2MagicAI(TA2Magic):
    """Infers invariants and ranking functions using an LLM.

    By default uses Claude via the Anthropic API (requires ANTHROPIC_API_KEY).
    Pass `base_url` to use a local LLM via any OpenAI-compatible API instead,
    e.g. base_url="http://localhost:11434/v1" for Ollama.
    """

    def __init__(
        self,
        source: str,
        model: str = "claude-sonnet-4-6",
        max_attempts: int = 5,
        base_url: str | None = None,
    ):
        super().__init__(source)
        self.max_attempts = max_attempts
        self._chat = _make_client(base_url, model)

    def infer(self, cd: CertificateData) -> CertificateData:
        feedback = None
        for attempt in range(self.max_attempts):
            print(f"Generating invariant and ranking function (attempt {attempt})")
            inv, ranking = self._generate(cd, feedback)
            print("Candidates:")
            print(f"  inv: {inv}")
            print(f"  ranking: {ranking}")
            print("Cross-checking...")
            ok, feedback = self._verify(cd, inv, ranking)
            if ok:
                cd.inv = inv
                cd.ranking = ranking
                return cd
        raise RuntimeError(
            f"Failed to find valid invariant/ranking after {self.max_attempts} attempts. "
            f"Last feedback: {feedback}"
        )

    def _generate(self, cd: CertificateData, feedback: str | None) -> tuple[str, str]:
        preconds = _describe_preconditions(cd)
        user_msg = (
            f"Source code:\n```python\n{self.source}\n```\n\n"
            f"{_state_type_line(self.source)}"
            f"Property (prp): {cd.prp}\n\n"
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
        self, cd: CertificateData, inv: str, ranking: str
    ) -> tuple[bool, str | None]:
        preconds = _describe_preconditions(cd)
        user_msg = (
            f"Source code:\n```python\n{self.source}\n```\n\n"
            f"{_state_type_line(self.source)}"
            f"Property (prp): {cd.prp}\n\n"
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
                inv = _unquote(line[len("INVARIANT:") :])
            elif line.startswith("RANKING:"):
                ranking = _unquote(line[len("RANKING:") :])
        if inv is None or ranking is None:
            raise ValueError(f"Failed to parse AI response:\n{text}")
        return inv, ranking
