"""AI-based inference of invariants and ranking functions.

For Claude API: pip install zrth[ai]
For local LLMs (Ollama, vLLM, etc.): pip install zrth[ai-local]
"""

from .cert import CertificateData
from .magic import TA2Magic

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
State components are accessed via `.1`, `.2.1`, `.2.2.1`, etc. (left-nested tuples).

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

Use Lean4 syntax. Access state components with `.1`, `.2.1`, etc. \
Use `∧`, `∨`, `¬` for logical connectives and `≤`, `<`, `=` for comparisons.

IMPORTANT — Lean 4 lambda syntax: write `fun s => expr` (NOT the Lean 3 form \
`λ s, expr` or `fun s, expr`). Do not use a comma after the binder.

State components are matrices of type `Mat t 1 1 := Fin 1 → Fin 1 → t`, so a \
scalar value is accessed as `s.1 0 0`, `s.2.1 0 0`, etc. The ranking function \
must return `Nat`; use `.toNat` or explicit conversion if a component is Int/Real.
"""

VERIFY_SYSTEM = """\
You are a formal verification auditor. You will be given:
1. Lean4 source code of a reactive module (init + update functions)
2. A property `prp` that should hold infinitely often
3. Preconditions on inputs (init_pre, update_pre) — assume these always hold
4. A proposed invariant and ranking function

The state type is `CtrlNative` (a left-nested tuple). \
Components are accessed via `.1`, `.2.1`, `.2.2.1`, etc.

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
        client = openai.OpenAI(base_url=base_url, api_key="unused")

        def chat(system: str, user: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=1024,
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
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
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
                inv = line[len("INVARIANT:") :].strip()
            elif line.startswith("RANKING:"):
                ranking = line[len("RANKING:") :].strip()
        if inv is None or ranking is None:
            raise ValueError(f"Failed to parse AI response:\n{text}")
        return inv, ranking
