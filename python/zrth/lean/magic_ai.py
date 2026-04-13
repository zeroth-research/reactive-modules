"""AI-based inference of invariants and ranking functions.

Requires the 'ai' optional dependency: pip install zrth[ai]
"""

from .cert import CertificateData
from .magic import TA2Magic

try:
    import anthropic
except ImportError:
    anthropic = None

GENERATE_SYSTEM = """\
You are a formal verification expert. Given the source code of a reactive module \
and a property, your task is to find an inductive invariant and a ranking function \
that together prove that the property holds infinitely often (G(F(prp))).

A reactive module has:
- `init(...)`: returns the initial state
- `update(old_state, ...)`: returns the new state each iteration

This encodes an infinite loop: initialize state, then repeatedly update.

The module may have PRECONDITIONS on its inputs:
- `init_pre`: constraint on the inputs to `init`
- `update_pre`: constraint on the inputs to `update` (external inputs, not state)

These preconditions are assumed to hold — you may rely on them when proving \
the invariant and ranking function.

An INDUCTIVE INVARIANT `inv(state)` must satisfy:
1. For all inputs satisfying init_pre: inv(init(inputs)) is true
2. For all states and inputs satisfying update_pre: \
if inv(state) then inv(update(state, inputs)) is true

A RANKING FUNCTION `ranking(state) -> Nat` must satisfy:
- For all states and inputs satisfying update_pre: \
if inv(state) AND NOT prp(state), then ranking(update(state, inputs)) < ranking(state)
- ranking(state) >= 0 (always, since it returns Nat)

This proves that the system cannot stay in non-prp states forever, \
so prp must hold infinitely often.

Reply with EXACTLY this format (NO OTHER TEXT):
INVARIANT: <expression using state variables>
RANKING: <expression using state variables>

Use Python-like syntax for the expressions. State variables are the return values \
of init/update. Use `and`, `or`, `not` for logical connectives, `>=`, `<=`, `==` \
for comparisons.
"""

VERIFY_SYSTEM = """\
You are a formal verification auditor. You will be given:
1. Source code of a reactive module (init + update functions)
2. A property `prp` that should hold infinitely often
3. Preconditions on inputs (init_pre, update_pre) — assume these always hold
4. A proposed invariant and ranking function

Your job is to rigorously check whether the invariant and ranking function \
are correct. Specifically, check ALL of the following:

1. INIT: For all inputs satisfying init_pre, does inv(init(inputs)) hold?
2. INDUCTIVE: For all states and inputs satisfying update_pre, \
if inv(state) holds, does inv(update(state, inputs)) hold?
3. RANKING DECREASE: For all states and inputs satisfying update_pre, \
if inv(state) AND NOT prp(state), \
does ranking(update(state, inputs)) < ranking(state)?
4. RANKING NON-NEGATIVE: Is ranking(state) >= 0 always?

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


class TA2MagicAI(TA2Magic):
    """Infers invariants and ranking functions using Claude."""

    def __init__(
        self, source: str, model: str = "claude-sonnet-4-6", max_attempts: int = 5
    ):
        super().__init__(source)
        if anthropic is None:
            raise ImportError(
                "anthropic package is required for TA2MagicAI. "
                "Install with: pip install zrth[ai]"
            )
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_attempts = max_attempts

    def infer(self, cd: CertificateData) -> CertificateData:
        feedback = None
        for attempt in range(self.max_attempts):
            print(f"Generating invariant and ranking funciton (attempt {attempt})")
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

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=GENERATE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        text = response.content[0].text
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

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=VERIFY_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        text = response.content[0].text.strip()
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
