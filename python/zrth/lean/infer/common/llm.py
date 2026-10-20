"""The language model the LLM routes (`--infer ai`, `--infer ai-cegis`) ask.

For Claude API: pip install zrth[ai]
For local LLMs (Ollama, vLLM, etc.) and OpenAI-compatible providers
(OpenRouter, ...): pip install zrth[ai-local]

Neither package is imported until a client is made: `cli` imports
`DEFAULT_MODEL`.
"""

import os

from ...core.certificate import Claim
from ...core.errors import Refused

DEFAULT_MODEL = "claude-sonnet-4-6"

# Room for the answer *after* the reasoning. The generate prompt asks for two
# lines, but a model reasons in prose first and the lines come last, so a cap
# sized for the answer cuts the reply before it: at 1024 tokens 56 of the
# bench matrix's `--infer ai` cells ended mid-sentence, some inside
# `INVARIANT: fun s => s`, and failed to parse. Kept below the size at which
# the Anthropic SDK insists on streaming.
MAX_TOKENS = 8192


def reply_fields(text: str, *fields: str) -> dict[str, str]:
    """The `FIELD: value` lines of a reply, by field, values stripped.

    Both LLM routes ask for their answer as such lines after whatever the
    model reasons first, so a line is read wherever it is and the last of
    each field wins: a model that corrects itself writes the correction
    after the mistake. A field the reply does not give is absent.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        for f in fields:
            if s.startswith(f"{f}:"):
                out[f] = s[len(f) + 1:].strip()
    return out


def describe_preconditions(claim: Claim) -> str:
    """`claim`'s preconditions, as the line of a prompt that states them."""
    parts = []
    if claim.init_pre is not None:
        parts.append(f"Precondition on init inputs (init_pre): {claim.init_pre}")
    if claim.update_pre is not None:
        parts.append(f"Precondition on update inputs (update_pre): {claim.update_pre}")
    if not parts:
        return "No preconditions on inputs."
    return "\n".join(parts)


def make_client(base_url: str | None, model: str):
    """Return a chat callable: (system, user) -> str."""
    if base_url is not None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package is required for local LLM support. "
                "Install with: pip install zrth[ai-local]"
            ) from None
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
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for the LLM routes. "
                "Install with: pip install zrth[ai]"
            ) from None
        # Without a key the request itself fails with a TypeError about
        # header resolution, which says nothing about the flag that needs
        # one. A run that never asks the model never gets here: the routes
        # build their client on the first question.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise Refused(
                "ANTHROPIC_API_KEY is not set, and the default model is "
                "reached through the Anthropic API. Set it, or point "
                "--base-url at an OpenAI-compatible server (a local one "
                "needs no key)."
            )
        client = anthropic.Anthropic()

        def chat(system: str, user: str) -> str:
            resp = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text

        return chat
