"""The LLM prompt of `--infer ai-cegis`, and its reply read back.

`prompt_inv_ranking(env, chat, source, prp, preconds, feedback)` formats
the request -- the module's state and transition as cvc5 sees them, what
earlier runs established, the last counterexample -- and parses the reply
into a `PromptResult`: the invariant and ranking as SMT-LIB source, and as
terms read by the run's `smt.predicates.PredicateParser`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvc5

from ...smt.predicates import PredicateParser, parse_predicate


CEGIS_GENERATE_SYSTEM = """\
You are a formal verification expert. Given a reactive module and a \
property, propose an inductive invariant and a ranking function, both \
as pure SMT-LIB 2 *expressions* (not lambdas, not definitions).

The module's state components are already declared as top-level SMT \
constants named `s0`, `s1`, …, `sN-1`. You MUST refer to them by these \
exact names. Do NOT introduce new variables, do NOT write a binder \
(no `lambda`, no `fun`, no `let`). Your expression's free variables \
must be a subset of {s0, s1, …, sN-1}.

Reply with EXACTLY this format (NO other text, NO code fences, NO \
markdown):
INVARIANT: <SMT-LIB 2 expression of sort Bool>
RANKING: <SMT-LIB 2 expression of sort Int>

Examples of valid expressions:
  (and (= s0 false) (<= s1 10))
  (ite (= s0 true) 0 (- 10 s1))

The ranking expression must have sort **Int**, and wherever the \
invariant holds and the property does not, it must be at least 1 and \
strictly smaller after the step. Where the property holds it is \
unconstrained, negative included. If a state component is \
Real, wrap it in `(to_int ...)` to convert to Int; Bool components \
can be mapped via `(ite s_k 1 0)`.

Use standard SMT-LIB syntax: (and ...), (or ...), (not ...), (= ...), \
(< ...), (<= ...), (+ ...), (- ...), (* ...), (ite cond a b). \
Tuple fields are accessed with `((_ tuple.select k) t)`.
"""


CEGIS_SAFETY_SYSTEM = """\
You are a formal verification expert. Given a reactive module and a \
property, propose an inductive invariant as a pure SMT-LIB 2 *expression* \
(not a lambda, not a definition).

The invariant must satisfy all three of:
1. it holds in every initial state (for inputs satisfying init_pre),
2. it is preserved by every update (for inputs satisfying update_pre),
3. it IMPLIES the property: every state satisfying the invariant also \
satisfies the property.

Point 3 is what makes this a safety proof. The property must hold in \
*every* reachable state, so the invariant has to be strong enough to rule \
out every state where the property is false. There is NO ranking function \
on this route -- do not propose one. If the property is already inductive, \
the invariant may simply be the property itself; more often it has to be \
strengthened with the extra facts that make it inductive.

The module's state components are already declared as top-level SMT \
constants named `s0`, `s1`, …, `sN-1`. You MUST refer to them by these \
exact names. Do NOT introduce new variables, do NOT write a binder \
(no `lambda`, no `fun`, no `let`). Your expression's free variables \
must be a subset of {s0, s1, …, sN-1}.

Reply with EXACTLY this format (NO other text, NO code fences, NO \
markdown):
INVARIANT: <SMT-LIB 2 expression of sort Bool>

Examples of valid expressions:
  (and (>= s0 0) (<= s0 100))
  (or (not s0) (= s1 0))

Use standard SMT-LIB syntax: (and ...), (or ...), (not ...), (= ...), \
(< ...), (<= ...), (+ ...), (- ...), (* ...), (ite cond a b). \
Tuple fields are accessed with `((_ tuple.select k) t)`.
"""


@dataclass
class PromptResult:
    inv_src: str
    # `None` on the safety route: `rule_globally` takes no ranking function,
    # so none is asked for and none comes back.
    ranking_src: str | None
    inv_term: cvc5.Term
    ranking_term: cvc5.Term | None


def state_description(env: PredicateParser) -> str:
    lines = ["State components (SMT sorts):"]
    for i, (w, sort) in enumerate(zip(env.msmt.ctrl_next, env.state_sorts)):
        lines.append(f"  s{i} : {sort}   # {w.dtype}")
    if env.extl_next_sorts or env.extl_latched_sorts:
        lines.append("External inputs (SMT sorts):")
        for i, sort in enumerate(env.extl_next_sorts):
            lines.append(f"  e{i}  : {sort}   # next")
        for i, sort in enumerate(env.extl_latched_sorts):
            lines.append(f"  el{i} : {sort}   # latched")
    return "\n".join(lines)


def transition_description(env: PredicateParser) -> str:
    """Symbolic transition formulas for the module.

    Renders `init(en)` and `update(sL, el, en)` as SMT expressions so
    the LLM can see the exact update semantics without reverse-
    engineering from source. Here `sL` = latched state (same variable
    space as `s0..sN-1` we ask the invariant about).
    """
    lines = ["Initial state (s_i after init, given extl-next e):"]
    init_vals = env.msmt.init_state(env.extl_next_vars)
    for i, t in enumerate(init_vals):
        lines.append(f"  s{i}_init = {t}")
    lines.append(
        "Update: latched state is s_i, latched inputs el_i, next inputs e_i; "
        "new state s_i' is:"
    )
    next_vals = env.msmt.update_state(
        env.state_vars, env.extl_latched_vars, env.extl_next_vars
    )
    for i, t in enumerate(next_vals):
        lines.append(f"  s{i}' = {t}")
    return "\n".join(lines)


class PromptParseError(ValueError):
    """Raised when the LLM reply fails to parse into valid SMT terms."""


def parse_llm_reply(
    text: str,
    *,
    require_inv: bool = True,
    require_ranking: bool = True,
) -> tuple[str | None, str | None]:
    """Extract INVARIANT/RANKING source lines from the LLM reply.

    Raises `ValueError` if a *required* field is missing.
    """
    inv_src = ranking_src = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("INVARIANT:"):
            inv_src = s[len("INVARIANT:") :].strip()
        elif s.startswith("RANKING:"):
            ranking_src = s[len("RANKING:") :].strip()
    missing = []
    if require_inv and inv_src is None:
        missing.append("INVARIANT")
    if require_ranking and ranking_src is None:
        missing.append("RANKING")
    if missing:
        raise ValueError(
            f"LLM reply missing {'/'.join(missing)} line(s). Got:\n{text}"
        )
    return inv_src, ranking_src


def prompt_inv_ranking(
    env: PredicateParser,
    chat,
    source: str,
    prp: str | None,
    preconds: str,
    feedback: str | None = None,
    *,
    fixed_inv_src: str | None = None,
    fixed_ranking_src: str | None = None,
    kind: str = "buchi",
    known: tuple[str, ...] = (),
) -> PromptResult:
    """Call `chat` with the CEGIS prompt, parse reply into SMT terms.

    If `fixed_inv_src` / `fixed_ranking_src` is supplied, that field is
    treated as a user-provided input: not requested from the LLM, and
    shown in the prompt so the LLM sees it while searching for the other.
    If both are fixed, no LLM call is made.

    `known` is what earlier runs *proved* about the shape of the answer --
    that no ranking function linear in the state works under this invariant,
    say. It is stated before the question rather than as feedback, because
    it is not about an attempt: it is a fact about the module that holds
    whatever this model proposes, and the one thing worth knowing before
    proposing anything.

    `kind="safety"` asks for an invariant alone. `rule_globally` takes no
    ranking function, so there is none to propose; what the invariant has
    to do instead is *imply* the property rather than merely be re-reached,
    and the system prompt says so. A fixed invariant is then the whole
    certificate, and no LLM call is made.
    """
    safety = kind == "safety"
    if safety:
        fixed_ranking_src = None
    if fixed_inv_src is not None and (safety or fixed_ranking_src is not None):
        return PromptResult(
            fixed_inv_src,
            fixed_ranking_src,
            parse_predicate(env, fixed_inv_src),
            None if fixed_ranking_src is None else parse_predicate(env, fixed_ranking_src),
        )
    goal = (
        "Property (prp), to hold in EVERY reachable state"
        if safety
        else "Property (prp), to hold infinitely often"
    )
    user_msg = (
        f"Source code:\n```lean\n{source}\n```\n\n"
        f"{goal}: {prp}\n\n"
        f"{preconds}\n\n"
        f"{state_description(env)}\n\n"
        f"{transition_description(env)}\n"
    )
    if known:
        # Numbered and quoted as established, not as advice: each line was
        # written by a decision procedure that refuted a whole space, and the
        # model's job is to propose something outside it.
        established = "\n".join(f"{i}. {text.strip()}"
                                 for i, text in enumerate(known, 1))
        user_msg += (
            "\nAlready established about this module by a decision procedure "
            "(cvc5), in earlier runs. These are proofs that a space is empty, "
            "not opinions -- do not propose anything of a shape ruled out "
            "here:\n"
            f"{established}\n"
        )
    if fixed_inv_src is not None:
        user_msg += (
            f"\nThe invariant is GIVEN (do not propose a new one):\n"
            f"INVARIANT: {fixed_inv_src}\n"
            "Reply with ONLY a `RANKING:` line.\n"
        )
    elif fixed_ranking_src is not None:
        user_msg += (
            f"\nThe ranking is GIVEN (do not propose a new one):\n"
            f"RANKING: {fixed_ranking_src}\n"
            "Reply with ONLY an `INVARIANT:` line.\n"
        )
    if feedback:
        what = "invariant" if safety else "invariant and ranking"
        user_msg += (
            "\nPrevious attempt was rejected. Feedback:\n"
            f"{feedback}\n"
            f"Please try again with a corrected {what}.\n"
        )
    text = chat(CEGIS_SAFETY_SYSTEM if safety else CEGIS_GENERATE_SYSTEM, user_msg)
    inv_src, ranking_src = parse_llm_reply(
        text,
        require_inv=fixed_inv_src is None,
        require_ranking=not safety and fixed_ranking_src is None,
    )
    if fixed_inv_src is not None:
        inv_src = fixed_inv_src
    if fixed_ranking_src is not None:
        ranking_src = fixed_ranking_src
    try:
        inv_term = parse_predicate(env, inv_src)
    except RuntimeError as e:
        raise PromptParseError(
            f"Could not parse INVARIANT `{inv_src}`: {e}. "
            f"Use only s0..s{len(env.state_vars) - 1}."
        ) from e
    if str(inv_term.getSort()) != "Bool":
        raise PromptParseError(
            f"INVARIANT must have sort Bool, got {inv_term.getSort()}."
        )
    if safety:
        # An LLM that volunteered a RANKING anyway has offered something
        # this route has no obligation to state; drop it rather than carry
        # a term nothing checks.
        return PromptResult(inv_src, None, inv_term, None)
    try:
        ranking_term = parse_predicate(env, ranking_src)
    except RuntimeError as e:
        raise PromptParseError(
            f"Could not parse RANKING `{ranking_src}`: {e}. "
            f"Use only s0..s{len(env.state_vars) - 1}."
        ) from e
    if str(ranking_term.getSort()) != "Int":
        raise PromptParseError(
            f"RANKING must have sort Int, got {ranking_term.getSort()}."
        )
    return PromptResult(inv_src, ranking_src, inv_term, ranking_term)
