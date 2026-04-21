"""LLM prompt + SMT-LIB parser for the CEGAR driver.

`CegarPromptEnv` owns the cvc5 parser machinery: it declares state
variables `s0..sN-1` once (sorted per ctrl-next wire), and exposes
`parse_expr(smt_lib: str) → cvc5.Term` that resolves references to those
state vars back to the cvc5 consts stored on `state_vars`.

`prompt_inv_ranking(chat, source, prp, preconds, feedback)` formats the
LLM request and parses the reply into a `(inv_src, ranking_src, inv_term,
ranking_term)` tuple.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvc5
from cvc5 import Kind

from .smt_encode import wire_sort
from .smt_module import ModuleSMT


# ---------------------------------------------------------------------
#   Python-expression DSL over cvc5 terms
# ---------------------------------------------------------------------


class _E:
    """Operator-overloading wrapper around a cvc5 Term.

    Lets the user write `s0 > 0`, `s0 == s4`, `a + b`, `a & b` etc. in
    Python, producing cvc5 Terms under the hood. Used as a convenience
    layer for `--property` / `--pre` / `--invariant` / `--ranking`.
    """

    __slots__ = ("_tm", "_t")

    def __init__(self, tm: cvc5.TermManager, t: cvc5.Term):
        self._tm = tm
        self._t = t

    @property
    def t(self) -> cvc5.Term:
        return self._t

    def _coerce(self, x):
        if isinstance(x, _E):
            return x._t
        if isinstance(x, bool):
            return self._tm.mkBoolean(x)
        if isinstance(x, int):
            return self._tm.mkInteger(x)
        if isinstance(x, float):
            return self._tm.mkReal(x)
        if isinstance(x, cvc5.Term):
            return x
        raise TypeError(f"Cannot coerce {type(x).__name__} to a cvc5 Term")

    def _bin(self, kind, other, rev=False):
        a, b = self._coerce(other), self._t
        if not rev:
            a, b = b, a
        return _E(self._tm, self._tm.mkTerm(kind, a, b))

    def __gt__(self, o):  return self._bin(Kind.GT, o)
    def __lt__(self, o):  return self._bin(Kind.LT, o)
    def __ge__(self, o):  return self._bin(Kind.GEQ, o)
    def __le__(self, o):  return self._bin(Kind.LEQ, o)
    def __eq__(self, o):  return self._bin(Kind.EQUAL, o)
    def __ne__(self, o):  return self._bin(Kind.DISTINCT, o)
    def __add__(self, o): return self._bin(Kind.ADD, o)
    def __radd__(self, o): return self._bin(Kind.ADD, o, rev=True)
    def __sub__(self, o): return self._bin(Kind.SUB, o)
    def __rsub__(self, o): return self._bin(Kind.SUB, o, rev=True)
    def __mul__(self, o): return self._bin(Kind.MULT, o)
    def __rmul__(self, o): return self._bin(Kind.MULT, o, rev=True)
    def __and__(self, o): return self._bin(Kind.AND, o)
    def __rand__(self, o): return self._bin(Kind.AND, o, rev=True)
    def __or__(self, o):  return self._bin(Kind.OR, o)
    def __ror__(self, o): return self._bin(Kind.OR, o, rev=True)
    def __neg__(self):    return _E(self._tm, self._tm.mkTerm(Kind.NEG, self._t))
    def __invert__(self): return _E(self._tm, self._tm.mkTerm(Kind.NOT, self._t))
    def __hash__(self):   return id(self._t)

    # Tuple (matrix) element selection: `e0[k]` on a tuple-sorted var.
    def __getitem__(self, k: int) -> "_E":
        if not isinstance(k, int):
            raise TypeError("index must be int")
        sort = self._t.getSort()
        if not sort.isTuple():
            raise TypeError(f"{self._t} is not a tuple; cannot index")
        ctor = sort.getDatatype()[0]
        sel = ctor[k].getTerm()
        return _E(self._tm, self._tm.mkTerm(Kind.APPLY_SELECTOR, sel, self._t))


def _pyeval_namespace(env: "CegarPromptEnv") -> dict:
    """Build the namespace exposed to Python-expression predicates."""
    tm = env.tm

    def _unwrap(x):
        if isinstance(x, _E):
            return x._t
        if isinstance(x, bool):
            return tm.mkBoolean(x)
        if isinstance(x, int):
            return tm.mkInteger(x)
        if isinstance(x, float):
            return tm.mkReal(x)
        return x

    def _mk(kind, *args):
        return _E(tm, tm.mkTerm(kind, *[_unwrap(a) for a in args]))

    ns = {
        "__builtins__": {},
        "And": lambda *a: _mk(Kind.AND, *a),
        "Or": lambda *a: _mk(Kind.OR, *a),
        "Not": lambda a: _mk(Kind.NOT, a),
        "Ite": lambda c, t, e: _mk(Kind.ITE, c, t, e),
        "Implies": lambda a, b: _mk(Kind.IMPLIES, a, b),
        "Eq": lambda a, b: _mk(Kind.EQUAL, a, b),
        "Ne": lambda a, b: _mk(Kind.DISTINCT, a, b),
        "true": _E(tm, tm.mkBoolean(True)),
        "false": _E(tm, tm.mkBoolean(False)),
        "True": _E(tm, tm.mkBoolean(True)),
        "False": _E(tm, tm.mkBoolean(False)),
        "ToInt": lambda a: _mk(Kind.TO_INTEGER, a),
    }
    for i, v in enumerate(env.state_vars):
        ns[f"s{i}"] = _E(tm, v)
    for i, v in enumerate(env.extl_next_vars):
        ns[f"e{i}"] = _E(tm, v)
    for i, v in enumerate(env.extl_latched_vars):
        ns[f"el{i}"] = _E(tm, v)
    return ns


def parse_predicate(env: "CegarPromptEnv", src: str) -> cvc5.Term:
    """Parse `src` as a cvc5 predicate. Tries Python expression first,
    falls back to SMT-LIB on failure."""
    ns = _pyeval_namespace(env)
    try:
        result = eval(src, ns)  # noqa: S307
        if isinstance(result, _E):
            return result.t
        if isinstance(result, cvc5.Term):
            return result
        # Some other Python object — fall through to SMT-LIB
    except Exception:
        pass
    return env.parse_expr(src)


CEGAR_GENERATE_SYSTEM = """\
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

The ranking expression must have sort **Int** and satisfy \
`ranking >= 0` whenever the invariant holds. If a state component is \
Real, wrap it in `(to_int ...)` to convert to Int; Bool components \
can be mapped via `(ite s_k 1 0)`.

Use standard SMT-LIB syntax: (and ...), (or ...), (not ...), (= ...), \
(< ...), (<= ...), (+ ...), (- ...), (* ...), (ite cond a b). \
Tuple fields are accessed with `((_ tuple.select k) t)`.
"""


@dataclass
class PromptResult:
    inv_src: str
    ranking_src: str
    inv_term: cvc5.Term
    ranking_term: cvc5.Term


class CegarPromptEnv:
    """Reusable parser context with module symbols declared once:
      * `s0..sN-1`     — ctrl-next state components
      * `e0..eM-1`     — extl-next inputs
      * `el0..elM-1`   — extl-latched inputs
    """

    def __init__(self, msmt: ModuleSMT):
        self.msmt = msmt
        self.tm = msmt.tm
        self.solver = cvc5.Solver(self.tm)
        self.solver.setLogic("ALL")
        self.sm = cvc5.SymbolManager(self.solver)
        self.parser = cvc5.InputParser(self.solver, self.sm)
        self.parser.setIncrementalStringInput(
            cvc5.InputLanguage.SMT_LIB_2_6, "cegar"
        )

        self.state_sorts = [wire_sort(self.tm, w) for w in msmt.ctrl_next]
        self.extl_next_sorts = [wire_sort(self.tm, w) for w in msmt.extl_next]
        self.extl_latched_sorts = [
            wire_sort(self.tm, w) for w in msmt.extl_latched
        ]

        decls = []
        for i, s in enumerate(self.state_sorts):
            decls.append(f"(declare-const s{i} {s})")
        for i, s in enumerate(self.extl_next_sorts):
            decls.append(f"(declare-const e{i} {s})")
        for i, s in enumerate(self.extl_latched_sorts):
            decls.append(f"(declare-const el{i} {s})")
        self._run_commands("\n".join(decls))

        names = (
            [f"s{i}" for i in range(len(self.state_sorts))]
            + [f"e{i}" for i in range(len(self.extl_next_sorts))]
            + [f"el{i}" for i in range(len(self.extl_latched_sorts))]
        )
        if names:
            self.parser.appendIncrementalStringInput(" ".join(names))
            parsed = [self.parser.nextTerm() for _ in names]
        else:
            parsed = []

        k = len(self.state_sorts)
        m = len(self.extl_next_sorts)
        self.state_vars: list[cvc5.Term] = parsed[:k]
        self.extl_next_vars: list[cvc5.Term] = parsed[k : k + m]
        self.extl_latched_vars: list[cvc5.Term] = parsed[k + m :]

    # --- internals ------------------------------------------------------

    def _run_commands(self, text: str) -> None:
        self.parser.appendIncrementalStringInput(text)
        while True:
            cmd = self.parser.nextCommand()
            if cmd.isNull():
                break
            cmd.invoke(self.solver, self.sm)

    def parse_expr(self, smt_src: str) -> cvc5.Term:
        """Parse a single SMT-LIB expression referring to s0..sN-1."""
        # Re-set the input source each call — the incremental stream ends
        # after a single nextTerm/nextCommand pair.
        self.parser.setIncrementalStringInput(
            cvc5.InputLanguage.SMT_LIB_2_6, "cegar"
        )
        self.parser.appendIncrementalStringInput(smt_src)
        return self.parser.nextTerm()

    # --- description for the LLM ---------------------------------------

    def state_description(self) -> str:
        lines = ["State components (SMT sorts):"]
        for i, (w, sort) in enumerate(zip(self.msmt.ctrl_next, self.state_sorts)):
            lines.append(f"  s{i} : {sort}   # {w.dtype}")
        if self.extl_next_sorts or self.extl_latched_sorts:
            lines.append("External inputs (SMT sorts):")
            for i, sort in enumerate(self.extl_next_sorts):
                lines.append(f"  e{i}  : {sort}   # next")
            for i, sort in enumerate(self.extl_latched_sorts):
                lines.append(f"  el{i} : {sort}   # latched")
        return "\n".join(lines)

    def transition_description(self) -> str:
        """Symbolic transition formulas for the module.

        Renders `init(en)` and `update(sL, el, en)` as SMT expressions so
        the LLM can see the exact update semantics without reverse-
        engineering from source. Here `sL` = latched state (same variable
        space as `s0..sN-1` we ask the invariant about).
        """
        lines = ["Initial state (s_i after init, given extl-next e):"]
        init_vals = self.msmt.init_state(self.extl_next_vars)
        for i, t in enumerate(init_vals):
            lines.append(f"  s{i}_init = {t}")
        lines.append(
            "Update: latched state is s_i, latched inputs el_i, next inputs e_i; "
            "new state s_i' is:"
        )
        next_vals = self.msmt.update_state(
            self.state_vars, self.extl_latched_vars, self.extl_next_vars
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
    env: CegarPromptEnv,
    chat,
    source: str,
    prp: str | None,
    preconds: str,
    feedback: str | None = None,
    *,
    fixed_inv_src: str | None = None,
    fixed_ranking_src: str | None = None,
) -> PromptResult:
    """Call `chat` with the CEGAR prompt, parse reply into SMT terms.

    If `fixed_inv_src` / `fixed_ranking_src` is supplied, that field is
    treated as a user-provided input: not requested from the LLM, and
    shown in the prompt so the LLM sees it while searching for the other.
    If both are fixed, no LLM call is made.
    """
    if fixed_inv_src is not None and fixed_ranking_src is not None:
        return PromptResult(
            fixed_inv_src,
            fixed_ranking_src,
            parse_predicate(env, fixed_inv_src),
            parse_predicate(env, fixed_ranking_src),
        )
    user_msg = (
        f"Source code:\n```python\n{source}\n```\n\n"
        f"Property (prp): {prp}\n\n"
        f"{preconds}\n\n"
        f"{env.state_description()}\n\n"
        f"{env.transition_description()}\n"
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
        user_msg += (
            "\nPrevious attempt was rejected. Feedback:\n"
            f"{feedback}\n"
            "Please try again with corrected invariant and ranking.\n"
        )
    text = chat(CEGAR_GENERATE_SYSTEM, user_msg)
    inv_src, ranking_src = parse_llm_reply(
        text,
        require_inv=fixed_inv_src is None,
        require_ranking=fixed_ranking_src is None,
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
    try:
        ranking_term = parse_predicate(env, ranking_src)
    except RuntimeError as e:
        raise PromptParseError(
            f"Could not parse RANKING `{ranking_src}`: {e}. "
            f"Use only s0..s{len(env.state_vars) - 1}."
        ) from e
    if str(inv_term.getSort()) != "Bool":
        raise PromptParseError(
            f"INVARIANT must have sort Bool, got {inv_term.getSort()}."
        )
    if str(ranking_term.getSort()) != "Int":
        raise PromptParseError(
            f"RANKING must have sort Int, got {ranking_term.getSort()}."
        )
    return PromptResult(inv_src, ranking_src, inv_term, ranking_term)
