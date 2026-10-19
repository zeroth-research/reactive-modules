"""One module in cvc5 for a whole run.

`ModuleEncoding` owns the run's `TermManager`: the module's transition, the
predicates read against it and every solver built on it. `lazy(module)` is
how a run holds one, so a run that asks cvc5 nothing never imports it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

from ..errors import PredicateRefused
from .limits import bounded_solver


class ModuleEncoding:
    """`module` in cvc5, encoded once.

    The transition is translated on first use over the parser's `s*`, `el*`
    and `e*`, and every other state or input is substituted into it; a
    module with no encoding raises the same error on every use. Retained
    until exit (`smt_module.keep_alive`), and nothing it owns refers back
    to it: a cycle through a `TermManager`'s owner is finalised in
    arbitrary order.
    """

    def __init__(self, module):
        import cvc5

        from ..smt_module import ModuleSMT, keep_alive
        from ..smt_prompt import CegisPromptEnv

        self.module = module
        self.tm = cvc5.TermManager()
        self.msmt = ModuleSMT(tm=self.tm, module=module)
        self.env = CegisPromptEnv(self.msmt)
        self._parsed: dict = {}
        self._blocks: dict = {}
        self._queries: dict = {}
        self._solvers: list = []
        self._rewriter = None
        keep_alive(self)

    @property
    def state(self) -> list:
        return self.env.state_vars

    @property
    def extl_latched(self) -> list:
        return self.env.extl_latched_vars

    @property
    def extl_next(self) -> list:
        return self.env.extl_next_vars

    # --- predicates -------------------------------------------------------

    def parse(self, src: str | None, role: str, want: str = "Bool",
              kind: str = "buchi"):
        """`src` as a term over the state and inputs, or `PredicateRefused`
        naming its `role` ("property" of `kind`, "pre", "invariant",
        "ranking").

        `want` is the sort the field is declared as and the Lean printer
        assumes: a property, an invariant and a precondition become the
        body of a `Prop`, a ranking function the body of a `Nat`. An Int
        where a Prop is expected elaborates to nothing, and a Bool ranking
        prints as Lean that does not parse, so the wrong sort is refused
        here, where it is known.
        """
        if src is None:
            return None
        hit = self._parsed.get(src)
        if hit is None:
            from ..smt_prompt import parse_predicate

            try:
                hit = parse_predicate(self.env, src)
            except Exception as e:
                hit = e
            self._parsed[src] = hit
        if isinstance(hit, Exception):
            raise PredicateRefused(
                role, f"cannot read `{src}`: {hit}", kind=kind) from hit
        got = hit.getSort()
        if not (got.isBoolean() if want == "Bool" else got.isInteger()):
            raise PredicateRefused(
                role, f"must have sort {want}, got {got}", kind=kind)
        return hit

    def queries(self, claim, cert=None):
        """`smt_query.ModuleQueries` for `claim` and `cert`, built once per
        pair. Raises as its constructor does."""
        from ..certificate import Certificate
        from ..smt_query import ModuleQueries

        key = (claim, cert or Certificate())
        q = self._queries.get(key)
        if q is None:
            q = self._queries[key] = ModuleQueries(self, *key)
        return q

    # --- the transition ---------------------------------------------------

    def _block(self, name: str) -> list:
        hit = self._blocks.get(name)
        if hit is None:
            try:
                if name == "init":
                    hit = self.msmt.init_state(self.extl_next)
                else:
                    hit = self.msmt.update_state(
                        self.state, self.extl_latched, self.extl_next)
            except Exception as e:
                hit = e
            self._blocks[name] = hit
        if isinstance(hit, Exception):
            raise hit
        return hit

    def init_state(self, en: list) -> list:
        """The state after tick 0, given the inputs `en` awaited then."""
        old = list(self.extl_next)
        terms = self._block("init")
        return [t.substitute(old, list(en)) for t in terms] if old else list(terms)

    def next_state(self, s: list, el: list, en: list) -> list:
        """The state after one round from `s`, given the inputs latched
        (`el`) and awaited (`en`)."""
        old = list(self.state) + list(self.extl_latched) + list(self.extl_next)
        new = list(s) + list(el) + list(en)
        terms = self._block("next")
        return [t.substitute(old, new) for t in terms] if old else list(terms)

    # --- terms ------------------------------------------------------------

    def fresh(self, prefix: str = "") -> tuple[list, list, list]:
        """New constants for a state and the inputs latched and awaited:
        `{prefix}s0..`, `{prefix}el0..`, `{prefix}en0..`."""
        return (self.msmt.fresh_ctrl(f"{prefix}s"),
                self.msmt.fresh_extl_l(f"{prefix}el"),
                self.msmt.fresh_extl_n(f"{prefix}en"))

    def bound(self, prefix: str = "") -> tuple[list, list, list]:
        """Bound variables for a state and the inputs latched and awaited,
        for a quantifier to close: `fresh`'s sibling, named the same way."""
        return (self._vars(self.state, f"{prefix}s"),
                *self.bound_inputs(prefix))

    def bound_inputs(self, prefix: str = "") -> tuple[list, list]:
        """`bound`'s inputs alone, for a relation that closes them over a
        state it is given."""
        return (self._vars(self.extl_latched, f"{prefix}el"),
                self._vars(self.extl_next, f"{prefix}en"))

    def _vars(self, terms: list, prefix: str) -> list:
        return [self.tm.mkVar(t.getSort(), f"{prefix}{i}")
                for i, t in enumerate(terms)]

    def at(self, term, vs: list):
        """`term` with the state replaced by `vs`."""
        return None if term is None else term.substitute(self.state, vs)

    def with_inputs(self, term, el: list, en: list):
        """`term` with the parser's inputs replaced by `el` / `en`."""
        if term is None:
            return None
        old = list(self.extl_next) + list(self.extl_latched)
        new = list(en) + list(el)
        return term.substitute(old, new) if old else term

    def simplify(self, term):
        """`term` through cvc5's rewriter, on a solver this encoding keeps
        for that alone."""
        if self._rewriter is None:
            import cvc5

            self._rewriter = cvc5.Solver(self.tm)
            self._rewriter.setLogic("ALL")
        return self._rewriter.simplify(term)

    def solver(self, ms: int, *, logic: str | None = None, **options):
        """`smt.limits.bounded_solver` on this encoding's terms."""
        s = bounded_solver(self.tm, ms, logic=logic, **options)
        self._solvers.append(s)
        return s


def lazy(module) -> Callable[[], ModuleEncoding]:
    """`module`'s encoding, built on the first call and the same after."""
    return functools.cache(lambda: ModuleEncoding(module))


def encoding_of(source) -> ModuleEncoding:
    """`source` as an encoding: itself, what a `lazy` factory builds, or a
    new one for a bare module."""
    if isinstance(source, ModuleEncoding):
        return source
    if callable(source):
        return source()
    return ModuleEncoding(source)
