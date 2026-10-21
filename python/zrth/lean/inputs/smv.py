"""SMV input: a NuSMV model, read as NuSMV reads it or refused."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.options import Opt
from .api import InputAdapter, InputError, Problem, SourceText, StatedClaim
from .names import Names, usable


def _load(source: SourceText, opts: Mapping[str, Any]) -> Problem:
    from lark.exceptions import LarkError

    from zrth.smv import SMVRefused, read_smv

    try:
        module, variables, specs = read_smv(source.text)
    except SMVRefused as e:
        raise InputError(f"{source.path}: {e}") from e
    except LarkError as e:
        raise InputError(f"{source.path} is not SMV verith reads: {e}") from e
    variables_of = {*module.ctrl, *module.extl}
    # A variable the module never reads (an unused IVAR) is not one of its
    # variables, and a name a claim could not use is left positional.
    named = {n: v for n, v in variables.items()
             if v in variables_of and usable(n)}
    claims = tuple(StatedClaim(s.kind, s.where, text)
                   for s, text in zip(specs, _internal(module, specs)))
    return Problem(module, source, Names.of(module, named), claims)


def _internal(module, specs) -> list[str]:
    """Each spec's formula as SMT-LIB in internal names: its terms read the
    variables' next values, which a claim calls `s{i}`. One term manager
    serves every spec, and lives until exit."""
    if not specs:
        return []
    import cvc5

    from zrth import X

    from ..smt.encode import translate_terms, wire_sort
    from ..smt.module import keep_alive

    tm = cvc5.TermManager()
    keep_alive(tm)
    bindings = {X(v).id: tm.mkConst(wire_sort(tm, X(v)), f"s{i}")
                for i, v in enumerate(module.ctrl)}
    texts = []
    for spec in specs:
        term = translate_terms(tm, list(spec.terms), bindings)[spec.wire.id]
        if term.getSort().isBitVector():
            term = tm.mkTerm(cvc5.Kind.EQUAL, term, tm.mkBitVector(1, 1))
        keep_alive(term)
        texts.append(str(term))
    return texts


ADAPTER = InputAdapter(
    name="smv",
    summary=(
        "a NuSMV model (one MODULE main), read as NuSMV reads it -- a model "
        "that would mean something else is refused -- with claims in its "
        "variable names; its INVARSPEC / LTLSPEC G F is the claim when "
        "--safety/--buchi is not given"
    ),
    suffixes=(".smv",),
    load=_load,
    options=(
        Opt(("--spec",), {
            "type": int,
            "metavar": "N",
            "help": (
                "Which of the model's specifications is the claim, counted "
                "from 1 in the order they appear; needed when it states "
                "more than one."
            ),
        }),
    ),
    states_claims=True,
)
