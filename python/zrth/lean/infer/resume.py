"""What a previous run left a route to resume from, read off `artifacts/`:
an invariant to take as given (`resumable_inv`, `taking_inv`), and the
shapes a search proved empty (`ruled_out`).

The policy of the two routes that take a given invariant -- `--infer
ai-cegis` and `--infer smt-linear` -- rather than a property of what every
route is handed: which statuses make an invariant fit to be held fixed,
and when an explicit `--invariant` wins. A route reads the store through
`ProjectHandle.resume`, which is the access its row declared, and this
module goes through it too.
"""

from __future__ import annotations

from dataclasses import replace

from ..artifacts import Artifact
from .io import InferInput


# Statuses that make a previously-found invariant a *fixed* invariant for
# a route that holds it as given: it has to be inductive. `too_weak` is
# inductive and merely fails to imply the property, which `rule_buchi` never
# asked of it -- so it is usable for a Buchi certificate and is exactly the
# wrong thing for a safety one.
FIXABLE = ("proved", "too_weak")


def resumable_inv(inp: InferInput, *,
                  strengthen: bool = False) -> "Artifact | None":
    """An invariant from a previous run to start from, or `None`.

    An explicit `--invariant` wins: inheriting a predicate the user did
    not ask for, over one they did, is the kind of cross-run surprise
    that makes two identical command lines mean different things.

    By default the invariant is one to take as *given*:
    `infer.ai_cegis.engine` infers only what is left around a fixed
    `given.inv`, so it has to be inductive and, on a safety claim, imply
    the property. `too_weak` is inductive and merely fails to imply it,
    which `rule_buchi` never asked of an invariant and exactly what
    `rule_globally` needs -- so it is skipped for a safety claim, the one
    thing the store cannot filter on. With `strengthen=True` the
    invariant is one to *add to*: `--infer smt-linear --safety` searches
    for the conjuncts that are missing, so an invariant that is inductive
    and does not yet imply the property is the best thing it can be
    handed.
    """
    if inp.given.inv:
        return None
    for a in inp.project.resume("inv", languages=("smt",), statuses=FIXABLE):
        if (a.status == "too_weak" and inp.claim.is_safety
                and not strengthen):
            continue
        return a
    return None


def taking_inv(inp: InferInput, resumed: "Artifact | None",
               what: str) -> InferInput:
    """This input with a resumed invariant as its given one, saying so."""
    if resumed is None:
        return inp
    inp.log(f".. resuming from {resumed.name} ({resumed.status}): {what}")
    inv = inp.project.artifacts.read(resumed)
    return replace(inp, given=replace(inp.given, inv=inv))


def ruled_out(inp: InferInput) -> tuple[str, ...]:
    """What earlier runs *proved* is not there, as prose for the prompt.

    Only `no_solution`: a note whose status is `unknown` records a search
    that ran out of budget, and telling a model that a shape is empty when
    all that happened is a timeout would put a falsehood in the prompt.
    The store has already dropped anything found against another module,
    property or proof rule, which is the other half of the same care.
    """
    notes = inp.project.resume("note", languages=("md",),
                               statuses=("no_solution",))
    out = []
    for a in notes:
        inp.log(f".. resuming from {a.name}: a space an earlier run "
                f"ruled out")
        out.append(inp.project.artifacts.read(a))
    return tuple(out)
