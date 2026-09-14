"""NA translation — the Lean shape `lean-ltl-certifying`'s `lean2vmt` reads.

`proveit.py` over in ``lean-ltl-certifying`` turns a Lean model into a VMT
transition system (`lake exe lean2vmt`), model-checks it with ic3ia and
renders the resulting inductive invariant back as a Lean certificate.  Its
input format is not the FBK encoding: `lean2vmt` pattern-matches on very
specific syntax, and the FBK encoding misses on every point.

======================  ==================================  ==================
`lean2vmt` requires     why                                 FBK emits
======================  ==================================  ==================
binders `state`,        `emitDefs` looks the binders up by  `state`, `newstate`,
`statenext`             name to decide current vs. next     `s`
`var_i state`           `exprToSMT`'s ``.fvar`` case        `(state i)`
                        returns the bare binder name and
                        *drops the index*, so every slot
                        would collapse onto one variable
`var_i statenext`       this is how `collectLatchesIndices` `(newstate i)`
                        finds the latches at all
`INIT`/`TRANS`/         they become `:init`, `:trans` and   `InitCond`,
`PROPERTY`              `:invar-property`                   `TransRel`, nothing
top-level `StateType`,  the certificate template imports    all inside
`abbrev M : … NA …`     the model and refers to both        `namespace FBK`
self-contained          it is elaborated inside the         imports `Core.Basic`,
                        `lean-ltl-certifying` package       `System.Scalar`, …
======================  ==================================  ==================

So this is a separate encoding rather than a tweak of `fbk.py`.  It is only
emitted for the ``--fbk-proveit`` route and is never part of the generated
lake project's build.

Everything `lean2vmt` and `vmt2lean.py` cannot represent is rejected up
front by :func:`check_na_supported`, so nothing silently degrades into a
wrong transition system.

Where the transition comes from
-------------------------------

One VMT variable per state *element*.  A wire holding a 3-vector is
`var_k, var_k+1, var_k+2`; `R_k` compares one scalar against one scalar,
which is all `lean2vmt` can translate.  This encoding used to reject any
wire wider than 1x1 for exactly that reason — `R_i` compared `var_i
statenext` against a tuple — and 11 of the limit matrix's 77 cases were
refused by it.

The body of each slot is *not* written by this module.  It is
`smt_encode`'s term for that element — the encoder `--pre-check` and
`--infer ai-cegar` already run on — printed by `smt_to_lean_bool`, whose
fragment is chosen to be what `lean2vmt` reads.  Three things follow:

* an op with no *scalar Lean* form is no longer a problem, because no
  scalar Lean is emitted: `Linear` arrives as the affine sum it expands to,
  `Argmax` as the nested `ite` chain, and both are inside the fragment;
* an op neither component handles cannot reach the file at all — the
  encoder or the printer raises, and `check_na_supported` reports it;
* the model `lean2vmt` reads and the obligations cvc5 answers about the
  same module are now the *same encoding*, rather than two readings that
  could drift apart.
"""

from zrth.lean.common import (
    LeanContext,
    _flat_element_type,
    _flat_indices,
    _flat_size,
)
from zrth.lean.native import _reachable_terms


class NAUnsupported(Exception):
    """The module cannot be expressed in the shape `lean2vmt` reads."""


# What the model imports -- and, because the driver builds exactly this
# list, what `lean2vmt` needs compiled before it can elaborate the file.
#
# The hand-written reference model
# `LTLCertifying/RMNATranslation/NACounterInt.lean` also pulls in `Mathlib`
# and `LTLCertifying.Safety.Lemmas`, but neither is used here: everything
# below `namespace Definition` is core Lean, and the LTL vocabulary belongs
# to the certificate, which imports it on its own.  Keeping the list at one
# entry is what makes the pre-build cheap.
NA_IMPORTS = ["Cslib.Computability.Automata.NA.Basic"]

# `vmt2lean.py`'s `tp()` asserts on anything that is neither Int nor Bool,
# and `lean2vmt` declares state constants by the abbrev's ascribed type.
NA_ELEMENT_TYPES = frozenset({"Int", "Bool"})

# There is no op allowlist any more, and the reason is worth stating.
#
# This encoding used to emit each wire's transition through the *scalar Lean*
# printer, which meant an op with no scalar form -- `Linear` (`matVecAffine`),
# `Argmax`, `Transpose` -- reached `lean2vmt` as an unapplied leaf: a VMT file
# that parses and does not describe the module. The allowlist was what stood
# between that and a silently wrong model.
#
# The transition is now built by `smt_encode` (the encoder `--pre-check` and
# `--infer ai-cegar` already run on) and printed by `smt_to_lean_bool`, which
# raises on anything outside the fragment `lean2vmt` reads rather than
# printing it. Two consequences: `Linear` and the rest come through as the
# scalar arithmetic they expand to, and an op neither component handles
# cannot reach the file at all. `check_na_supported` therefore *does* the
# translation and reports whatever it raises.
def check_na_supported(ctx: LeanContext) -> None:
    """Raise :class:`NAUnsupported` unless `ctx` fits the NA encoding.

    Every restriction here is one `lean2vmt`/`vmt2lean.py` cannot express;
    letting one through would produce a VMT model that parses but does not
    describe the module.
    """
    if ctx.extl_latched or ctx.extl_next:
        raise NAUnsupported(
            f"the module has {len(ctx.extl_next)} external input wire(s); "
            "lean2vmt models only `state`/`statenext`, so inputs have no "
            "VMT counterpart"
        )
    if not ctx.ctrl_next:
        raise NAUnsupported("the module has no controlled state to encode")

    elem_types = {_flat_element_type(w) for w in ctx.ctrl_next}
    bad = elem_types - NA_ELEMENT_TYPES
    if bad:
        raise NAUnsupported(
            f"state element type(s) {', '.join(sorted(bad))} unsupported; "
            "vmt2lean.py only maps Int and Bool back to Lean"
        )

    if not list(ctx.atom.init):
        raise NAUnsupported("the module has no init terms, so there is no INIT")
    if not list(ctx.atom.update):
        raise NAUnsupported("the module has no update terms, so there is no TRANS")

    # A ctrl wire nothing writes is a free (nondeterministic) variable, and
    # this encoding has no way to say that: `Init_k`/`R_k` pin every slot to
    # a value. Caught here rather than left to the encoder, whose own error
    # for an unbound wire says nothing about why it matters.
    for block, terms in (("init", ctx.atom.init), ("update", ctx.atom.update)):
        loose = [
            f"#{i}"
            for i, w in enumerate(ctx.ctrl_next)
            if not _reachable_terms(terms, [w])
        ]
        if loose:
            raise NAUnsupported(
                f"no {block} term writes ctrl wire(s) {', '.join(loose)}; "
                "the NA encoding has no way to leave a state variable "
                "unconstrained"
            )

    # The one check that cannot be made by inspection: encode the module and
    # print it. Whatever `smt_encode` or `smt_to_lean_bool` refuses is what
    # this encoding cannot express, and their messages name the op or the
    # SMT kind. Cheap -- the same work the emission does, on modules small
    # enough that `lake` dominates either way.
    try:
        _slot_bodies(ctx)
    except NAUnsupported:
        raise
    except Exception as e:
        raise NAUnsupported(
            f"the transition cannot be expressed in the fragment lean2vmt "
            f"reads: {e}"
        ) from e


# ══════════════════════════════════════════════════════════════════════
# State layout
#
# One VMT variable per *element*, not per wire. A wire holding a 3-vector
# becomes `var_k, var_k+1, var_k+2`, in the row-major order `mat_select`
# packs a matrix into a cvc5 tuple, so slot arithmetic here and element
# selection there stay in step by construction.
#
# Per wire was the older layout, and it is why this encoding used to reject
# any wire holding more than one element: `R_i` compares `var_i statenext`
# against `effect_i`, and a tuple has no comparison `lean2vmt` translates.
# Per element there is nothing to compare but scalars.
# ══════════════════════════════════════════════════════════════════════


def _slot_layout(ctrl_next) -> list[tuple[int, int, int]]:
    """`(wire index, row, col)` for each state slot, in slot order."""
    return [
        (i, r, c) for i, w in enumerate(ctrl_next) for (r, c) in _flat_indices(w)
    ]


def _slot_accessors(ctrl_next, binder: str = "state") -> dict[str, list[str]]:
    """`s{i}` → the `(var_k <binder>)` reads of wire `i`'s elements.

    This is the map `smt_to_lean_bool` resolves both the state constants and
    their tuple selectors through, so it is the single place that decides
    which slot an element lives in.
    """
    out: dict[str, list[str]] = {}
    k = 0
    for i, w in enumerate(ctrl_next):
        n = _flat_size(w)
        out[f"s{i}"] = [f"(var_{k + j} {binder})" for j in range(n)]
        k += n
    return out


# Kept alive for the process's lifetime: the cvc5 bindings segfault at
# shutdown when a TermManager is collected out of order with the solvers and
# terms minted from it, and a `verith` run builds one of these per model.
_LIVE: list = []


def _scalar_element(tm, term, shape, i: int, j: int):
    """Element `[i][j]` of a matrix-valued term, as a *scalar* term.

    `mat_select` alone leaves `select k (…)` sitting on whatever built the
    matrix, and the two things that build one here are a tuple constructor
    and an `ite` over tuples. Pushing the selection through both is what
    turns a wire's transition into one scalar expression per slot; the
    rewrite is the tuple axiom in each case, and cvc5's own simplifier does
    the rest.
    """
    from cvc5 import Kind

    from ..smt_encode import mat_select

    if shape.is_scalar:
        return term
    idx = i * shape.n + j
    kind = term.getKind()
    if kind == Kind.APPLY_CONSTRUCTOR:
        return term[idx + 1]          # child 0 is the constructor
    if kind == Kind.ITE:
        return tm.mkTerm(
            Kind.ITE,
            term[0],
            _scalar_element(tm, term[1], shape, i, j),
            _scalar_element(tm, term[2], shape, i, j),
        )
    return mat_select(tm, term, shape, i, j)


def _simplify(solver, term):
    """cvc5's rewriter, but never on a Bool-sorted slot.

    On the arithmetic it is what makes the model tractable: `m_vec32`'s
    affine layer is 97 KB of term before the constant weights are folded and
    1.9 KB after. On a *Bool* slot it rewrites `ite c b (!b)` into the
    equality `b = c`, which moves the state variable from a branch into a
    condition -- and a condition is where `vmt2lean`'s certificate cannot
    follow it: `generalizeNatVar` retypes the slot to the unreduced
    `TypeMap` match, and the `Decidable`/`BEq` instance the condition needs
    can no longer be synthesised. The measured symptom is `MixedBoolInt`
    failing its certificate with "Tactic `generalize` failed".
    """
    return term if term.getSort().isBoolean() else solver.simplify(term)


def _slot_bodies(ctx: LeanContext) -> tuple[list[str], list[str]]:
    """`(update, init)` Lean text for every state slot, in slot order.

    The transition comes from `smt_encode` -- the encoder `--pre-check` and
    `--infer ai-cegar` already run on, so the model `lean2vmt` reads and the
    obligations cvc5 answers are the same encoding, not two readings of one
    module -- and is printed by `smt_to_lean_bool`, whose fragment is chosen
    to be what `lean2vmt` translates.
    """
    import cvc5

    from ..smt_encode import wire_shape
    from ..smt_module import ModuleSMT
    from ..smt_to_lean import smt_to_lean_bool

    tm = cvc5.TermManager()
    msmt = ModuleSMT(tm=tm, module=ctx.module)
    solver = cvc5.Solver(tm)
    solver.setLogic("ALL")
    _LIVE.append((tm, msmt, solver))

    acc = _slot_accessors(ctx.ctrl_next)
    state = msmt.fresh_ctrl("s")
    nxt = msmt.update_state(state, [], [])
    ini = msmt.init_state([])

    update_text: list[str] = []
    init_text: list[str] = []
    for i, r, c in _slot_layout(ctx.ctrl_next):
        shape = wire_shape(ctx.ctrl_next[i])
        # Simplified because the encoder builds a matrix and then takes one
        # element of it: `m_vec32`'s affine layer is 97 KB of term before
        # the rewriter folds the constant weights and 1.9 KB after.
        upd = _simplify(solver, _scalar_element(tm, nxt[i], shape, r, c))
        init = _simplify(solver, _scalar_element(tm, ini[i], shape, r, c))
        update_text.append(smt_to_lean_bool(upd, acc))
        init_text.append(smt_to_lean_bool(init, acc))
    return update_text, init_text


def atom_to_lean_na(
    ctx: LeanContext,
    property_lean: str,
    *,
    module_name: str = "",
) -> str:
    """Emit the whole NA model file for `ctx`.

    `property_lean` is a Bool-valued Lean expression over the same
    `(var_i state)` bindings this encoding uses — see
    ``smt_to_lean.smt_to_lean_bool``.  Call :func:`check_na_supported` first.
    """
    check_na_supported(ctx)

    layout = _slot_layout(ctx.ctrl_next)
    n = len(layout)
    slot_ty = [_flat_element_type(ctx.ctrl_next[i]) for i, _, _ in layout]
    update_text, init_text = _slot_bodies(ctx)

    subject = f"reactive module `{module_name}`" if module_name else "a reactive module"
    lines: list[str] = [
        f"/- NA encoding of {subject}, for lean-ltl-certifying's lean2vmt.\n"
        "   Generated by `verith --fbk-proveit`. -/"
    ]
    lines += [f"import {m}" for m in NA_IMPORTS]
    lines.append("")

    # `TypeMap`/`StateType` sit *outside* `Definition`: the certificate
    # template opens `Definition` but names `StateType` unqualified.
    # A uniform state gets the wildcard arm alone; a mixed one gets an arm
    # per slot. The equation compiler builds a `TypeMap.match_1` for the
    # latter, which `emitDefs` skips of its own accord: it keeps only
    # declarations whose return type whnfs to `Prop`/`Int`/`Bool`, and a
    # matcher's is `motive n` with `motive` a free variable.
    lines.append("abbrev TypeMap : Nat → Type")
    if len(set(slot_ty)) == 1:
        lines.append(f"  | _ => {slot_ty[0]}")
    else:
        for i, ty in enumerate(slot_ty):
            lines.append(f"  | {i} => {ty}")
        lines.append(f"  | _ => {slot_ty[-1]}")
    lines.append("")
    lines.append("abbrev StateType := (n : Nat) → TypeMap n")
    lines.append("")
    lines.append("namespace Definition")
    lines.append("")

    # Binders are written out on every declaration rather than left to
    # `variable`: auto-binding drops a binder the body happens not to use,
    # and `M` below applies `INIT`/`TRANS` to a fixed number of arguments.
    lines.append("-- state variables")
    # The ascription is what `matchTypeName` reads to pick the VMT sort, so
    # it has to be the slot's own type, not a shared one.
    for i in range(n):
        lines.append(f"abbrev var_{i} (state : StateType) : {slot_ty[i]} := state {i}")
    lines.append("")

    # --- transition relation ---
    # A slot whose next value does not read the state is a closed term, and
    # the binder is left off rather than written and unused.
    effect_uses_state = ["state" in body for body in update_text]
    for i, body in enumerate(update_text):
        binder = " (state : StateType)" if effect_uses_state[i] else ""
        lines.append(f"abbrev effect_{i}{binder} : {slot_ty[i]} :=")
        lines.append(f"  {body}")
        lines.append("")

    for i in range(n):
        arg = " state" if effect_uses_state[i] else ""
        lines.append(f"abbrev R_{i} (state statenext : StateType) : Bool :=")
        lines.append(f"  var_{i} statenext == effect_{i}{arg}")
        lines.append("")

    lines.append("abbrev TRANS (state statenext : StateType) : Bool :=")
    lines.append("  " + " &&\n  ".join(f"R_{i} state statenext" for i in range(n)))
    lines.append("")

    # --- initial condition ---
    # No extl wires (checked), so `init_i` is a closed term.
    for i, body in enumerate(init_text):
        lines.append(f"abbrev init_{i} : {slot_ty[i]} :=")
        lines.append(f"  {body}")
        lines.append("")

    for i in range(n):
        lines.append(f"abbrev Init_{i} (state : StateType) : Bool :=")
        lines.append(f"  var_{i} state == init_{i}")
        lines.append("")

    lines.append("abbrev INIT (state : StateType) : Bool :=")
    lines.append("  " + " &&\n  ".join(f"Init_{i} state" for i in range(n)))
    lines.append("")

    # --- property ---
    lines.append("abbrev PROPERTY (state : StateType) : Bool :=")
    lines.append(f"  {property_lean}")
    lines.append("")
    lines.append("end Definition")
    lines.append("")

    # The certificate template refers to `M` by name, unqualified.
    lines.append("abbrev M : Cslib.Automata.NA StateType (Unit × Unit) := {")
    lines.append("  start := fun s => Definition.INIT s,")
    lines.append("  Tr := fun s _ s' => Definition.TRANS s s'")
    lines.append("}")

    return "\n".join(lines) + "\n"
