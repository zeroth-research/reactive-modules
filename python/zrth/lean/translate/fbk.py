"""The Lean shape the FBK toolchain reads: `lean-ltl-certifying`'s `lean2vmt`.

`proveit.py` over in ``lean-ltl-certifying`` turns a Lean model into a VMT
transition system (`lake exe lean2vmt`), model-checks it with ic3ia and
renders the resulting inductive invariant back as a Lean certificate.  The
model it reads is a `Cslib.Automata.NA`, written in a syntax `lean2vmt`
pattern-matches on -- and every mismatch is silent, a VMT file that parses
and does not describe the module:

======================  ==================================  ==================
`lean2vmt` requires     why                                 otherwise
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
`abbrev M : … NA …`     the model and refers to both        a namespace
self-contained          it is elaborated inside the         imports `Core.Basic`,
                        `lean-ltl-certifying` package       `System.Scalar`, …
======================  ==================================  ==================

The right-hand column is what this file's predecessor emitted: a Bool-valued
relational encoding, `System/FBK.lean`, generated into every project and read
by nothing.  It missed all six rows, which is why it could never be this, and
why it is gone and its name is here.

This is emitted only for the ``--fbk-proveit`` route.  The generated project
does build it -- as the `<Proj>NA` lean_lib, because the installed
certificate imports it -- but it is not one of the project's own encodings.

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
`--infer ai-cegis` already run on — printed by `smt_to_lean_bool`, whose
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

from typing import NamedTuple

from zrth.lean.common import (
    LeanContext,
    _flat_element_type,
    flat_layout,
)
from zrth.lean.native import _reachable_terms
from zrth.lean.translate._skeleton import RelBlock, RelSyntax, emit_rel_block


class NAUnsupported(Exception):
    """The module cannot be expressed in the shape `lean2vmt` reads."""


class SlotBodies(NamedTuple):
    """The Lean text of every state slot, in slot order: `(update, init)`.

    What `_slot_bodies` builds and what `check_na_supported` hands back --
    the check *is* the encoding (see the comment on it), so the bodies it
    printed to decide the question are the bodies the emission wants, and
    returning them is what keeps one cvc5 encoding per module instead of
    one per caller.
    """

    update: list[str]
    init: list[str]


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
# `--infer ai-cegis` already run on) and printed by `smt_to_lean_bool`, which
# raises on anything outside the fragment `lean2vmt` reads rather than
# printing it. Two consequences: `Linear` and the rest come through as the
# scalar arithmetic they expand to, and an op neither component handles
# cannot reach the file at all. `check_na_supported` therefore *does* the
# translation and reports whatever it raises -- and hands the result back,
# so that translation is the one the emission uses rather than one thrown
# away and repeated.
def check_na_supported(ctx: LeanContext, simplify: bool = True) -> SlotBodies:
    """`ctx`'s slot bodies; :class:`NAUnsupported` if it has no NA encoding.

    Both emissions on this route want exactly those bodies --
    `atom_to_lean_na` for the model, `atom_to_lean_fbk_bridge` for the proof
    that the model is the module -- so they take them rather than ask for
    them again. A caller that only wants the question answered can discard
    the answer.

    Every restriction here is one `lean2vmt`/`vmt2lean.py` cannot express;
    letting one through would produce a VMT model that parses but does not
    describe the module.

    `simplify` has to match what the emission will use: the rewriter can
    fold an unsupported subterm away (a `Linear` whose weights are all
    zero), so a check run with it on would pass a module the emission then
    could not print.
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
        return _slot_bodies(ctx, simplify)
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
# packs a matrix into a cvc5 tuple -- `common.flat_layout` owns that order
# for every encoding, and `tests/test_lean_flat_layout.py` is what keeps the
# slot arithmetic here and the element selection there in step.
#
# Per wire was the older layout, and it is why this encoding used to reject
# any wire holding more than one element: `R_i` compares `var_i statenext`
# against `effect_i`, and a tuple has no comparison `lean2vmt` translates.
# Per element there is nothing to compare but scalars.
# ══════════════════════════════════════════════════════════════════════


def _slot_accessors(ctrl_next, binder: str = "state") -> dict[str, list[str]]:
    """`s{i}` → the `(var_k <binder>)` reads of wire `i`'s elements.

    This is the map `smt_to_lean_bool` resolves both the state constants and
    their tuple selectors through: the encoder names a wire `s{i}` and
    selects element `k` of it, and this says which VMT variable that is.
    """
    layout = flat_layout(ctrl_next)
    out: dict[str, list[str]] = {}
    for i, w in enumerate(ctrl_next):
        offset, size = layout.span(w)
        out[f"s{i}"] = [f"(var_{offset + j} {binder})" for j in range(size)]
    return out


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


def _simplify(solver, term, enabled: bool = True):
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

    `enabled=False` (`--fbk-simplify none`) turns the rewriter off
    everywhere, leaving each slot the shape the module's own terms give it:
    the same transition, spelled the way the module spells it rather than
    the way cvc5 prefers to. Useful for reading the model against the
    source; expensive on anything with a constant matrix in it.
    """
    if not enabled or term.getSort().isBoolean():
        return term
    return solver.simplify(term)


def _slot_bodies(ctx: LeanContext, simplify: bool = True) -> SlotBodies:
    """`(update, init)` Lean text for every state slot, in slot order.

    One cvc5 `TermManager` and `Solver` per call, both retained for the
    process's lifetime (`smt_module.keep_alive`), so the callers go through
    :func:`check_na_supported` and pass the result along rather than calling
    this again. Retained per *model*, not per caller: one entry where it
    used to be five (measured, one-wire counter).

    The transition comes from `smt_encode` -- the encoder `--pre-check` and
    `--infer ai-cegis` already run on, so the model `lean2vmt` reads and the
    obligations cvc5 answers are the same encoding, not two readings of one
    module -- and is printed by `smt_to_lean_bool`, whose fragment is chosen
    to be what `lean2vmt` translates.
    """
    import cvc5

    from ..smt_encode import wire_shape
    from ..smt_module import ModuleSMT, keep_alive
    from ..smt_to_lean import smt_to_lean_bool

    tm = cvc5.TermManager()
    msmt = ModuleSMT(tm=tm, module=ctx.module)
    solver = cvc5.Solver(tm)
    solver.setLogic("ALL")
    keep_alive(tm, msmt, solver)

    acc = _slot_accessors(ctx.ctrl_next)
    layout = flat_layout(ctx.ctrl_next)
    state = msmt.fresh_ctrl("s")
    nxt = msmt.update_state(state, [], [])
    ini = msmt.init_state([])

    update_text: list[str] = []
    init_text: list[str] = []
    for i, r, c in layout.slots:
        shape = wire_shape(ctx.ctrl_next[i])
        # Simplified because the encoder builds a matrix and then takes one
        # element of it: `m_vec32`'s affine layer is 97 KB of term before
        # the rewriter folds the constant weights and 1.9 KB after.
        upd = _simplify(solver, _scalar_element(tm, nxt[i], shape, r, c), simplify)
        init = _simplify(solver, _scalar_element(tm, ini[i], shape, r, c), simplify)
        update_text.append(smt_to_lean_bool(upd, acc))
        init_text.append(smt_to_lean_bool(init, acc))
    return SlotBodies(update_text, init_text)


def atom_to_lean_na(
    ctx: LeanContext,
    property_lean: str,
    *,
    module_name: str = "",
    simplify: bool = True,
    bodies: SlotBodies | None = None,
) -> str:
    """Emit the whole NA model file for `ctx`.

    `property_lean` is a Bool-valued Lean expression over the same
    `(var_i state)` bindings this encoding uses — see
    ``smt_to_lean.smt_to_lean_bool``.

    `bodies` is what :func:`check_na_supported` returned for this `ctx` and
    `simplify`; passing it is what keeps a route that has already checked
    the module from encoding it into cvc5 a second time.  Left out, the
    check runs here -- so a caller that has not checked still cannot emit an
    unsupported module.
    """
    if bodies is None:
        bodies = check_na_supported(ctx, simplify)
    update_text, init_text = bodies

    layout = flat_layout(ctx.ctrl_next)
    n = layout.total
    slot_ty = layout.element_types()

    subject = f"reactive module `{module_name}`" if module_name else "a reactive module"
    how = "" if simplify else " --fbk-simplify none"
    lines: list[str] = [
        f"/- NA encoding of {subject}, for lean-ltl-certifying's lean2vmt.\n"
        f"   Generated by `verith --fbk-proveit{how}`. -/"
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

    # The body def / relation / conjunction shape below is the one `Rel` and
    # `ScalarRel` emit too, so it comes from the shared skeleton -- spelled
    # here in `abbrev`, `Bool`, `==` and `&&`, which is what `lean2vmt`
    # reads, and with no `*_eq` theorem block: a cvc5-printed body is the
    # module's transition spelled differently, not syntactically equal to the
    # Lean one, so this route relates the two semantically (`fbk_bridge.py`).
    syn = RelSyntax(
        body_decl="abbrev",
        rel_decl="abbrev",
        prop="Bool",
        eq="==",
        conj="&&",
        slot_ty=lambda i: slot_ty[i],
        project=lambda e, i: f"var_{i} {e}",
    )

    # --- transition relation ---
    # A slot whose next value does not read the state is a closed term, and
    # the binder is left off rather than written and unused.
    effect_uses_state = ["state" in body for body in update_text]
    lines += emit_rel_block(
        syn,
        RelBlock(
            body_name="effect",
            rel_name="R",
            conj_name="TRANS",
            state_binders="(state statenext : StateType)",
            state_args="state statenext",
            target="statenext",
            binders=[" (state : StateType)" if u else "" for u in effect_uses_state],
            args=[" state" if u else "" for u in effect_uses_state],
        ),
        [f"  {body}" for body in update_text],
    )

    # --- initial condition ---
    # No extl wires (checked), so `init_i` is a closed term.
    lines += emit_rel_block(
        syn,
        RelBlock(
            body_name="init",
            rel_name="Init",
            conj_name="INIT",
            state_binders="(state : StateType)",
            state_args="state",
            target="state",
            binders=[""] * n,
            args=[""] * n,
        ),
        [f"  {body}" for body in init_text],
    )

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
