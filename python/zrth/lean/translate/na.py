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
front by :func:`check_na_supported` — see the module-level constants for the
exact set — so nothing silently degrades into a wrong transition system.
"""

from zrth.lean.common import (
    LeanContext,
    _flat_element_type,
    _flat_size,
    is_constant_name,
    itype_name,
)
from zrth.lean.native import _reachable_terms, _translate_terms_scalar
from zrth.lean.translate._shared import _effect_type


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

# Ops whose scalar emission (`native._SCALAR_OP`) lands inside what
# `exprToSMT` translates.  Deliberately *not* here:
#
#   ToUnsigned`Int.toNat x` → "toNat", and the result would be Nat-sorted
#             where the VMT declares Int
#   Argmax    calls the `argmax1d_scalar_*` axioms from `Core.Basic`
#   Linear    calls `matVecAffine`, likewise from `Core.Basic`
#
# each of which `exprToSMT` prints as an unapplied leaf — a silently wrong
# transition system rather than an error.  `MatMul` is here because on two
# 1x1 wires it emits plain `*`.
#
# `Ne`, `ReLU`, `Max` and `Min` used to be on that list too and are now
# translated, but only by a `lean2vmt` carrying the commit "lean2vmt:
# translate mod, max/min and a negated decidable instance".  Against an
# older checkout they go back to being printed as "instDecidableNot",
# "max" and "min", so this set and that commit have to travel together.
NA_OPS = frozenset({
    "Not", "And", "Or", "Ite",
    "Add", "Sub", "Mul", "Neg",
    "Lt", "Le", "Gt", "Ge", "Eq", "Ne",
    "Id", "TensorGet", "MatMul",
    "ReLU", "Max", "Min",
})

# Constant variants that inline as a bare Int/Bool literal.
NA_CONSTANTS = frozenset({"Int", "Bool"})


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

    wide = [
        f"#{i}" for i, w in enumerate(ctx.ctrl_next) if _flat_size(w) != 1
    ]
    if wide:
        raise NAUnsupported(
            f"ctrl wire(s) {', '.join(wide)} hold more than one element; "
            "the NA encoding needs one state slot per wire, because "
            "`R_i` compares `var_i statenext` against `effect_i` and "
            "lean2vmt cannot translate a tuple comparison"
        )

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

    # A ctrl wire nothing writes is a free (nondeterministic) variable.
    # `_translate_terms_scalar` renders it as `sorry`, and `Init_i`/`R_i`
    # would then pin the state to that -- an unprovable model rather than
    # an unconstrained one.
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

    unsupported: set[str] = set()
    for term in list(ctx.atom.init) + list(ctx.atom.update):
        name = itype_name(term.itype)
        if type(term.itype).__name__.startswith("BV_"):
            unsupported.add(f"BV.{name}")
        elif is_constant_name(name):
            if name not in NA_CONSTANTS:
                unsupported.add(name)
        elif name not in NA_OPS:
            unsupported.add(name)
    if unsupported:
        raise NAUnsupported(
            f"op(s) {', '.join(sorted(unsupported))} have no lean2vmt "
            "translation; their Lean form would be emitted as an unapplied "
            "leaf, silently changing the transition relation"
        )


def _state_binding(index: int) -> str:
    """How the encoding reads state slot `index` — never `(state i)`."""
    return f"(var_{index} state)"


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

    n = len(ctx.ctrl_next)
    slot_ty = [_flat_element_type(w) for w in ctx.ctrl_next]

    # One slot per wire (`check_na_supported` rejects wider wires), so wire
    # `i` is slot `i` on both the read and the write side.
    update_bindings = {w.id: _state_binding(i) for i, w in enumerate(ctx.ctrl_latched)}
    update_flat = {w.id: [_state_binding(i)] for i, w in enumerate(ctx.ctrl_latched)}

    def _reads_state(terms, wire) -> bool:
        consumed = {
            r.id for t in _reachable_terms(terms, [wire]) for r in t.read
        }
        return any(w.id in consumed for w in ctx.ctrl_latched)

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
    effect_uses_state: list[bool] = []
    for i, w in enumerate(ctx.ctrl_next):
        body = _translate_terms_scalar(
            _reachable_terms(ctx.atom.update, [w]),
            update_bindings,
            [w],
            ctx.constants,
            flat_slots=update_flat,
            flatten_outputs=True,
        )
        uses_state = _reads_state(ctx.atom.update, w)
        effect_uses_state.append(uses_state)
        binder = " (state : StateType)" if uses_state else ""
        lines.append(f"abbrev effect_{i}{binder} : {_effect_type(w)} :=")
        lines.append(body)
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
    for i, w in enumerate(ctx.ctrl_next):
        body = _translate_terms_scalar(
            _reachable_terms(ctx.atom.init, [w]),
            {},
            [w],
            ctx.constants,
            flat_slots={},
            flatten_outputs=True,
        )
        lines.append(f"abbrev init_{i} : {_effect_type(w)} :=")
        lines.append(body)
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
