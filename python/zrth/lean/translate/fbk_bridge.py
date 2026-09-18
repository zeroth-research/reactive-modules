"""The proof that the FBK model is the module.

`translate/fbk.py` writes a model for `lean2vmt`, and the certificate that
comes back is about *that model*. Every other encoding in this package
carries an equivalence theorem to the functional one and would be worthless
without it; this route had none, so a mistranslation anywhere in
`smt_encode`, `_scalar_element`, cvc5's rewriter or `smt_to_lean_bool`
produced a model that parses, model-checks, and describes a different
system.

This module emits that theorem. The chain, and why it is a chain:

    TRANS s s' = true
       |  structural -- `rfl` and `simp`
    slots s' = (effect_0_fn (slots s), ..., effect_{N-1}_fn (slots s))
       |  semantic -- one obligation per slot, two translators disagreeing
       |              about spelling: `x - 1` against `-1 + x`, `ite c b !b`
       |              against an equality, an affine layer expanded
    slots s' = Scalar.update (slots s) () ()
       |  `update_scalar_eq`, already generated and proved by `scalar.py`
    update s () () -- the functional encoding

`Scalar.update` rather than `update` is the target on purpose: it is the
flat-slot transition function, so the obligation the solver sees carries no
matrices. The last link is free.

The one place a goal must not carry the model's own `state` is the semantic
step: `state k` has type `TypeMap k`, a stuck `match` that `omega` will not
read as an `Int` and `smt` will not read at all. Hence `effect_k_fn`, the
same body over honest scalar binders, tied to the model by `rfl`.

`FBK_EQUIVALENCE.md` has the measurements this is built on, including why
the cascade is what it is: `decide` cannot close these goals (free
variables), `split_ifs` before `simp` splits `(x == 0)` and `decide (x = 0)`
into four branches, and the `Linear` cases need the certificate's own matrix
simp set because `_translate_terms_scalar` does not scalarise a `Linear` --
it rebuilds the `Mat` and calls `matVecAffine`.
"""

import re

from zrth.lean.common import (
    LeanContext,
    flat_layout,
)
from zrth.lean.native import _product_type, _product_type_scalar
from zrth.lean.translate.fbk import SlotBodies, check_na_supported

# The certificate's own `simp_mat` arsenal. Without it a `Linear` arrives as
# `matVecAffine 2 [[1, 0], [0, 1]] [0, 1] (fun i j => match i, j with ...)`
# and no arithmetic prover can touch it.
_MAT_SIMP = (
    "MatAdd_apply, MatMul_apply, MatZero_apply, Pi.add_apply, mul_Mat_apply, "
    "add_Mat_apply, Bool.or_eq_true, decide_eq_true_eq, Fin.sum_univ_succ, "
    "Fin.sum_univ_zero, Fin.isValue, Fin.sum_univ_one, Fin.sum_univ_two, "
    "Fin.sum_univ_three, matVecAffine, dotL, List.ofFn_succ, List.ofFn_zero, "
    # The three reductions unfold to a `List.ofFn` fold that the two lemmas
    # above then compute; without the defs here the fold never opens.
    "matMin, matMax, argmax_1d, argmax"
)


def _lemmas(*names: str) -> str:
    """`", a, b, c"`, or `""` when there are none.

    A module can have no pinned slot at all -- every one of them starts at
    an unconstrained input -- and a bare `", "` before the closing bracket
    is a simp set with a hole in it.
    """
    return "".join(f", {n}" for n in names)


def _cascade(defs: str, extra: str = "", *, from_hyps: bool = False) -> list[str]:
    """The closer chain, cheapest first.

    Measured over every module shape the route accepts: `rfl` closes the
    trivial ones and `simp` + `omega` the rest.

    `from_hyps` switches `simp` for `simp_all`, for the one goal whose
    content is in a hypothesis rather than in the definitions: `start_maps`
    under `--pre` has to carry `init_pre l` into a `PRE` conjunct, and a
    `simp` that rewrites only the goal never reads it. Off elsewhere,
    because `simp_all` also rewrites *with* every hypothesis in scope, which
    on the other goals is work that cannot help.

    There is no `smt` arm, and this file cannot have one.  `smt` comes from
    `lean-smt`, which pulls in `auto` and with it `Auto.instBEqInt_auto`, a
    `BEq Int` instance that outranks the `instBEqOfDecidableEq` the NA model
    elaborated its own `==` against -- the model imports `Cslib` alone.
    Adding `import Smt` here re-elaborates `effect_k_fn` against Auto's
    instance, and `link_k := rfl`, the one step that crosses from the model
    into this file, stops being a definitional equality.  The measurements
    say the arm never fired anyway: it was insurance, and the insurance is
    what broke the `rfl`.
    """
    plain = f"{defs}{extra}"
    mat = f"{defs}, {_MAT_SIMP}{extra}"
    s = "simp_all" if from_hyps else "simp"
    return [
        "  first",
        "    | rfl",
        f"    | ({s} [{plain}]; done)",
        f"    | ({s} [{plain}]; omega)",
        f"    | ({s} [{mat}]; done)",
        f"    | ({s} [{mat}]; omega)",
        f"    | ({s} [{mat}]; split_ifs <;> omega)",
        f"    | (simp [{mat}]; simp_all)",
    ]


def atom_to_lean_fbk_bridge(
    ctx: LeanContext,
    *,
    na_module: str,
    simplify: bool = True,
    bodies: SlotBodies | None = None,
    pre_lean: str = "",
) -> str:
    """Emit the bridge file for `ctx`.

    `na_module` is the Lean module name of the NA model (`project.na_module_name`).

    `bodies` are the slot texts the model was written from -- the same ones,
    because the theorem below is about *that* file. Pass what
    `check_na_supported` returned; left out, it is asked again, which encodes
    the module into cvc5 a second time.

    `pre_lean` is what the model's `PRE` was written from, and it is passed
    rather than a flag so that the two files cannot disagree about whether
    there is one: `start_maps` proves `INIT`, and an `INIT` with a conjunct
    this file does not know about is a proof that does not close.
    """
    layout = flat_layout(ctx.ctrl_next)
    n = layout.total
    ty = layout.element_types()
    slot_bodies = bodies if bodies is not None else check_na_supported(ctx, simplify)
    upd = slot_bodies.update
    # `INIT` says nothing about a slot the module starts at an unconstrained
    # input, so there is no `Init_k` to unfold for one (`fbk._free_init_slots`)
    # -- unless `--pre` put it back under a constraint, which is a conjunct
    # over the slots rather than a body for any one of them.
    pinned = [k for k in range(n) if k not in slot_bodies.free_init]
    has_pre = bool(pre_lean)

    binders = " ".join(f"(x{k} : {ty[k]})" for k in range(n))
    args = " ".join(f"x{k}" for k in range(n))
    flat = ", ".join(f"x{k}" for k in range(n))
    slots_of_state = " ".join(f"(Definition.var_{k} state)" for k in range(n))
    ctrl_native = _product_type(ctx.ctrl_next)
    extl_native = (
        f"({_product_type(ctx.extl_latched)}) × ({_product_type(ctx.extl_next)})"
    )
    # The transition reads no input -- `check_na_supported` refuses a module
    # whose does -- so `bridge_k` quantifies over them and `Scalar.update`
    # is applied to the binders rather than to a value that would have to be
    # invented. With no inputs at all both sides are `Unit`, spelled `()`.
    if ctx.extl_next:
        extl_binders = (
            f" (el : {_product_type_scalar(ctx.extl_latched)})"
            f" (en : {_product_type_scalar(ctx.extl_next)})"
        )
        extl_args = "el en"
    else:
        extl_binders, extl_args = "", "() ()"

    def as_fn(body: str) -> str:
        """`(var_k state)` is the model's read; `xk` is the same slot."""
        return re.sub(r"\(var_(\d+) state\)", lambda m: f"x{m.group(1)}", body)

    lines = [
        "/- Why the certificate beside this file is about the module.",
        "",
        f"   `{na_module}` is a model of this module, and the certificate",
        "   `proveit.py` installed proves a property *of that model*. These",
        "   theorems say the model admits every run the module has, so the",
        "   property transfers -- see `module_safety` at the end, and",
        "   `zrth/lean/FBK_EQUIVALENCE.md` for the design. Generated by",
        "   `verith --fbk-proveit`. -/",
        "import System.System",
        "import System.Scalar",
        "import Certificate.Data",
        f"import {na_module}",
        "",
        "",
        "-- The same budgets the certificate and the scalar equivalence carry:",
        "-- a 64-deep straight-line transition exhausts the default recursion",
        "-- depth inside `simp` before any prover sees the goal.",
        "set_option maxRecDepth 100000",
        "set_option maxHeartbeats 2000000",
        "",
        "open LTLFormula",
        "",
        "namespace Bridge",
        "",
        f"abbrev CtrlNative := {ctrl_native}",
        "",
        "/-- A module state, read as the model's slots. The arms are the flat",
        "    element order `common.flat_layout` fixes, which is the order",
        "    `Scalar.unpack_ctrl` unpacks in -- the same call, not a matching",
        "    one. The last arm is `_ + n` rather",
        "    than `_`: at a variable index `TypeMap` is stuck, and only the",
        "    successor pattern lets it reduce to the fallback type. -/",
        "def toSlots (s : CtrlNative) : StateType",
    ]
    for k, slot in enumerate(layout.slots):
        lines.append(f"  | {k} => {layout.wire_accessor('s', slot)}")
    lines.append(f"  | _ + {n} => {layout.wire_accessor('s', layout.slots[-1])}")
    lines.append("")

    lines += [
        "-- ── the model's slot bodies, over honest scalar binders ──────────",
        "--",
        "-- `state k` has type `TypeMap k`, a stuck `match`; `omega` will not",
        "-- read it as an `Int` and `smt` not at all. These are the same",
        "-- bodies with the slots as ordinary variables, which `link_k` shows",
        "-- is the same function.",
        "",
    ]
    for k, body in enumerate(upd):
        lines += [f"abbrev effect_{k}_fn {binders} : {ty[k]} :=", f"  {as_fn(body)}", ""]

    lines += ["-- ── structural: the model reads its slots ────────────────────────", ""]
    for k in range(n):
        lines += [
            f"theorem link_{k} (state : StateType) :",
            f"    Definition.effect_{k} state = effect_{k}_fn {slots_of_state} := rfl",
            "",
        ]

    lines += [
        "-- ── semantic: two independent translators agree ──────────────────",
        "--",
        "-- The left side came through `smt_encode` and cvc5, the right",
        "-- through `_translate_terms_scalar`. `rfl` closes the easy ones and",
        "-- nothing else; the rest is what this file exists for.",
        "",
    ]
    # The right-hand sides are slot `k` of the flat transition, read out by
    # the same layout the left-hand sides were numbered by.
    scalar_slots = layout.flat_accessors(f"(Scalar.update ({flat}) {extl_args})")
    for k in range(n):
        rhs = scalar_slots[k]
        lines += [
            f"theorem bridge_{k} {binders}{extl_binders} :",
            f"    effect_{k}_fn {args} = {rhs} := by",
            *_cascade(f"effect_{k}_fn, Scalar.update"),
            "",
        ]

    lines += [
        "-- ── the module as a reactive module, and the two hypotheses ──────",
        "",
        f"def RM : ReactiveModule ({extl_native}) (CtrlNative) := {{",
        "    init := fun e => init e.2",
        "    update := fun x e => update x e.1 e.2",
        "    init_pre := init_pre",
        "    update_pre := update_pre",
        "}",
        "",
        "theorem start_maps (s : CtrlNative) :",
        "    RM.toTS.start s → Definition.INIT (toSlots s) = true := by",
        "  intro hs",
        "  simp only [RM, ReactiveModule.toTS, ReactiveModule.TS_init] at hs",
        # The `rfl` pattern substitutes `s := init l.2` rather than rewriting
        # under an equation that stays in scope: the arms below reason from
        # the hypotheses, and `init l.2 = s` left in context is a rewrite
        # back to `s` that undoes the step this line just took.
        #
        # `hpre : init_pre l` is named only when `INIT` has a `PRE` conjunct
        # to discharge from it. Without one the module's precondition is
        # `True` and naming it leaves an unused hypothesis; with one it is
        # the whole content of that conjunct -- `PRE (toSlots (init l.2))`
        # reads back the very inputs `init` wrote to those slots, so it *is*
        # `init_pre l`, spelled in `Bool`.
        f"  obtain ⟨l, {'hpre' if has_pre else '_'}, rfl⟩ := hs",
        *_cascade(
            "Definition.INIT, toSlots, init_scalar_eq, Scalar.pack, Scalar.init"
            + (", Definition.PRE, init_pre" if has_pre else ""),
            _lemmas(*(f"Definition.Init_{k}, Definition.var_{k}" for k in pinned)),
            from_hyps=has_pre,
        ),
        "",
        "theorem step_maps (s : CtrlNative) (l : " + extl_native + ") (s' : CtrlNative) :",
        "    RM.toTS.Tr s l s' → Definition.TRANS (toSlots s) (toSlots s') = true := by",
        "  intro hstep",
        "  simp only [RM, ReactiveModule.toTS, ReactiveModule.TS_update] at hstep",
        "  rw [← hstep.2]",
        # `effect_k_fn` is unfolded only when the module has inputs, and it
        # has to be: `bridge_k` is then not a rewrite `simp` can apply,
        # because its right-hand side binds the inputs and its left-hand
        # side does not, so there is nothing to instantiate them from. The
        # transition reads no input (`check_na_supported`), so unfolding the
        # body leaves the same arithmetic on both sides and the cascade
        # closes it where `bridge_k` would have. Left off otherwise: it is
        # another definition in every `simp` arm of every slot.
        *_cascade(
            "Definition.TRANS, toSlots, update_scalar_eq, Scalar.pack, "
            "Scalar.unpack_ctrl",
            ", " + ", ".join(
                f"Definition.R_{k}, Definition.var_{k}, link_{k}, bridge_{k}"
                + (f", effect_{k}_fn" if ctx.extl_next else "")
                for k in range(n)
            ),
        ),
        "",
        "-- ── the property, translated twice and proved equal ──────────────",
        "--",
        "-- `PROPERTY` is `smt_to_lean_bool`'s reading of the formula and `P`",
        "-- is `smt_predicates_to_lean`'s. Without this the chain would be",
        "-- sound about the transition and silent about what is proved of it.",
        "",
        "theorem PROPERTY_iff (s : CtrlNative) :",
        "    Definition.PROPERTY (toSlots s) = true ↔ P s := by",
        *_cascade(
            "Definition.PROPERTY, toSlots, P",
            ", " + ", ".join(f"Definition.var_{k}" for k in range(n)),
        ),
        "",
        "theorem property_eq : (fun s => Definition.PROPERTY (toSlots s) = true) = P := by",
        "  funext s",
        "  exact propext (PROPERTY_iff s)",
        "",
        "-- ── what it all buys ─────────────────────────────────────────────",
        "",
        "/-- the model as a transition system -/",
        "-- The model's own label type, not the module's: the NA reads its",
        "-- inputs out of the state and is labelled by `Unit × Unit`, while a",
        "-- reactive module is labelled by its external inputs. `TS.transfer`",
        "-- takes the map between them, and nothing reads a label.",
        "abbrev MTS : TS StateType (Unit × Unit) := { toNA := M }",
        "",
        "/-- **Safety of the model is safety of the module.**",
        "",
        "    Stated over the module's own `init`/`update` (through `RM`) and",
        "    its own `P` (from `Certificate/Data.lean`), with the model appearing",
        "    only in the hypothesis. Discharging that hypothesis from the",
        "    certificate `proveit.py` installed is the one step this file",
        "    does not take: it is `main_theorem` there, in",
        "    `lean-ltl-certifying`'s LTL vocabulary rather than `Core.LTL`'s,",
        "    and translating between the two is a fixed piece of work that",
        "    does not vary per module. -/",
        "theorem module_safety",
        "    (hM : ∀ ts μs, MTS.ωTrace ts μs →",
        "            ts ⊧ G (AP (fun st => Definition.PROPERTY st = true))) :",
        "    ∀ ss μs, RM.toTS.ωTrace ss μs → ss ⊧ G (AP P) := by",
        "  have h := TS.transfer RM.toTS MTS toSlots (fun _ => ((), ()))\n            start_maps step_maps _ hM",
        "  rw [← property_eq]",
        "  exact h",
        "",
        "end Bridge",
    ]
    return "\n".join(lines) + "\n"
