"""Probe table for the `verith --fbk-proveit` sweep.

One probe is one `(module, safety property)` pair.  `expect` is what the
route *should* report, so a surprise is visible:

    certified  route succeeds and Lean checks the certificate
    unsafe     ic3ia finds a counterexample; the route aborts
    unknown    ic3ia cannot decide; the route aborts
    abort      rejected before ic3ia (unsupported shape or property)
    lean-fail  certificate is produced but Lean rejects it  -- a known bug,
               see README.md; every one of these is upstream, not verith

The modules live in the limit-probe fixture set (`--mods`), the same one
`VERITH_LIMITS.md` uses.  Note the properties are **not** the `P` of that
file's case matrix: `P` there is a reachability target for verith's own
liveness-style certificate (`inv` + `ranking` => `P` is reached), whereas
this route proves `[] PROPERTY`.  Feeding `P` straight through makes every
case trivially unsafe -- Countdown starts at 100, so `[](x = 0)` is false
at step 0.  What transfers is each case's *invariant*, which is the thing
meant to hold always; those are the `inv-` probes below.
"""

# ── A. invariants from the VERITH_LIMITS case matrix, used as safety
#      properties.  These are inductive by construction, so they measure
#      the plumbing rather than ic3ia.
FROM_CASE_MATRIX = [
    ("InvBase",     "m_countdown", "(and (>= s0 0) (<= s0 100))", "certified"),
    ("InvNe",       "m_countdown", "(and (>= s0 0) (<= s0 100) (distinct s0 101))", "certified"),
    ("InvDisj",     "m_step2",     "(or (= s0 0) (= s0 2) (= s0 4) (= s0 6) (= s0 8) (= s0 10))", "certified"),
    ("InvTwoVars",  "m_twovars",   "(and (>= s0 0) (<= s0 s1) (= s1 10))", "certified"),
    ("InvLex",      "m_lex",       "(and (>= s0 0) (<= s0 3) (>= s1 0) (<= s1 3))", "certified"),
    # negative control: x reaches 100, so the bound at 50 is false
    ("InvHalf",     "m_countdown", "(and (>= s0 0) (<= s0 50))", "unsafe"),
    # `mod` now reaches the VMT (lean2vmt translates `%`) and ic3ia proves
    # it safe -- but the *witness* comes back with the mod eliminated, as
    # `x + (-2) * to_int ((1/2) * to_real x) = 0`, and vmt2lean renders
    # neither `to_real` nor `to_int`, nor a Real inside a Bool `INVAR`.
    # So verith still refuses it up front, where the message can say why.
    ("InvMod",      "m_step2",     "(and (= (mod s0 2) 0) (>= s0 0) (<= s0 10))", "abort"),
    # a trivially safe property, so ic3ia's invariant is literally `true`.
    # Two vmt2lean bugs used to fire here at once: `MSAT_TAG_TRUE` rendered
    # as the Lean *Prop* `True` inside `abbrev INVAR : Bool`, and `INVAR`
    # left `state` to the section `variable`, which binds nothing when the
    # invariant mentions no state.
    ("InvTrue",     "m_countdown", "true", "certified"),
]

# ── B. net-shaped invariants: a ReLU net *is* the property.  Built by
#      `cases.net(...)` in the limit harness; inlined here so this file
#      stands alone.
_RELU = "(ite (>= {e} 0) {e} 0)"


def _relu(e):
    return _RELU.format(e=e)


NET_SHAPED = [
    # relu(x) + relu(100-x) = 100  <=>  0 <= x <= 100
    ("NetBox",      "m_countdown",
     f"(= (+ {_relu('(* 1 s0)')} {_relu('(+ (* (- 1) s0) 100)')}) 100)", "certified"),
    # the same as an inequality -- the shape a learned barrier takes
    ("NetBoxIneq",  "m_countdown",
     f"(<= (+ {_relu('(* 1 s0)')} {_relu('(+ (* (- 1) s0) 100)')}) 100)", "certified"),
    # two-input net over (x, y): relu(y-x) + relu(x) = y
    ("NetTwoInput", "m_twovars",
     f"(= (+ {_relu('(+ (* (- 1) s0) (* 1 s1))')} {_relu('(* 1 s0)')}) s1)", "certified"),
    # |x-5| as a two-unit net: a Lyapunov function for a converging plant
    ("NetLyapunov", "m_toward5",
     f"(<= (+ {_relu('(+ (* 1 s0) (- 5))')} {_relu('(+ (* (- 1) s0) 5)')}) 5)", "certified"),
]

# ── C. properties written for this route rather than lifted from the case
#      matrix.  The case invariants are inductive by construction, so they
#      never make ic3ia work; these do.
HAND_WRITTEN = [
    # -- true but NOT inductive: ic3ia has to synthesise the bound itself.
    #    x != 101 holds (reachable set is 0..100) but has a predecessor
    #    outside itself, namely 102.
    ("NiCdNe101",   "m_countdown", "(distinct s0 101)", "certified"),
    ("NiT5Ne4",     "m_toward5",   "(not (= s0 4))", "certified"),
    ("NiLexNe4",    "m_lex",       "(not (= s0 4))", "certified"),
    ("NiTvNe11",    "m_twovars",   "(not (= s0 11))", "certified"),
    # needs *parity*, not just a bound: 3 is inside 0..10 and only
    # unreachable because x steps by 2.  ic3ia finds the invariant; the
    # `smt` tactic then miscompiles the proof -- see lean-smt-bug.md
    ("NiS2Odd3",    "m_step2",     "(not (= s0 3))", "lean-fail"),

    # -- relational: needs an invariant over both state variables at once.
    #    m_toward2d walks x down from 10 and y up from 0 in lockstep, so
    #    the reachable set is the diagonal x + y = 10.
    ("RelT2dSum",   "m_toward2d",  "(= (+ s0 s1) 10)", "certified"),
    ("RelT2dPoint", "m_toward2d",  "(not (and (= s0 6) (= s1 0)))", "certified"),
    ("RelTvImplies", "m_twovars",  "(=> (= s0 5) (= s1 10))", "certified"),

    # -- nonlinear: ic3ia is IC3 with implicit predicate abstraction over
    #    *linear* arithmetic, so a product is out of scope by construction.
    ("NonlinLexMul", "m_lex",      "(<= (* s0 s1) 9)", "unknown"),

    # -- false, and only refutable after a long unrolling: exercises the
    #    counterexample path rather than the invariant path.  x = 0 is
    #    first reached at step 100.
    ("BmcCdReach0", "m_countdown", "(not (= s0 0))", "unsafe"),
    ("BmcT5Reach5", "m_toward5",   "(not (= s0 5))", "unsafe"),

    # -- `ite` in the property itself, so the Bool printer has to nest one.
    ("IteCd",       "m_countdown",
     "(ite (= s0 0) (>= s0 0) (and (> s0 0) (<= s0 100)))", "certified"),

    # -- transition-body size, with a non-inductive property so ic3ia is
    #    doing real work at each depth.
    ("DeepBody64",  "m_deep",      "(distinct s0 101)", "certified"),
    ("DeepBody48",  "m_depth48",   "(distinct s0 101)", "certified"),
    ("DeepBody8",   "m_depth8",    "(distinct s0 101)", "certified"),

    # -- mixed Bool+Int state: a per-index `TypeMap`, so `var_0 : Bool` and
    #    `var_1 : Int` have to reach the VMT as different sorts.
    ("MixedBoolInt", "m_boolint",  "(and (>= s1 0) (<= s1 5))", "certified"),

    # -- ReLU in the *transition*. This was the second lean-smt symptom for
    #    as long as the model spelled it `Max.max 0 (x - 1)`: the VMT was
    #    right and ic3ia proved it, and `smt` then died on the `Max` left in
    #    the model -- "incorrect number of universe levels Max". The NA
    #    encoding now takes its transition from `smt_encode`, where a ReLU
    #    is an `ite`, so no `Max` reaches Lean and the certificate checks.
    #    The bug is still there (`lean-smt-bug.md` keeps the reproducer);
    #    this route no longer walks into it.
    ("ReluTrans",   "m_relu",      "(and (>= s0 0) (<= s0 5))", "certified"),
]

# ── D. harder properties for the modules the route certifies.  Group A's
#      invariants are inductive by construction and group C's are mostly
#      single disequalities; these are neither.  Each one needs a
#      strengthening it does not state -- a coupling between two state
#      components, a parity argument, or a case split -- so ic3ia has to
#      *find* an invariant rather than check one.  Found by running the
#      whole limit matrix through this route (`run_fbk_limits.py`).
HARDER = [
    # -- three bands covering 0..100.  True only because of the bound, and
    #    a three-way case split the property does not state.
    ("CdBands",     "m_countdown",
     "(or (<= s0 33) (and (> s0 33) (<= s0 66)) (and (> s0 66) (<= s0 100)))",
     "certified"),
    # the same bands with the middle one removed: x walks through 50, so
    # this exercises the counterexample path on a *disjunctive* property
    # rather than on a disequality.
    ("CdGapFalse",  "m_countdown", "(or (<= s0 33) (> s0 66))", "unsafe"),
    # -- parity again, but asked as an implication into a disjunction of
    #    equalities instead of a disequality.  Same module and the same
    #    parity argument as `NiS2Odd3`, and it certifies where that one
    #    hits the lean-smt bug -- the evidence that narrowed the
    #    reproducer in lean-smt-bug.md.
    ("Step2Odd",    "m_step2",
     "(=> (>= s0 5) (or (= s0 6) (= s0 8) (= s0 10)))", "certified"),
    # -- relational, with a conjunction under the implication: the bound on
    #    x holds only under y = 10, which the property assumes not states.
    ("TvRelational", "m_twovars",
     "(=> (> s0 0) (and (= s1 10) (<= s0 10)))", "certified"),
    # -- one linear inequality coupling both counters, true only from the
    #    conjunction of their separate bounds (max is 4*3 + 3).
    ("LexLinComb",  "m_lex",       "(<= (+ (* 4 s1) s0) 15)", "certified"),
    # -- an exact disjunctive description of a reachable set, rather than a
    #    box that contains it.
    ("T5Exact",     "m_toward5",
     "(or (= s0 5) (and (>= s0 6) (<= s0 10)))", "certified"),
    # -- Bool/Int split: the same bound under each branch of the mode flag,
    #    so the case split has to survive the encoding as well as ic3ia.
    ("BiSplit",     "m_boolint",
     "(or (and s0 (and (>= s1 0) (<= s1 5))) (and (not s0) (and (>= s1 0) (<= s1 5))))",
     "certified"),
    # -- the three-band property over a 64-deep straight-line transition:
    #    body size and case split at once.
    ("DeepBands",   "m_deep",
     "(or (<= s0 33) (and (> s0 33) (<= s0 66)) (and (> s0 66) (<= s0 100)))",
     "certified"),
    # -- a fixture from `tests/fixtures` rather than the probe set: x resets
    #    before reaching 10, so this is true but not inductive.
    ("CounterNe10", "TESTS/counter", "(not (= s0 10))", "certified"),
    # -- the same plant under an implication rather than a bound: two
    #    property shapes over the transition that used to carry `Max`.
    ("ReluImplies", "m_relu",      "(=> (> s0 0) (<= s0 5))", "certified"),
]

PROBES = FROM_CASE_MATRIX + NET_SHAPED + HAND_WRITTEN + HARDER


# ── D. modules the route rejects before ic3ia ever runs.  `run_fbk.py
#      --screen` checks that each still fails for the reason recorded here,
#      so a lifted restriction shows up as a surprise rather than silently.
#      "verith" = a restriction in `translate/na.py`; "lean2vmt" /
#      "vmt2lean" = an upstream limit that has to be fixed there first.
# Nine modules left this table when the encoding stopped writing the
# transition itself. A wire wider than 1x1 now becomes one state slot per
# element instead of being refused (`m_mixed`, `m_relu_*`, `m_vec32`), and
# an op with no scalar *Lean* form is no longer a problem because no scalar
# Lean is emitted: `smt_encode` gives the term and `smt_to_lean_bool` prints
# it, so `Linear` arrives as its affine sum and `Argmax` as nested `ite`s
# (`m_argmax`, `m_max`, `m_min`). What is left is what neither component can
# express at all.
REJECTED = {
    "m_lra_conv":   ("vmt2lean", "Real state: tp() maps only Int and Bool"),
    "m_lra_half":   ("vmt2lean", "Real state: tp() maps only Int and Bool"),
    "m_lra_lin":    ("vmt2lean", "Real state: tp() maps only Int and Bool"),
    "m_lra_two":    ("vmt2lean", "Real state: tp() maps only Int and Bool"),
    "m_relu_lra":   ("vmt2lean", "Real state: tp() maps only Int and Bool"),
    "m_transpose":  ("verith",   "smt_encode has no term for Transpose"),
    "m_uninterp":   ("verith",   "smt_encode has no term for Uninterpreted"),
    "m_relu_input": ("lean2vmt", "models only state/statenext, no inputs"),
}
