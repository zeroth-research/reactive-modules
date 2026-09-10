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
    # `mod` has no path through: lean2vmt has no case for it, so emitting it
    # would produce a VMT file that parses and describes a different system
    ("InvMod",      "m_step2",     "(and (= (mod s0 2) 0) (>= s0 0) (<= s0 10))", "abort"),
    # ic3ia's invariant is literally `true`; vmt2lean renders that as the
    # Lean *Prop* `True` inside `abbrev INVAR : Bool`
    ("InvTrue",     "m_countdown", "true", "lean-fail"),
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
]

PROBES = FROM_CASE_MATRIX + NET_SHAPED + HAND_WRITTEN


# ── D. modules the route rejects before ic3ia ever runs.  `run_fbk.py
#      --screen` checks that each still fails for the reason recorded here,
#      so a lifted restriction shows up as a surprise rather than silently.
#      "verith" = a restriction in `translate/na.py`; "lean2vmt" /
#      "vmt2lean" = an upstream limit that has to be fixed there first.
REJECTED = {
    "m_mixed":      ("verith",   "ctrl wire holds more than one element"),
    "m_relu_net":   ("verith",   "ctrl wire holds more than one element"),
    "m_relu_net8":  ("verith",   "ctrl wire holds more than one element"),
    "m_relu_net16": ("verith",   "ctrl wire holds more than one element"),
    "m_relu_vec":   ("verith",   "ctrl wire holds more than one element"),
    "m_transpose":  ("verith",   "ctrl wire holds more than one element"),
    "m_vec32":      ("verith",   "ctrl wire holds more than one element"),
    "m_lra_conv":   ("vmt2lean", "Real state: tp() maps only Int and Bool"),
    "m_lra_half":   ("vmt2lean", "Real state: tp() maps only Int and Bool"),
    "m_lra_lin":    ("vmt2lean", "Real state: tp() maps only Int and Bool"),
    "m_lra_two":    ("vmt2lean", "Real state: tp() maps only Int and Bool"),
    "m_relu_lra":   ("vmt2lean", "Real state: tp() maps only Int and Bool"),
    "m_argmax":     ("lean2vmt", "Argmax/Linear reach exprToSMT as a leaf"),
    "m_max":        ("lean2vmt", "Max/Linear reach exprToSMT as a leaf"),
    "m_min":        ("lean2vmt", "Min/Linear reach exprToSMT as a leaf"),
    "m_relu":       ("lean2vmt", "ReLU emits Max.max, printed as `max`"),
    "m_uninterp":   ("lean2vmt", "no Lean counterpart for an uninterpreted op"),
    "m_relu_input": ("lean2vmt", "models only state/statenext, no inputs"),
}
