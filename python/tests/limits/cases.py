"""Case matrix for the `uv run verith` limit probe.

Each case is one full Lean project: `verith ... -o projects/<name> -p Rea`,
then `lake build` against a shared `.lake` (Mathlib + Core + ZerothHammer
already built), so only System/* and Certificate/* recompile.

`expect` records what the case is *meant* to show, so a surprise is visible:
  ok    — should generate and build clean
  fail  — should fail (negative control, or a known limit)
  ?     — genuinely unknown, this is what the probe is for
"""

C = dict

# ──────────────────────────────────────────────────────────────
# Neural-network predicates
#
# A ReLU net written as an SMT-LIB expression over the state, so that an
# invariant or a ranking function can *be* a net rather than merely mention
# one. There is no `max` in the SMT→Lean translator, so a ReLU unit is an
# `ite`; a `k`-unit layer therefore costs `k` branches to `split_ifs`.
#
# Built from explicit weight matrices: the point is that these are nets, not
# formulas reverse-engineered from what the prover happens to close.
# ──────────────────────────────────────────────────────────────


def _num(v, real):
    if not real:
        return f"{v}" if v >= 0 else f"(- {abs(v)})"
    return f"{float(v)}" if v >= 0 else f"(- {float(abs(v))})"


def _lin(row, xs, b):
    """One affine unit: Σ wᵢ·xᵢ + b, skipping zero weights."""
    terms = [f"(* {w} {x})" for w, x in zip(row, xs) if w != 0]
    if b:
        terms.append(f"{b}" if b > 0 else f"(- {abs(b)})")
    if not terms:
        return "0"
    return terms[0] if len(terms) == 1 else "(+ " + " ".join(terms) + ")"


def _relu(e):
    return f"(ite (>= {e} 0) {e} 0)"


def _lin_py(row, xs, b):
    terms = [f"({w} * {x})" for w, x in zip(row, xs) if w != 0]
    if b:
        terms.append(str(b))
    return " + ".join(terms) if terms else "0"


def net_py(layers, xs):
    """`net`, emitted in the Python predicate DSL for tuple-element access."""
    cur = list(xs)
    for i, (W, b) in enumerate(layers):
        out = [_lin_py(row, cur, bias) for row, bias in zip(W, b)]
        cur = out if i == len(layers) - 1 else [f"Ite(({o}) >= 0, {o}, 0)" for o in out]
    return cur[0] if len(cur) == 1 else cur


def net(layers, xs, real=False):
    """Evaluate `layers` = [(W, b), …] on inputs `xs`.

    Every layer but the last is followed by a ReLU; the last is linear, as in
    a regression head. A single-row final layer yields a scalar expression.
    `real=True` emits Real-sorted literals.
    """
    z = _num(0, real)

    def lin(row, cur, bias):
        terms = [f"(* {_num(w, real)} {x})" for w, x in zip(row, cur) if w != 0]
        if bias:
            terms.append(_num(bias, real))
        if not terms:
            return z
        return terms[0] if len(terms) == 1 else "(+ " + " ".join(terms) + ")"

    cur = list(xs)
    for i, (W, b) in enumerate(layers):
        out = [lin(row, cur, bias) for row, bias in zip(W, b)]
        cur = out if i == len(layers) - 1 else [
            f"(ite (>= {o} {z}) {o} {z})" for o in out
        ]
    return cur[0] if len(cur) == 1 else cur


# 1 hidden unit per bias: relu(x - 1), relu(x) → 2·h0 + 1·h1
_RANK_NET_1 = net([([[1], [1]], [-1, 0]), ([[2, 1]], [0])], ["s0"])
# three units with distinct thresholds — a wider layer
_RANK_NET_W = net([([[1], [1], [1]], [-2, -1, 0]), ([[3, 2, 1]], [0])], ["s0"])
# two hidden layers: relu(h0+h1), relu(h0-h1) over relu(x), relu(x-1)
_RANK_NET_2L = net(
    [([[1], [1]], [0, -1]), ([[1, 1], [1, -1]], [0, 0]), ([[1, 1]], [0])], ["s0"]
)
# a "box" net: relu(x) + relu(c - x) = c exactly on 0 ≤ x ≤ c
_BOX_100 = net([([[1], [-1]], [0, 100]), ([[1, 1]], [0])], ["s0"])
_BOX_9 = net([([[1], [-1]], [0, 9]), ([[1, 1]], [0])], ["s0"])
_BOX_4 = net([([[1], [-1]], [0, 4]), ([[1, 1]], [0])], ["s0"])
# two-input box: relu(y - x) + relu(x) = y exactly on 0 ≤ x ≤ y
_BOX_XY = net([([[-1, 1], [1, 0]], [0, 0]), ([[1, 1]], [0])], ["s0", "s1"])
# Over a 2-vector state, reached through the Python DSL.
_BOX_4_PY = net_py([([[1], [-1]], [0, 4]), ([[1, 1]], [0])], ["s[0][0]"])
_NONNEG_PY = net_py([([[1]], [0]), ([[1]], [0])], ["s[1][0]"])
_RANK_NET_PY = net_py([([[1], [1]], [-1, 0]), ([[2, 1]], [0])], ["s[0][0]"])


CASES = [
    # ───────────────────────── baseline ─────────────────────────
    C(name="Countdown", group="baseline", mod="m_countdown", expect="ok",
      P="(= s0 0)", inv="(and (>= s0 0) (<= s0 100))", rank="(ite (= s0 0) 0 s0)",
      note="LIA 1x1, conjunctive bound, Ite ranking — the known-good control"),
    C(name="TwoVars", group="baseline", mod="m_twovars", expect="ok",
      P="(= s0 s1)", inv="(and (>= s0 0) (<= s0 s1) (= s1 10))",
      rank="(ite (= s0 s1) 0 (- s1 s0))",
      note="two 1x1 wires, relational invariant, difference ranking"),
    C(name="NoCert", group="baseline", mod="m_countdown", expect="fail",
      note="no property at all: what does a bare `verith` project contain?"),
    C(name="PropOnly", group="baseline", mod="m_countdown", expect="fail",
      P="(= s0 0)",
      note="--buchi but no --invariant/--ranking: inv defaults to True, ranking to sorry"),

    # ─────────────────────── invariant shapes ───────────────────
    C(name="InvTrue", group="invariant", mod="m_countdown", expect="fail",
      P="(= s0 0)", inv="true", rank="(ite (= s0 0) 0 s0)",
      note="trivially inductive invariant; hrank has no bound to work with"),
    C(name="InvDisj", group="invariant", mod="m_step2", expect="?",
      P="(= s0 0)",
      inv="(or (= s0 0) (= s0 2) (= s0 4) (= s0 6) (= s0 8) (= s0 10))",
      rank="(ite (= s0 0) 0 (- 12 s0))",
      note="6-way disjunctive invariant — needs the casesm* _ v _ branch"),
    C(name="InvMod", group="invariant", mod="m_step2", expect="?",
      P="(= s0 0)", inv="(and (= (mod s0 2) 0) (>= s0 0) (<= s0 10))",
      rank="(ite (= s0 0) 0 (- 12 s0))",
      note="parity invariant: INTS_MODULUS through smt_to_lean, then omega"),
    C(name="InvNe", group="invariant", mod="m_countdown", expect="?",
      P="(= s0 0)", inv="(and (>= s0 0) (<= s0 100) (distinct s0 101))",
      rank="(ite (= s0 0) 0 s0)",
      note="DISTINCT in an invariant"),
    C(name="InvIte", group="invariant", mod="m_boolint", expect="?",
      P="(= s1 0)",
      inv="(and (>= s1 0) (<= s1 5) (ite s0 (<= s1 5) (>= s1 0)))",
      rank="(ite (and (not s0) (= s1 0)) 0 (+ (ite s0 (- 12 s1) s1) 1))",
      note="Ite in Prop position over a Bool state component"),
    C(name="InvImplies", group="invariant", mod="m_boolint", expect="?",
      P="(= s1 0)",
      inv="And(s1[0][0] >= 0, s1[0][0] <= 5, Implies(s0[0][0], s1[0][0] <= 5))",
      rank="(ite (and (not s0) (= s1 0)) 0 (+ (ite s0 (- 12 s1) s1) 1))",
      note="Implies in a Prop position — same module and ranking as InvIte, "
           "so the only difference from that case is the connective"),
    C(name="InvMixed", group="invariant", mod="m_mixed", expect="?",
      P="s1[0][0] == 0",
      inv="And(s1[0][0] >= 0, s1[0][0] <= 1, s1[1][0] == 2, s1[2][0] == 3)",
      rank="Ite(s1[0][0] == 0, 0, s1[0][0])",
      note="1x1 + 3x1 state: wire index and flat slot disagree; x is unbounded"),

    # ──────────────────────── ranking shapes ────────────────────
    C(name="RankConst", group="ranking", mod="m_countdown", expect="fail",
      P="(= s0 0)", inv="(and (>= s0 0) (<= s0 100))", rank="0",
      note="constant ranking — hrank needs 0 < 0"),
    C(name="RankLinear", group="ranking", mod="m_countdown", expect="?",
      P="(= s0 0)", inv="(and (>= s0 0) (<= s0 100))", rank="s0",
      note="bare linear ranking, no Ite guard"),
    C(name="RankLex", group="ranking", mod="m_lex", expect="?",
      P="(and (= s0 0) (= s1 0))",
      inv="(and (>= s0 0) (<= s0 3) (>= s1 0) (<= s1 3))",
      rank="(+ (* 4 s1) s0)",
      note="nested loops; lexicographic (y,x) folded into one Nat"),
    C(name="RankQuadratic", group="ranking", mod="m_countdown", expect="?",
      P="(= s0 0)", inv="(and (>= s0 0) (<= s0 100))", rank="(* s0 s0)",
      note="nonlinear but correct ranking — omega/linarith territory"),
    C(name="RankToInt", group="ranking", mod="m_lra_half", expect="?",
      P="(= s0 0.0)", inv="(or (= s0 0.0) (= s0 0.5) (= s0 1.0) (= s0 1.5) (= s0 2.0) (= s0 2.5) (= s0 3.0))", rank="(to_int (* 2.0 s0))",
      note="non-integral Real state (steps of 0.5): to_int has to scale"),

    # ───────────────────────────── ReLU ─────────────────────────
    C(name="Relu", group="relu", mod="m_relu", expect="?",
      P="(= s0 0)", inv="(and (>= s0 0) (<= s0 5))", rank="s0",
      note="ReLU in the transition: x' = relu(x-1) becomes Max.max 0 (x-1)"),
    C(name="ReluRankRelu", group="relu", mod="m_relu", expect="?",
      P="(= s0 0)", inv="(and (>= s0 0) (<= s0 5))", rank="(ite (> s0 0) s0 0)",
      note="ReLU-shaped ranking (no max kind in smt_to_lean, so Ite)"),
    C(name="ReluInvRelu", group="relu", mod="m_relu", expect="?",
      P="(= s0 0)", inv="(and (= s0 (ite (>= s0 0) s0 0)) (<= s0 5))",
      rank="s0",
      note="ReLU-shaped invariant: `x = relu(x)` written as an Ite"),
    C(name="ReluVec", group="relu", mod="m_relu_vec", expect="?",
      P="s[0][0] == 0",
      inv="And(s[0][0] >= 0, s[0][0] <= 3, s[1][0] >= 0, s[2][0] >= 0)",
      rank="s[0][0]",
      note="element-wise ReLU on a 3-vector state"),
    C(name="ReluLRA", group="relu", mod="m_relu_lra", expect="?",
      P="(= s0 0.0)", inv="(or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0) (= s0 4.0) (= s0 5.0))", rank="(to_int s0)",
      note="ReLU over Real — noncomputable RM, linarith instead of omega"),
    C(name="ReluNet", group="relu", mod="m_relu_net", expect="?",
      P="s[0][0] == 0",
      inv="And(s[0][0] >= 0, s[0][0] <= 4, s[1][0] >= 0)",
      rank="s[0][0]",
      note="Linear -> ReLU -> Linear, the shape a small Q-network compiles to"),
    C(name="ReluInput", group="relu", mod="m_relu_input", expect="?",
      P="(= s0 0)", inv="(and (>= s0 0) (<= s0 5))", rank="s0",
      pre="(>= e0 1)",
      note="ReLU + external input, sound under the precondition"),
    C(name="ReluInputNoPre", group="relu", mod="m_relu_input", expect="fail",
      P="(= s0 0)", inv="(and (>= s0 0) (<= s0 5))", rank="s0",
      note="same module without --pre: e is unconstrained, ranking cannot decrease"),

    # ─────────────────────────── theories ───────────────────────
    C(name="BoolState", group="theory", mod="TESTS/twobit_lia", expect="?",
      P="(and (not s0) (not s1))", inv="true",
      rank="(ite (and (not s0) (not s1)) 0 (- 4 (+ (ite s0 1 0) (* 2 (ite s1 1 0)))))",
      pre="e0",
      note="Bool state (2-bit counter), Bool->Int ranking via Ite"),
    C(name="BVState", group="theory", mod="TESTS/twobit", expect="?",
      P="(and (= s0 (_ bv0 1)) (= s1 (_ bv0 1)))", inv="true",
      rank="(ite (and (= s0 (_ bv0 1)) (= s1 (_ bv0 1))) 0 "
           "(- 4 (+ (ite (= s0 (_ bv1 1)) 1 0) (* 2 (ite (= s1 (_ bv1 1)) 1 0)))))",
      pre="(= e0 (_ bv1 1))",
      note="BitVec state: omega does not model BitVec, and a branch that is "
           "contradictory only because a 1-bit vector has two values needs "
           "the state enumerated"),
    C(name="LRALinear", group="theory", mod="m_lra_lin", expect="?",
      P="(= s0 0.0)", inv="(or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0) (= s0 4.0) (= s0 5.0))",
      rank="(to_int s0)",
      note="Real state, no ReLU — an interval invariant is not inductive "
           "over the reals, so the invariant pins exact values"),

    # ──────────────────────────── ops ───────────────────────────
    C(name="OpMax", group="ops", mod="m_max", expect="?",
      P="(= s0 0)", inv="(and (>= s0 0) (<= s0 5))", rank="s0",
      note="Max as a unary reduction: x' = max(x-1, 0), i.e. ReLU spelled Max"),
    C(name="OpMin", group="ops", mod="m_min", expect="?",
      P="(= s0 5)", inv="(and (>= s0 0) (<= s0 5))", rank="(- 5 s0)",
      note="Min as a unary reduction: x' = min(x+1, 5)"),
    C(name="OpArgmax", group="ops", mod="m_argmax", expect="?",
      P="(= s0 0)", inv="(= s0 0)", rank="s0",
      note="Argmax in the transition — does argmax_1d reduce under the hammer?"),
    C(name="OpTranspose", group="ops", mod="m_transpose", expect="fail",
      P="s[0][0] == 1", inv="s[0][0] == 1", rank="0",
      note="Transpose in the transition (MatTranspose / Box.transpose)"),
    C(name="OpUninterp", group="ops", mod="m_uninterp", expect="fail",
      P="(= s0 0)", inv="true", rank="0",
      note="an uninterpreted symbol — no Lean counterpart exists"),

    # ─────────────────────────── scale ──────────────────────────
    C(name="Vec32", group="scale", mod="m_vec32", expect="?", timeout=2400,
      P="s[0][0] == 0", inv="And(s[0][0] >= 0, s[0][0] <= 100)",
      rank="Ite(s[0][0] == 0, 0, s[0][0])",
      note="32-wide state, all six encodings — the scaling ceiling"),
    C(name="Deep64", group="scale", mod="m_deep", expect="?", timeout=1200,
      P="(= s0 0)", inv="(and (>= s0 0) (<= s0 100))", rank="(ite (= s0 0) 0 s0)",
      note="64-deep straight-line transition body"),


    # ─────────────────── neural-network certificates ────────────────
    C(name="NNRank", group="nn", mod="m_countdown", expect="?",
      P="(= s0 0)", inv="(and (>= s0 0) (<= s0 100))", rank=_RANK_NET_1,
      note="ranking is a 2-unit ReLU net: 2·relu(x-1) + relu(x)"),
    C(name="NNRankWide", group="nn", mod="m_countdown", expect="?",
      P="(= s0 0)", inv="(and (>= s0 0) (<= s0 100))", rank=_RANK_NET_W,
      note="ranking is a 3-unit ReLU net with distinct thresholds"),
    C(name="NNRankDeep", group="nn", mod="m_countdown", expect="?",
      P="(= s0 0)", inv="(and (>= s0 0) (<= s0 100))", rank=_RANK_NET_2L,
      note="ranking is a two-hidden-layer ReLU net"),
    C(name="NNInv", group="nn", mod="TESTS/counter", expect="?",
      P="(= s0 0)", inv=f"(= {_BOX_9} 9)", rank="(ite (= s0 0) 0 (- 10 s0))",
      note="invariant is a ReLU net: relu(x) + relu(9-x) = 9, i.e. 0 ≤ x ≤ 9"),
    C(name="NNInvIneq", group="nn", mod="m_countdown", expect="?",
      P="(= s0 0)", inv=f"(<= {_BOX_100} 100)", rank="(ite (= s0 0) 0 s0)",
      note="net invariant in inequality form — the shape a learned "
           "barrier/Lyapunov function takes"),
    C(name="NNBoth", group="nn", mod="m_countdown", expect="?",
      P="(= s0 0)", inv=f"(= {_BOX_100} 100)", rank=_RANK_NET_1,
      note="invariant and ranking both nets, over the same state"),
    C(name="NNTwoInput", group="nn", mod="m_twovars", expect="?",
      P="(= s0 s1)", inv=f"(and (= {_BOX_XY} s1) (= s1 10))",
      rank="(ite (= s0 s1) 0 (- s1 s0))",
      note="two-input net invariant: relu(y-x) + relu(x) = y"),
    C(name="NNNetModule", group="nn", mod="m_relu_net", expect="?",
      P="s[0][0] == 0",
      inv=f"And(({_BOX_4_PY}) == 4, ({_NONNEG_PY}) == s[1][0])",
      rank=_RANK_NET_PY,
      note="the module is Linear->ReLU->Linear *and* both certificate "
           "predicates are nets — the fully neural case"),

    # ──────────── shapes a single fixed chain cannot serve ──────────
    C(name="RealConjDisj", group="hard", mod="m_lra_conv", expect="?",
      P="(and (= s0 0.0) (= s1 2.0))",
      inv="(and (or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0)) "
          "(or (= s1 2.0) (= s1 3.0) (= s1 4.0) (= s1 5.0)))",
      rank="(+ (to_int s0) (to_int (- s1 2.0)))",
      note="Real, invariant is a conjunction of two 4-way disjunctions: omega "
           "does not apply and `constructor <;> linarith` cannot prove a "
           "disjunct, so the plan has to case on both"),
    C(name="RealConjDisjUnsat", group="control", mod="m_lra_two", expect="fail",
      P="(and (= s0 0.0) (= s1 2.0))",
      inv="(and (or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0)) (or (= s1 2.0) (= s1 5.0)))",
      rank="(ite (and (= s0 0.0) (= s1 2.0)) 0 (+ (to_int (- 4.0 s0)) (ite (= s1 2.0) 0 4)))",
      note="two Real components that both *cycle*: the product of their ranges "
           "admits (1,2)->(2,5)->(3,2)->(0,5)->(1,2), which never reaches P, so "
           "hrank is false for every ranking. Must be rejected"),
    C(name="RealNonlin", group="hard", mod="m_lra_lin", expect="?",
      P="(= s0 0.0)", inv="(or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0) (= s0 4.0) (= s0 5.0))",
      rank="(to_int (* s0 s0))",
      note="Real *and* nonlinear: needs nlinarith over an ordered field, "
           "not omega"),
    C(name="RealNet", group="hard", mod="m_lra_lin", expect="?",
      P="(= s0 0.0)", inv="(or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0) (= s0 4.0) (= s0 5.0))",
      rank=f"(to_int {net([([[1], [1]], [-1, 0]), ([[2, 1]], [0])], ["s0"], real=True)})",
      note="a ReLU net over Real as the ranking: ite branches, real "
           "literals and a floor, all at once"),
    # ────────────────────────── controls ────────────────────────
    C(name="BadInv", group="control", mod="m_countdown", expect="fail",
      P="(= s0 0)", inv="(and (>= s0 0) (<= s0 50))", rank="(ite (= s0 0) 0 s0)",
      note="not inductive (init is 100) — init_inv must fail"),
    C(name="BadRankDir", group="control", mod="m_countdown", expect="fail",
      P="(= s0 0)", inv="(and (>= s0 0) (<= s0 100))", rank="(- 100 s0)",
      note="ranking increases along the transition — hrank must fail"),
]


# ══════════════════════════════════════════════════════════════════
# Group `nn2`: where do neural certificates actually stop working?
#
# The eight `nn` cases all pass, so they are not probing hard enough. These
# push the three axes a *learned* certificate really varies along — hidden
# width, depth, and the sort/sign/scale of the weights — plus the shapes a
# learned certificate takes that a hand-written one does not: a piecewise
# linear Lyapunov function, a net on each side of a comparison, a net whose
# value dips below zero and is clamped by `Int.toNat`.
#
# Every obligation below was verified numerically by exhaustion over the
# invariant's states before the case was added (`check_nn2.py`): a wrong test
# failing is not a finding.
# ══════════════════════════════════════════════════════════════════


def _width(k):
    """One hidden layer, unit j = relu(x - j), output weight (k - j).

    Strictly increasing in x on x ≥ 0 and non-negative, and the j = 0 unit
    makes the decrease strict, so it is a valid ranking for a countdown.
    All k conditions are syntactically distinct, so `split_ifs` cannot merge
    them: k conditions per copy of the net, 2k in `hrank`.
    """
    return [([[1]] * k, [-j for j in range(k)]),
            ([[k - j for j in range(k)]], [0])]


def _width_pos(k):
    """k units relu(x + j), every one of them non-negative under x >= 0.

    Same shape and same cost as , but the invariant *settles*
    every condition instead of just the first, so --smt-tactics has all k to
    discharge rather than one. rank = sum (k-j)(x+j), whose x coefficient is
    k(k+1)/2 > 0, so it is still a valid ranking for a countdown.
    """
    return [([[1]] * k, [j for j in range(k)]),
            ([[k - j for j in range(k)]], [0])]


def _width_dup(k):
    """k *identical* units relu(x) with distinct output weights.

    Same term count as `_width(k)`, but one distinct condition instead of k —
    the control that separates term size from branch count.
    """
    return [([[1]] * k, [0] * k), ([[j + 1 for j in range(k)]], [0])]


def _deep(L):
    """relu(x), relu(x-1) then L-1 dense [[1,1],[1,-1]] layers, summed.

    Dense, so the *text* of the net doubles per layer as well as adding two
    conditions. On x ≥ 1 the value is affine and increasing; at x = 0 it is 0.
    """
    return ([([[1], [1]], [0, -1])]
            + [([[1, 1], [1, -1]], [0, 0])] * (L - 1)
            + [([[1, 1]], [0])])


def _narrow(L):
    """L pass-through layers: relu(relu(...relu(x))) = x on x ≥ 0.

    L conditions but only linear growth in text — the other half of the
    text-size / branch-count separation.
    """
    return [([[1]], [0])] * L + [([[1]], [0])]


def _invbox(m):
    """2m units whose sum is constant exactly on 0 ≤ x ≤ 100.

    Σ_j relu(x + j) + relu(100 + j - x) is convex, flat at C = Σ (100 + 2j)
    on the box and strictly larger outside it, so `= C` *is* the box — a
    genuine net invariant of tunable width.
    """
    W, b = [], []
    for j in range(m):
        W += [[1], [-1]]
        b += [j, 100 + j]
    return [(W, b), ([[1] * (2 * m)], [0])]


def _invbox_c(m):
    return sum(100 + 2 * j for j in range(m))


_RANK3 = [([[1], [1], [1]], [0, -1, -2]), ([[3, 2, 1]], [0])]
_XS3 = ["s[0][0]", "s[1][0]", "s[2][0]"]
_XS8 = [f"s[{i}][0]" for i in range(8)]

# |x - 5| and |x-5| + |y-5| as ReLU nets — piecewise-linear Lyapunov functions
_LYAP1 = [([[1], [-1]], [-5, 5]), ([[1, 1]], [0])]
_LYAP2 = [([[1, 0], [-1, 0], [0, 1], [0, -1]], [-5, 5, -5, 5]),
          ([[1, 1, 1, 1]], [0])]
# mixed signs, a bias that keeps several branches simultaneously live, and a
# final bias that keeps the value non-negative so `Int.toNat` never clamps
_MIXED = [([[1], [-1], [1]], [-10, 10, 0]), ([[2, -1, 3]], [10])]
# net(x) = 3x - 2: negative at x = 0, so toNat *does* clamp — and the
# obligation is still true, because the clamp only bites where P holds
_TONAT = [([[3]], [0]), ([[1]], [-2])]
_BIGW = [([[1], [1], [1]], [0, -1, -2]), ([[10007, 3001, 499]], [0])]
# nets on both sides of a comparison: relu(x) + relu(y-x) ≤ relu(y)
_CMP_L = net([([[1, 0], [-1, 1]], [0, 0]), ([[1, 1]], [0])], ["s0", "s1"])
_CMP_R = net([([[0, 1]], [0]), ([[1]], [0])], ["s0", "s1"])
# Real: fractional weights, mixed signs, a negative output weight
_RFRAC = [([[1.5], [0.25], [-0.5]], [-0.5, 0, 2.0]),
          ([[1.0, 2.0, -1.0]], [1.0])]
_RBOX5 = net([([[1], [-1]], [0, 5]), ([[1, 1]], [0])], ["s0"], real=True)
# 3-input net over a vector state: Σ relu(vᵢ) = Σ vᵢ iff every vᵢ ≥ 0
_VECSUM = net_py([([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0]),
                  ([[1, 1, 1]], [0])], _XS3)
_VECRANK = net_py([([[1, 0, 0]], [0]), ([[1]], [0])], _XS3)
# 8-input, 2-unit net: a box on Σ sᵢ, over the 32-wide state
_WIDEIN = net_py([([[1] * 8, [-1] * 8], [0, 100]), ([[1, 1]], [0])], _XS8)
_NM_BOX6 = net_py([([[1], [-1]], [0, 6]), ([[1, 1]], [0])], ["s[0][0]"])
_NM_NN1 = net_py([([[1]], [0]), ([[1]], [0])], ["s[1][0]"])
_NM_RANK = net_py(_RANK3, ["s[0][0]"])

_BOX_INV = "(and (>= s0 0) (<= s0 100))"
_ITE_RANK = "(ite (= s0 0) 0 s0)"

CASES += [
    # ─────────── hidden width: how many units before it gives way ──────────
    C(name="NN2Width4", group="nn2", mod="m_countdown", expect="?",
      P="(= s0 0)", inv=_BOX_INV, rank=net(_width(4), ["s0"]),
      note="4-unit ranking net, distinct thresholds — 8 ite in hrank"),
    C(name="NN2Width6", group="nn2", mod="m_countdown", expect="?",
      P="(= s0 0)", inv=_BOX_INV, rank=net(_width(6), ["s0"]),
      note="6-unit ranking net — 12 ite in hrank"),
    C(name="NN2Width8", group="nn2", mod="m_countdown", expect="?",
      P="(= s0 0)", inv=_BOX_INV, rank=net(_width(8), ["s0"]),
      note="8-unit ranking net — 16 ite in hrank"),
    C(name="NN2Width10", group="nn2", mod="m_countdown", expect="?", timeout=600,
      P="(= s0 0)", inv=_BOX_INV, rank=net(_width(10), ["s0"]),
      note="10-unit ranking net — 20 ite in hrank"),
    C(name="NN2Width12", group="nn2", mod="m_countdown", expect="?", timeout=600,
      P="(= s0 0)", inv=_BOX_INV, rank=net(_width(12), ["s0"]),
      note="12-unit ranking net — 24 ite in hrank"),
    C(name="NN2WidthDup8", group="nn2", mod="m_countdown", expect="?",
      P="(= s0 0)", inv=_BOX_INV, rank=net(_width_dup(8), ["s0"]),
      note="8 units but one distinct condition — control separating the "
           "cost of net *size* from the cost of net *branching*"),

    # ─────────────────────────── hidden depth ──────────────────────────────
    C(name="NN2Deep3", group="nn2", mod="m_countdown", expect="?",
      P="(= s0 0)", inv=_BOX_INV, rank=net(_deep(3), ["s0"]),
      note="3 dense hidden layers — 6 ite per copy, text doubles per layer"),
    C(name="NN2Deep4", group="nn2", mod="m_countdown", expect="?", timeout=600,
      P="(= s0 0)", inv=_BOX_INV, rank=net(_deep(4), ["s0"]),
      note="4 dense hidden layers — 8 ite per copy, 16 in hrank"),
    C(name="NN2Deep5", group="nn2", mod="m_countdown", expect="?", timeout=600,
      P="(= s0 0)", inv=_BOX_INV, rank=net(_deep(5), ["s0"]),
      note="5 dense hidden layers — 10 ite per copy, 20 in hrank"),
    C(name="NN2Narrow8", group="nn2", mod="m_countdown", expect="?", timeout=600,
      P="(= s0 0)", inv=_BOX_INV, rank=net(_narrow(8), ["s0"]),
      note="8 layers of one unit each: same 16 ite as NN2Width8 but linear "
           "text — does depth or does size cost?"),

    # ─────────────── width on the invariant side (step_inv) ────────────────
    C(name="NN2InvWide4", group="nn2", mod="m_countdown", expect="?",
      P="(= s0 0)", inv=f"(= {net(_invbox(2), ['s0'])} {_invbox_c(2)})",
      rank=_ITE_RANK,
      note="invariant is a 4-unit net equality that *is* the box 0..100"),
    C(name="NN2InvWide8", group="nn2", mod="m_countdown", expect="?", timeout=600,
      P="(= s0 0)", inv=f"(= {net(_invbox(4), ['s0'])} {_invbox_c(4)})",
      rank=_ITE_RANK,
      note="the same, 8 units wide — step_inv now carries 16 ite"),

    # ─────────────────────────── nets over ℝ ───────────────────────────────
    C(name="NN2RealAllPos4", group="nn2", mod="m_lra_lin", expect="?",
      P="(= s0 0.0)",
      inv="(or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0) (= s0 4.0) (= s0 5.0))",
      rank=f"(to_int {net(_width_pos(4), ['s0'], real=True)})",
      note="4-unit net over Real whose every ReLU is non-negative under the "
           "invariant — the shape --smt-tactics is built for"),
    C(name="NN2RealWide4", group="nn2", mod="m_lra_lin", expect="?",
      P="(= s0 0.0)",
      inv="(or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0) (= s0 4.0) (= s0 5.0))",
      rank=f"(to_int {net(_width(4), ['s0'], real=True)})",
      note="4-unit net over Real — no omega, linarith on 2^8 branches"),
    C(name="NN2RealFrac", group="nn2", mod="m_lra_lin", expect="?",
      P="(= s0 0.0)",
      inv="(or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0) (= s0 4.0) (= s0 5.0))",
      rank=f"(to_int {net(_RFRAC, ['s0'], real=True)})",
      note="weights 1.5 / 0.25 / -0.5 and a negative output weight — the "
           "weights a trained net actually has"),
    C(name="NN2RealNetInv", group="nn2", mod="m_lra_lin", expect="?",
      P="(= s0 0.0)",
      inv="(and (or (= s0 0.0) (= s0 1.0) (= s0 2.0) (= s0 3.0) (= s0 4.0) "
          f"(= s0 5.0)) (= {_RBOX5} 5.0))",
      rank="(to_int s0)",
      note="a net *invariant* over Real: the exact-value disjunction keeps it "
           "inductive, the net box is the learned part"),

    # ──────────────── nets over vector state / net modules ─────────────────
    C(name="NN2VecNet", group="nn2", mod="m_relu_vec", expect="?",
      P="s[0][0] == 0",
      inv=f"And(({_VECSUM}) == s[0][0] + s[1][0] + s[2][0], ({_VECSUM}) <= 6)",
      rank=_VECRANK,
      note="3-input net invariant over a 3-vector state: Σ relu(vᵢ) = Σ vᵢ "
           "is componentwise non-negativity, and Σ relu(vᵢ) ≤ 6 bounds it"),
    C(name="NN2NetMod8", group="nn2", mod="m_relu_net8", expect="?",
      P="s[0][0] == 0",
      inv=f"And(({_NM_BOX6}) == 6, ({_NM_NN1}) == s[1][0])",
      rank=_NM_RANK,
      note="the module's own net is 8 wide and both predicates are nets — "
           "module width against certificate width"),
    C(name="NN2NetMod16", group="nn2", mod="m_relu_net16", expect="?", timeout=900,
      P="s[0][0] == 0",
      inv=f"And(({_NM_BOX6}) == 6, ({_NM_NN1}) == s[1][0])",
      rank=_NM_RANK,
      note="the same with a 16-wide module net"),
    C(name="NN2WideInput", group="nn2", mod="m_vec32", expect="?", timeout=900,
      P="s[0][0] == 0",
      inv="And((" + _WIDEIN + ") == 100, "
          + ", ".join(f"s[{i}][0] == 0" for i in range(1, 8)) + ")",
      rank=_NM_RANK,
      note="a net that reads 8 state slots but has only 2 units — input "
           "width without branch width, over the 32-wide state"),

    # ─────────────────── shapes a learned certificate takes ────────────────
    C(name="NN2MixedSign", group="nn2", mod="m_countdown", expect="?",
      P="(= s0 0)", inv=_BOX_INV, rank=net(_MIXED, ["s0"]),
      note="mixed-sign weights, thresholds at 0 and 10 so two branches are "
           "live at once, output bias keeping the value non-negative"),
    C(name="NN2BigWeights", group="nn2", mod="m_countdown", expect="?",
      P="(= s0 0)", inv=_BOX_INV, rank=net(_BIGW, ["s0"]),
      note="weights 10007 / 3001 / 499 — coefficient size, not branch count"),
    C(name="NN2ToNatClamp", group="nn2", mod="m_countdown", expect="?",
      P="(= s0 0)", inv=_BOX_INV, rank=net(_TONAT, ["s0"]),
      note="net(x) = 3x - 2 is negative at x = 0, so Int.toNat clamps; the "
           "obligation is still true because the clamp only bites under P"),
    C(name="NN2CmpBoth", group="nn2", mod="m_twovars", expect="?",
      P="(= s0 s1)", inv=f"(and (<= {_CMP_L} {_CMP_R}) (= s1 10))",
      rank="(ite (= s0 s1) 0 (- s1 s0))",
      note="a net on each side of ≤ — relu(x)+relu(y-x) ≤ relu(y) is 0≤x≤y"),
    C(name="NN2Lyapunov", group="nn2", mod="m_toward5", expect="?",
      P="(= s0 5)", inv=f"(<= {net(_LYAP1, ['s0'])} 5)",
      rank=net(_LYAP1, ["s0"]),
      note="|x-5| as a 2-unit net, used as *both* invariant and ranking over "
           "a plant that converges from either side: a genuine piecewise-"
           "linear Lyapunov function whose decrease needs the ReLU split"),
    C(name="NN2Lyap2D", group="nn2", mod="m_toward2d", expect="?", timeout=600,
      P="(and (= s0 5) (= s1 5))",
      inv=f"(<= {net(_LYAP2, ['s0', 's1'])} 10)",
      rank=net(_LYAP2, ["s0", "s1"]),
      note="the same in two dimensions: a 4-unit Lyapunov net over a plant "
           "with two independent piecewise-linear legs"),
]


# ── follow-ups, added once the first sweep located the knee ──────────
# NNRankWide is already the width-3 point of the `_width` family (same three
# units, weights permuted) and it passes, so 3 and 5 bracket the knee from
# both sides. NN2Width6Big is the decisive control: the *same* 6-unit ranking
# net and the *same* `if x = 0 then reset else x-1` transition, but over the
# 32-slot module, which `tactics.py` classifies as "slow" and therefore gives
# 2 000 000 heartbeats instead of 400 000.
CASES += [
    C(name="NN2Width3", group="nn2", mod="m_countdown", expect="?",
      P="(= s0 0)", inv=_BOX_INV, rank=net(_width(3), ["s0"]),
      note="3-unit ranking net — the last width that closes"),
    C(name="NN2Width5", group="nn2", mod="m_countdown", expect="?",
      P="(= s0 0)", inv=_BOX_INV, rank=net(_width(5), ["s0"]),
      note="5-unit ranking net — bisects the knee"),
    C(name="NN2Width6Big", group="nn2", mod="m_vec32", expect="?", timeout=900,
      P="s[0][0] == 0", inv="And(s[0][0] >= 0, s[0][0] <= 100)",
      rank=net_py(_width(6), ["s[0][0]"]),
      note="the 6-unit net of NN2Width6 over the 32-slot module: identical "
           "certificate and identical transition shape, but the plan calls "
           "the module slow and raises the heartbeat budget 5x"),
]
