"""The NA encoding and the `--fbk-proveit` route (`zrth.lean.translate.fbk`).

The shape asserted here is not cosmetic: `lean2vmt` in `lean-ltl-certifying`
pattern-matches on it, and every mismatch is silent. `(state i)` in place of
`var_i state`, for instance, still elaborates and still produces a VMT file —
one where all state slots have collapsed onto a single variable.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

from zrth import Bool, Int, LIA, LRA, Module, Real, Term, Var, Wire, X
from zrth.lean.common import LeanContext, dtype_shape
from zrth.lean.fbk_proveit import ProveItError, property_to_bool_lean, resolve_project
from zrth.lean.translate.fbk import NAUnsupported, atom_to_lean_na, check_na_supported

FIXTURE_DIR = Path(__file__).parent / "fixtures"
PKG_ROOT = Path(__file__).parent.parent


# ── module builders ─────────────────────────────────────────────────────────


def _counter() -> Module:
    """x := if x + 1 = 10 then 0 else x + 1, from 0."""
    x = Var(Int([1, 1]))
    one = Wire(Int([1, 1]))
    ten = Wire(Int([1, 1]))
    zero = Wire(Int([1, 1]))
    inc = Wire(Int([1, 1]))
    at_ten = Wire(Bool([1, 1]))
    init = [Term(LIA.Int(torch.tensor([[0]])), [X(x)])]
    update = [
        Term(LIA.Int(torch.tensor([[1]])), [one]),
        Term(LIA.Int(torch.tensor([[10]])), [ten]),
        Term(LIA.Int(torch.tensor([[0]])), [zero]),
        Term(LIA.Add(), [inc], [x, one]),
        Term(LIA.Eq(), [at_ten], [inc, ten]),
        Term(LIA.Ite(), [X(x)], [at_ten, zero, inc]),
    ]
    return Module.sequential([x], init, update)


def _two_bools() -> Module:
    """b0 toggles, b1 follows b0."""
    b0 = Var(Bool([1, 1]))
    b1 = Var(Bool([1, 1]))
    not_b0 = Wire(Bool([1, 1]))
    init = [
        Term(LIA.Bool(torch.tensor([[False]])), [X(b0)]),
        Term(LIA.Bool(torch.tensor([[False]])), [X(b1)]),
    ]
    update = [
        Term(LIA.Not(), [not_b0], [b0]),
        Term(LIA.Id(), [X(b0)], [not_b0]),
        Term(LIA.Id(), [X(b1)], [b0]),
    ]
    return Module.sequential([b0, b1], init, update)


def _with_extl() -> Module:
    """x := x + e. `e` is never written, so it is an external input."""
    x = Var(Int([1, 1]))
    e = Var(Int([1, 1]))
    init = [Term(LIA.Int(torch.tensor([[0]])), [X(x)])]
    update = [Term(LIA.Add(), [X(x)], [x, X(e)])]
    return Module.sequential([x, e], init, update)


def _real_state() -> Module:
    x = Var(Real([1, 1]))
    one = Wire(Real([1, 1]))
    init = [Term(LRA.Real(torch.tensor([[0.0]])), [X(x)])]
    update = [
        Term(LRA.Real(torch.tensor([[1.0]])), [one]),
        Term(LRA.Add(), [X(x)], [x, one]),
    ]
    return Module.sequential([x], init, update)


def _mixed_state() -> Module:
    """Int and Bool state side by side."""
    n = Var(Int([1, 1]))
    flag = Var(Bool([1, 1]))
    one = Wire(Int([1, 1]))
    init = [
        Term(LIA.Int(torch.tensor([[0]])), [X(n)]),
        Term(LIA.Bool(torch.tensor([[False]])), [X(flag)]),
    ]
    update = [
        Term(LIA.Int(torch.tensor([[1]])), [one]),
        Term(LIA.Add(), [X(n)], [n, one]),
        Term(LIA.Lt(), [X(flag)], [n, one]),
    ]
    return Module.sequential([n, flag], init, update)


def _uses_ne() -> Module:
    """`Ne` reaches `exprToSMT` as `instDecidableNot`."""
    n = Var(Int([1, 1]))
    one = Wire(Int([1, 1]))
    differs = Wire(Bool([1, 1]))
    init = [Term(LIA.Int(torch.tensor([[0]])), [X(n)])]
    update = [
        Term(LIA.Int(torch.tensor([[1]])), [one]),
        Term(LIA.Ne(), [differs], [n, one]),
        Term(LIA.Ite(), [X(n)], [differs, one, n]),
    ]
    return Module.sequential([n], init, update)


def _wide_vec() -> Module:
    """`m_vec32` in miniature: a 3-wide state, `Linear` and a wide `Ite`.

    The shape of every module the width restriction used to refuse -- a
    scalar condition over two matrix-valued branches, each an affine map of
    the whole state. Element 0 counts down from 9 and resets; the other two
    are carried along.
    """
    n = 3
    v = Var(Int([n, 1]))
    z11 = torch.zeros((1, 1), dtype=torch.int64)
    x, zc = Wire(Int([1, 1])), Wire(Int([1, 1]))
    cond = Wire(Bool([1, 1]))
    reset, dec = Wire(Int([n, 1])), Wire(Int([n, 1]))

    row0 = torch.zeros((1, n), dtype=torch.int64)
    row0[0][0] = 1
    keep = torch.eye(n, dtype=torch.int64)
    keep[0][0] = 0
    b9 = torch.zeros((n, 1), dtype=torch.int64)
    b9[0][0] = 9
    bneg = torch.zeros((n, 1), dtype=torch.int64)
    bneg[0][0] = -1

    init_vec = torch.zeros((n, 1), dtype=torch.int64)
    init_vec[0][0] = 9
    return Module.sequential(
        [v],
        [Term(LIA.Int(init_vec), [X(v)])],
        [
            Term(LIA.Linear(row0, z11), [x], [v]),
            Term(LIA.Int(torch.tensor([[0]])), [zc]),
            Term(LIA.Eq(), [cond], [x, zc]),
            Term(LIA.Linear(keep, b9), [reset], [v]),
            Term(LIA.Linear(torch.eye(n, dtype=torch.int64), bneg), [dec], [v]),
            Term(LIA.Ite(), [X(v)], [cond, reset, dec]),
        ],
    )


def _wide_mixed() -> Module:
    """A Bool wire and a 3-wide Int wire: slots of two types in one state.

    Each element moves by a different amount, so a slot reading the wrong
    element shows up as a wrong value rather than a wrong shape.
    """
    b = Var(Bool([1, 1]))
    v = Var(Int([3, 1]))
    nb = Wire(Bool([1, 1]))
    step = Wire(Int([3, 1]))
    moved = Wire(Int([3, 1]))
    init = [
        Term(LIA.Bool(torch.tensor([[False]])), [X(b)]),
        Term(LIA.Int(torch.tensor([[0], [1], [2]])), [X(v)]),
    ]
    update = [
        Term(LIA.Not(), [nb], [b]),
        Term(LIA.Id(), [X(b)], [nb]),
        Term(LIA.Int(torch.tensor([[1], [10], [100]])), [step]),
        Term(LIA.Add(), [moved], [v, step]),
        Term(LIA.Id(), [X(v)], [moved]),
    ]
    return Module.sequential([b, v], init, update)


def _wide_matrix() -> Module:
    """A genuine 2x2 state: the only shape that can tell row-major from
    column-major, since either order flattens a vector the same way."""
    v = Var(Int([2, 2]))
    step = Wire(Int([2, 2]))
    moved = Wire(Int([2, 2]))
    return Module.sequential(
        [v],
        [Term(LIA.Int(torch.tensor([[0, 1], [2, 3]])), [X(v)])],
        [
            Term(LIA.Int(torch.tensor([[1, 10], [100, 1000]])), [step]),
            Term(LIA.Add(), [moved], [v, step]),
            Term(LIA.Id(), [X(v)], [moved]),
        ],
    )


def _na(module: Module, prop_smt: str) -> str:
    ctx = LeanContext(module)
    prop = property_to_bool_lean(module, prop_smt, len(ctx.ctrl_next))
    return atom_to_lean_na(ctx, prop, module_name="T")


# ── the shape lean2vmt reads ────────────────────────────────────────────────


def test_state_is_read_through_var_abbrevs_never_through_the_binder():
    """`(state i)` would collapse every slot onto one VMT variable.

    `exprToSMT`'s `.fvar` case returns the binder's name and discards the
    application's argument, so only `var_i state` survives the translation.
    """
    src = _na(_two_bools(), "(not s0)")
    assert "abbrev var_0 (state : StateType) : Bool := state 0" in src
    assert "abbrev var_1 (state : StateType) : Bool := state 1" in src
    body = src.split("namespace Definition", 1)[1]
    for line in body.splitlines():
        if line.startswith("abbrev var_"):
            continue
        assert not re.search(r"\bstate\s+\d", line), f"indexed state read in: {line}"
    assert "var_0 state" in body and "var_1 state" in body


def test_next_state_is_var_i_statenext():
    """`collectLatchesIndices` finds latches only through this application."""
    src = _na(_counter(), "(not (= s0 15))")
    assert "abbrev R_0 (state statenext : StateType) : Bool:=" not in src
    assert "  var_0 statenext == effect_0 state" in src
    assert "newstate" not in src


def test_definition_names_are_the_ones_lean2vmt_annotates():
    src = _na(_counter(), "(not (= s0 15))")
    for name in ("TRANS", "INIT", "PROPERTY"):
        assert f"abbrev {name} " in src
    for fbk_name in ("TransRel", "InitCond"):
        assert fbk_name not in src


def test_state_type_is_top_level_and_M_is_emitted():
    """The certificate template names `StateType` and `M` unqualified."""
    src = _na(_counter(), "(not (= s0 15))")
    head, _, tail = src.partition("namespace Definition")
    assert "abbrev StateType := (n : Nat) → TypeMap n" in head
    assert "abbrev M : Cslib.Automata.NA StateType (Unit × Unit) := {" in tail
    assert "start := fun s => Definition.INIT s," in tail
    assert "Tr := fun s _ s' => Definition.TRANS s s'" in tail
    assert tail.index("end Definition") < tail.index("abbrev M")


def test_imports_are_only_what_M_needs():
    """The driver pre-builds exactly this list, so it stays minimal.

    Everything below `namespace Definition` is core Lean and the LTL
    vocabulary belongs to the certificate, which imports it itself — so
    neither `Mathlib` nor `LTLCertifying.*` belongs here, and nothing from
    the generated project can, since the file is elaborated inside
    lean-ltl-certifying.
    """
    src = _na(_counter(), "(not (= s0 15))")
    imports = [l for l in src.splitlines() if l.startswith("import ")]
    assert imports == ["import Cslib.Computability.Automata.NA.Basic"]


def test_the_prebuild_targets_are_the_models_imports():
    """Drift between the two is what leaves `lean2vmt` unable to elaborate."""
    from zrth.lean.fbk_proveit import _LAKE_TARGETS
    from zrth.lean.translate.fbk import NA_IMPORTS

    assert _LAKE_TARGETS == NA_IMPORTS


def test_binders_are_explicit_so_INIT_and_TRANS_keep_their_arity():
    """`M` applies them to a fixed number of arguments.

    Left to `variable`, a `TRANS` whose every `effect_i` is constant would
    auto-bind `statenext` alone and `Definition.TRANS s s'` would not
    elaborate.
    """
    src = _na(_counter(), "(not (= s0 15))")
    assert "abbrev TRANS (state statenext : StateType) : Bool :=" in src
    assert "abbrev INIT (state : StateType) : Bool :=" in src
    assert "abbrev PROPERTY (state : StateType) : Bool :=" in src
    assert "\nvariable " not in src


def test_typemap_is_a_single_wildcard_when_the_state_is_uniform():
    assert "abbrev TypeMap : Nat → Type\n  | _ => Int" in _na(_counter(), "(= s0 0)")
    assert "abbrev TypeMap : Nat → Type\n  | _ => Bool" in _na(_two_bools(), "(not s0)")


def test_typemap_has_an_arm_per_slot_when_the_state_is_mixed():
    src = _na(_mixed_state(), "(>= s0 0)")
    assert "abbrev TypeMap : Nat → Type\n  | 0 => Int\n  | 1 => Bool\n  | _ => Bool" in src


def test_var_abbrevs_are_ascribed_their_own_slot_type():
    """`matchTypeName` reads the ascription to pick the VMT sort."""
    src = _na(_mixed_state(), "(>= s0 0)")
    assert "abbrev var_0 (state : StateType) : Int := state 0" in src
    assert "abbrev var_1 (state : StateType) : Bool := state 1" in src


# ── the property, in Bool ───────────────────────────────────────────────────


def test_property_is_bool_valued_not_prop_valued():
    """`decide` over a conjunction hands lean2vmt `instDecidableAnd`."""
    src = _na(_counter(), "(and (<= 0 s0) (<= s0 10))")
    line = [l for l in src.splitlines() if l.strip().startswith("((decide")][0]
    assert "&&" in line and " ∧ " not in line
    assert line.count("decide") == 2


def test_ge_and_gt_are_normalised_to_le_and_lt():
    """Only `Int.decLe`/`Int.decLt` are instances lean2vmt recognises."""
    src = _na(_counter(), "(and (>= s0 0) (> s0 3))")
    assert "(decide ((0 : Int) ≤ (var_0 state)))" in src
    assert "(decide ((3 : Int) < (var_0 state)))" in src


def test_implication_becomes_or_not():
    src = _na(_counter(), "(=> (<= 0 s0) (<= s0 10))")
    prop = src.split("abbrev PROPERTY (state : StateType) : Bool :=", 1)[1]
    prop = prop.splitlines()[1]
    assert "||" in prop
    assert "→" not in prop


def test_property_outside_the_fragment_aborts():
    with pytest.raises(ProveItError, match="cannot encode --safety"):
        _na(_counter(), "(> (div s0 2) 3)")


def test_property_must_be_boolean():
    with pytest.raises(ProveItError, match="sort Bool"):
        _na(_counter(), "(+ s0 1)")


# ── what the route refuses ──────────────────────────────────────────────────


def test_external_inputs_abort():
    with pytest.raises(NAUnsupported, match="external input"):
        check_na_supported(LeanContext(_with_extl()))


def test_real_state_aborts():
    with pytest.raises(NAUnsupported, match="Real"):
        check_na_supported(LeanContext(_real_state()))


def test_mixed_element_types_are_accepted():
    """Verified against the real lean2vmt: a per-index `TypeMap` declares
    `var_0 Bool` / `var_1 Int` correctly, and the `TypeMap.match_1` the
    equation compiler generates never reaches the VMT — `emitDefs` keeps
    only declarations whose return type whnfs to Prop/Int/Bool."""
    check_na_supported(LeanContext(_mixed_state()))


def test_ne_is_accepted_now_that_lean2vmt_translates_it():
    """`decide (a ≠ b)` hands `exprToSMT` the *instance*
    `instDecidableNot (Int.decEq a b)`, which it used to print as that bare
    word. Verified against the real translator: it now yields
    `(not (= var_0 7))`."""
    check_na_supported(LeanContext(_uses_ne()))


def _xor_module() -> Module:
    b = Var(Bool([1, 1]))
    t = Wire(Bool([1, 1]))
    init = [Term(LIA.Bool(torch.tensor([[False]])), [X(b)])]
    update = [
        Term(LIA.Bool(torch.tensor([[True]])), [t]),
        Term(LIA.Xor(), [X(b)], [b, t]),
    ]
    return Module.sequential([b], init, update)


def test_an_op_with_no_scalar_lean_form_is_still_encoded():
    """`Xor` has no `_SCALAR_OP` entry and used to be refused for it.

    The transition no longer goes through the scalar *Lean* printer, so
    "has a scalar Lean form" stopped being the question: `smt_encode` gives
    the term and `smt_to_lean_bool` prints `Xor` as `!(a == b)`, which is
    inside the fragment lean2vmt reads.
    """
    check_na_supported(LeanContext(_xor_module()))
    src = _na(_xor_module(), "true")
    assert "!(" in src and "==" in src


def test_an_op_the_encoder_has_no_term_for_aborts():
    """The boundary is now `smt_encode` and the Bool printer, and it is the
    *reason* they give that reaches the user -- `Transpose` has no SMT
    term at all, so no VMT model can be built from it."""
    v = Var(Int([1, 3]))
    t = Wire(Int([3, 1]))
    init = [Term(LIA.Int(torch.tensor([[0, 0, 0]])), [X(v)])]
    update = [
        Term(LIA.Transpose(), [t], [v]),
        Term(LIA.Transpose(), [X(v)], [t]),
    ]
    with pytest.raises(NAUnsupported, match="Transpose"):
        check_na_supported(LeanContext(Module.sequential([v], init, update)))


def test_a_non_scalar_ctrl_wire_becomes_one_slot_per_element():
    """The restriction this encoding used to carry.

    `R_i` compares `var_i statenext` against `effect_i`, and a tuple has no
    comparison lean2vmt translates -- so a wire holding three elements gets
    three slots rather than one, and every comparison stays scalar.
    """
    v = Var(Int([1, 3]))
    init = [Term(LIA.Int(torch.tensor([[0, 1, 2]])), [X(v)])]
    update = [Term(LIA.Id(), [X(v)], [v])]
    module = Module.sequential([v], init, update)
    check_na_supported(LeanContext(module))

    src = _na(module, "(<= ((_ tuple.select 1) s0) 10)")
    for k in range(3):
        assert f"abbrev var_{k} (state : StateType) : Int := state {k}" in src
        assert f"abbrev R_{k} (state statenext : StateType) : Bool :=" in src
        assert f"var_{k} statenext == effect_{k}" in src
    assert "abbrev var_3" not in src
    assert "TRANS" in src and "R_0 state statenext &&" in src
    # init is a literal per slot, not a tuple
    assert "abbrev init_1 : Int :=\n  (1 : Int)" in src
    # and the property's tuple selector resolves to that element's slot
    assert "(var_1 state)" in src


def test_an_element_of_a_wide_wire_reads_its_own_slot():
    """The property and the transition have to agree about which slot an
    element lives in; both take the layout from `_slot_accessors`."""
    v = Var(Int([1, 3]))
    init = [Term(LIA.Int(torch.tensor([[0, 0, 0]])), [X(v)])]
    update = [Term(LIA.Id(), [X(v)], [v])]
    module = Module.sequential([v], init, update)
    src = _na(module, "(= ((_ tuple.select 2) s0) 7)")
    assert "abbrev PROPERTY (state : StateType) : Bool :=\n  ((var_2 state) == (7 : Int))" in src


# ── the emitted transition means what the module means ──────────────────────
#
# The slot bodies come from `smt_encode` and are printed by
# `smt_to_lean_bool`, so the arithmetic is not re-implemented here -- but the
# *flattening* is: which slot an element lands in, and which element of a
# wire's next value a slot's `effect_k` computes. Getting that wrong produces
# a model that elaborates, model-checks, and describes a different system.
# So the emitted text is evaluated against the module's own execution.


def _tokens(src: str) -> list:
    return re.findall(r"\(|\)|[^\s()]+", src)


def _eval_lean(src: str, slots: list):
    """Evaluate the fragment `smt_to_lean_bool` emits, on concrete slots."""
    toks = _tokens(src)
    pos = 0

    def peek(k=0):
        return toks[pos + k] if pos + k < len(toks) else None

    def take(expected=None):
        nonlocal pos
        t = toks[pos]
        assert expected is None or t == expected, f"expected {expected}, got {t}"
        pos += 1
        return t

    def atom():
        nonlocal pos
        t = peek()
        if t == "(":
            return parens()
        take()
        if t == "true":
            return True
        if t == "false":
            return False
        if re.fullmatch(r"-?\d+", t):
            return int(t)
        raise AssertionError(f"unexpected token {t!r} in {src!r}")

    def parens():
        take("(")
        if peek() == "if":
            take("if")
            cond = atom()
            take("then")
            a = atom()
            take("else")
            b = atom()
            take(")")
            return a if cond else b
        if peek() == "decide":
            take("decide")
            v = parens()
            take(")")
            return v
        if peek() == "!":
            take("!")
            v = atom()
            take(")")
            return not v
        if peek() == "-" :
            take("-")
            v = atom()
            take(")")
            return -v
        if peek() is not None and re.fullmatch(r"var_\d+", peek()):
            k = int(take()[4:])
            take("state")
            take(")")
            return slots[k]
        left = atom()
        while peek() != ")":
            op = take()
            if op == ":":            # `(7 : Int)` -- an ascription, not an op
                take()               # the type
                continue
            right = atom()
            left = {
                "+": lambda a, b: a + b,
                "-": lambda a, b: a - b,
                "*": lambda a, b: a * b,
                "&&": lambda a, b: a and b,
                "||": lambda a, b: a or b,
                "==": lambda a, b: a == b,
                "<": lambda a, b: a < b,
                "<=": lambda a, b: a <= b,
                "≤": lambda a, b: a <= b,
                ">": lambda a, b: a > b,
                ">=": lambda a, b: a >= b,
                "≥": lambda a, b: a >= b,
                "=": lambda a, b: a == b,
            }[op](left, right)
        take(")")
        return left

    value = atom()
    assert pos == len(toks), f"trailing tokens in {src!r}"
    return value


def _effect_bodies(src: str) -> list[str]:
    """The `effect_k` body lines of an emitted model, in slot order."""
    out = {}
    lines = src.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"abbrev effect_(\d+)", line)
        if m:
            out[int(m.group(1))] = lines[i + 1].strip()
    return [out[k] for k in sorted(out)]


def _reference_step(module, values: list) -> list:
    """Run one `update` of the module on `values`, flat element order."""
    from zrth.eval import execute_update

    state = {}
    k = 0
    for var in module.ctrl:
        shape = dtype_shape(var.dtype)
        rows, cols = (1, shape[0]) if len(shape) == 1 else tuple(shape)
        elems = values[k : k + rows * cols]
        k += rows * cols
        is_bool = isinstance(var.dtype, Bool)
        state[var] = torch.tensor(
            [[elems[r * cols + c] for c in range(cols)] for r in range(rows)],
            dtype=torch.bool if is_bool else torch.int64,
        )
    execute_update(state, module.atoms)
    out = []
    for var in module.ctrl:
        t = state[X(var)]
        out.extend(v.item() for v in t.flatten())
    return out


@pytest.mark.parametrize(
    "module_fn,values",
    [
        (_counter, [[0], [4], [9], [10], [-3]]),
        (_two_bools, [[True, False], [False, True], [True, True]]),
        (_wide_vec, [[0, 1, 2], [5, -5, 100], [7, 7, 7]]),
        (_wide_mixed, [[True, 0, 1, 2], [False, 3, 4, 5]]),
        (_wide_matrix, [[0, 1, 2, 3], [7, -7, 0, 9]]),
    ],
)
def test_the_emitted_transition_agrees_with_the_module(module_fn, values):
    module = module_fn()
    bodies = _effect_bodies(_na(module, "true"))
    for state in values:
        expected = _reference_step(module, state)
        got = [_eval_lean(b, state) for b in bodies]
        assert got == expected, f"state {state}: {got} != {expected}"


def test_simplification_can_be_turned_off_without_changing_the_meaning():
    """`--fbk-simplify none` is a spelling, not a semantics.

    The module says "if x + 1 = 10 then 0 else x + 1". cvc5's rewriter
    rebalances that into "if x = 9 then 0 else 1 + x" -- the same
    transition, and no longer the one you can read against the source.
    Both must agree with the module on every state.
    """
    module = _counter()
    folded = _effect_bodies(_na(module, "true"))
    raw = _effect_bodies(
        atom_to_lean_na(
            LeanContext(module), "true", module_name="T", simplify=False
        )
    )
    for state in ([0], [4], [9], [10], [-3]):
        expected = _reference_step(module, state)
        assert [_eval_lean(b, state) for b in folded] == expected
        assert [_eval_lean(b, state) for b in raw] == expected

    assert "(9 : Int)" in folded[0], "the rewriter folds the +1 into the test"
    assert "(10 : Int)" in raw[0], "unsimplified, the module's own 10 survives"
    assert "((var_0 state) + (1 : Int))" in raw[0]


# ── the proof that the model is the module ──────────────────────────────────


def _bridge(module: Module) -> str:
    from zrth.lean.translate.fbk_bridge import atom_to_lean_fbk_bridge

    return atom_to_lean_fbk_bridge(LeanContext(module), na_module="TNA")


def test_the_bridge_states_one_obligation_per_slot():
    """`link_k` is structural and `bridge_k` semantic, and there is one of
    each per *element* -- the same layout the model uses."""
    src = _bridge(_wide_mixed())          # 1 Bool slot + a 3-wide Int wire
    for k in range(4):
        assert f"theorem link_{k} (state : StateType)" in src
        assert f"theorem bridge_{k} " in src
        assert f"abbrev effect_{k}_fn " in src
    assert "theorem link_4" not in src
    assert "theorem bridge_4" not in src


def test_the_bridge_reads_slots_as_variables_not_through_the_state():
    """`state k` has type `TypeMap k`, a stuck match that `omega` will not
    read as an `Int`. The semantic obligation has to be over plain binders,
    which is why `effect_k_fn` exists at all."""
    src = _bridge(_counter())
    body = src.split("abbrev effect_0_fn")[1].split("theorem")[0]
    assert "var_0 state" not in body, f"the model's state leaked into: {body}"
    assert "(x0 : Int)" in src.split("abbrev effect_0_fn")[1].split(":=")[0]
    # and the two are tied by `rfl`, which is where the TypeMap is crossed
    assert "effect_0_fn (Definition.var_0 state) := rfl" in src


def test_the_bridge_targets_the_scalar_encoding():
    """`Scalar.update` is the flat-slot transition function, and
    `update_scalar_eq` already ties it to the functional encoding -- so the
    solver never sees a matrix."""
    src = _bridge(_counter())
    assert "Scalar.update (x0) () ()" in src
    assert "update_scalar_eq" in src
    assert "init_scalar_eq" in src


def test_the_bridge_proves_the_property_twice_translated():
    """`PROPERTY` is `smt_to_lean_bool`'s reading of the formula and `P` is
    `smt_predicates_to_lean`'s. Without tying them the chain would be sound
    about the transition and silent about what is proved of it."""
    src = _bridge(_counter())
    assert "theorem PROPERTY_iff" in src
    assert "Definition.PROPERTY (toSlots s) = true ↔ P s" in src
    assert "theorem module_safety" in src


def test_the_bridges_fallback_arm_can_reduce():
    """`TypeMap` is stuck at a variable index, so a bare `_` arm does not
    typecheck -- only the successor pattern lets it reach the fallback."""
    src = _bridge(_wide_mixed())
    assert "| _ + 4 =>" in src
    assert "\n  | _ =>" not in src


def test_the_bridge_carries_the_budgets():
    """A 64-deep straight-line transition exhausts the default recursion
    depth inside `simp` before any prover sees the goal."""
    src = _bridge(_counter())
    assert "set_option maxRecDepth" in src
    assert "set_option maxHeartbeats" in src


def test_ic3ia_directory_resolves_to_the_binary_inside_it():
    """The build directory is the obvious thing to reach for, and
    `proveit.py` reports it as "executable not found" with the binary
    sitting inside."""
    from zrth.lean.fbk_proveit import resolve_ic3ia

    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "ic3ia"
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        assert resolve_ic3ia(tmp) == str(binary)
        assert resolve_ic3ia(tmp + "/") == str(binary)
        assert resolve_ic3ia(str(binary)) == str(binary)


def test_ic3ia_bad_paths_say_which_way_they_are_bad():
    from zrth.lean.fbk_proveit import resolve_ic3ia

    assert resolve_ic3ia(None) is None
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ProveItError, match="is a directory and holds no"):
            resolve_ic3ia(tmp)
        dud = Path(tmp) / "ic3ia"
        dud.write_text("not executable")
        dud.chmod(0o644)
        with pytest.raises(ProveItError, match="not executable"):
            resolve_ic3ia(str(dud))
        with pytest.raises(ProveItError, match="no such file"):
            resolve_ic3ia(str(Path(tmp) / "nope"))


def test_resolve_project_rejects_a_directory_that_is_not_the_checkout():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ProveItError, match="does not look like"):
            resolve_project(tmp)


# ── CLI wiring ──────────────────────────────────────────────────────────────


def _verith(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["uv", "run", "verith", *args],
        capture_output=True,
        text=True,
        cwd=PKG_ROOT,
    )


@pytest.mark.parametrize(
    "extra,expected",
    [
        ([], "--fbk-proveit requires --safety"),
        # ic3ia decides `G P`; a Buchi property is a different question.
        (["--buchi", "(= s0 0)"], "incompatible with --buchi"),
        (["--safety", "(= s0 0)", "--infer"], "incompatible with --infer"),
        (["--safety", "(= s0 0)", "--invariant", "(= s0 0)"], "incompatible with --invariant"),
        (["--safety", "(= s0 0)", "--ranking", "s0"], "is meaningless with --safety"),
        (["--safety", "(= s0 0)", "--pre", "true"], "incompatible with --pre"),
    ],
)
def test_fbk_proveit_rejects_what_it_would_have_to_ignore(extra, expected, tmp_path):
    r = _verith(
        str(FIXTURE_DIR / "counter.py"),
        "-o", str(tmp_path), "-p", "P",
        "--fbk-proveit", str(tmp_path),
        *extra,
    )
    assert r.returncode != 0
    assert expected in r.stderr


@pytest.mark.parametrize(
    "flag,reported",
    [
        ("--fbk-proveit", "--fbk-proveit"),
        ("--ic3ia", "--ic3ia"),
        ("--safety", "--safety"),
    ],
)
def test_an_empty_value_is_not_an_absent_flag(flag, reported, tmp_path):
    """`--fbk-proveit "$LTL"` with `LTL` unset used to read as "not passed".

    Every check gating the route is a truthiness test, so an empty path
    dropped the route *and* the checks that would have reported it: verith
    generated an ordinary project and exited 0, having silently certified
    nothing.  The blank is the error.
    """
    out = tmp_path / "out"
    r = _verith(
        str(FIXTURE_DIR / "counter.py"),
        "--safety", "(= s0 0)", "-o", str(out), "-p", "P",
        "--fbk-proveit", str(tmp_path),
        flag, "",
    )
    assert r.returncode != 0
    assert f"{reported} was given an empty value" in r.stderr
    assert not out.exists()


def test_ic3ia_alone_is_an_error(tmp_path):
    r = _verith(
        str(FIXTURE_DIR / "counter.py"),
        "--safety", "(= s0 0)", "-o", str(tmp_path), "-p", "P",
        "--ic3ia", "/bin/true",
    )
    assert r.returncode != 0
    assert "only meaningful together with --fbk-proveit" in r.stderr



# ── the certificate as a lake target ───────────────────────────────────────


def test_the_proveit_lakefile_can_see_what_the_certificate_imports():
    """`Certificate/Certificate.lean` is root-imported, so `lake build`
    reaches it. It imports `LTLCertifying.*`, `Smt` and the NA model, and a
    lakefile missing any of the three turns that reach into an error."""
    from zrth.lean.project import generate_lakefile, na_module_name

    bare = generate_lakefile("Proj")
    assert "LTL_Certifying" not in bare
    assert "srcDir" not in bare

    wired = generate_lakefile("Proj", ltl_project="/checkout")
    assert 'name = "LTL_Certifying"' in wired
    assert 'path = "/checkout"' in wired
    # The NA lib maps the model's module name onto the directory it is
    # written to; `import ProjNA` in the certificate resolves through it.
    assert f'name = "{na_module_name("Proj")}"' in wired
    assert 'srcDir = "ProveIt"' in wired
    # `Smt` was already required for verith's own route.
    assert 'name = "smt"' in wired


def test_the_na_lib_names_the_file_the_model_is_written_to(tmp_path):
    """The lean_lib's name and srcDir are the model's module name and
    directory. Drift between them is a `lake build` that cannot find an
    import the certificate already carries."""
    from zrth.lean.project import PROVEIT_DIR, na_module_name
    from zrth.lean.fbk_proveit import write_na_model

    model = write_na_model(
        tmp_path, "Proj", LeanContext(_counter()), "true"
    )
    assert model.parent.name == PROVEIT_DIR
    assert model.stem == na_module_name("Proj")


def test_a_checkout_that_is_not_one_fails_before_the_project_exists(tmp_path):
    """The lakefile names the checkout, so it is resolved before anything is
    generated — otherwise a typo leaves a project whose lakefile points at
    nothing."""
    out = tmp_path / "out"
    r = _verith(
        str(FIXTURE_DIR / "counter.py"),
        "--safety", "(= s0 0)", "-o", str(out), "-p", "P",
        "--fbk-proveit", str(tmp_path / "nowhere"),
    )
    assert r.returncode != 0
    assert "--fbk-proveit: not a directory" in r.stderr
    assert not out.exists()
