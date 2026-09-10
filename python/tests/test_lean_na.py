"""The NA encoding and the `--fbk-proveit` route (`zrth.lean.translate.na`).

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
from zrth.lean.common import LeanContext
from zrth.lean.fbk_proveit import ProveItError, property_to_bool_lean, resolve_project
from zrth.lean.translate.na import NAUnsupported, atom_to_lean_na, check_na_supported

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


def test_file_is_self_contained():
    """It is elaborated inside lean-ltl-certifying, not the generated project."""
    src = _na(_counter(), "(not (= s0 15))")
    imports = [l for l in src.splitlines() if l.startswith("import ")]
    assert imports == [
        "import LTLCertifying.Safety.Lemmas",
        "import Mathlib",
        "import Mathlib.Logic.Basic",
        "import Cslib.Computability.Automata.NA.Basic",
    ]


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


def test_typemap_is_the_uniform_form():
    assert "abbrev TypeMap : Nat → Type\n  | _ => Int" in _na(_counter(), "(= s0 0)")
    assert "abbrev TypeMap : Nat → Type\n  | _ => Bool" in _na(_two_bools(), "(not s0)")


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
    with pytest.raises(ProveItError, match="cannot encode --property"):
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


def test_mixed_element_types_abort():
    with pytest.raises(NAUnsupported, match="mixes element types"):
        check_na_supported(LeanContext(_mixed_state()))


def test_op_without_a_lean2vmt_translation_aborts():
    with pytest.raises(NAUnsupported, match="Ne"):
        check_na_supported(LeanContext(_uses_ne()))


def test_non_scalar_ctrl_wire_aborts():
    v = Var(Int([1, 3]))
    init = [Term(LIA.Int(torch.tensor([[0, 0, 0]])), [X(v)])]
    update = [Term(LIA.Id(), [X(v)], [v])]
    with pytest.raises(NAUnsupported, match="more than one element"):
        check_na_supported(LeanContext(Module.sequential([v], init, update)))


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
        ([], "--fbk-proveit requires --property"),
        (["-P", "(= s0 0)", "--infer"], "incompatible with --infer"),
        (["-P", "(= s0 0)", "--invariant", "(= s0 0)"], "incompatible with --invariant"),
        (["-P", "(= s0 0)", "--ranking", "s0"], "incompatible with --ranking"),
        (["-P", "(= s0 0)", "--pre", "true"], "incompatible with --pre"),
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


def test_ic3ia_alone_is_an_error(tmp_path):
    r = _verith(
        str(FIXTURE_DIR / "counter.py"),
        "-P", "(= s0 0)", "-o", str(tmp_path), "-p", "P",
        "--ic3ia", "/bin/true",
    )
    assert r.returncode != 0
    assert "only meaningful together with --fbk-proveit" in r.stderr
