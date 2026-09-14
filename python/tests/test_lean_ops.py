"""The op table: every variant covered, every gap named.

`zrth/lean/ops.py` holds one row per op variant and one column per backend
(`mat`, `scalar`, `box`, `smt`). These tests are the other half of that: they
check the table against what `LIA`/`LRA`/`BV` actually expose, in both
directions.

The direction that is new here is *missing*. Six separate dicts could be
checked for keys no theory defines — a dead key silently disables the op it
was meant to serve — but nothing could see a variant with no emitter at all,
because there was no one place that knew which variants there are. That is
how every unsigned and signed BV comparison came to be encoded for cvc5 and
for no Lean backend: a module using one passed `--pre-check` and
`--infer ai-cegar`, then failed at project generation.

So: a variant added to a theory fails `test_every_variant_has_a_row` until
someone writes its row, and writing that row means saying what all four
backends do with it — an emitter, or the reason there is none.
"""

import torch
import pytest
from zrth import Wire, Term, Module, Bool, Int, BitVec, LIA, LRA, BV, Var, X
from zrth.lean import ModuleToLean4
from zrth.lean.common import itype_name
from zrth.lean import ops
from zrth.lean.ops import (
    COLUMNS,
    OPS,
    Inline,
    Op,
    Unsupported,
    ViaMat,
    box_for,
    cell_of,
    mat_emitter,
    op_for,
    scalar_emitter,
)

_THEORIES = {"LIA": LIA, "LRA": LRA, "BV": BV}


def _variant_names(theory) -> set[str]:
    """Every variant the theory exposes, as `itype_name` sees it."""
    return {n for n in dir(theory) if not n.startswith("_")}


def _row(theory: str, name: str) -> Op:
    return next(op for op in OPS if op.name == name and theory in op.theories)


# ══════════════════════════════════════════════════════════════════════
#  Completeness, both directions
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("theory_name", sorted(_THEORIES))
def test_every_variant_has_a_row(theory_name):
    """The guard the six tables could not give: no variant goes unanswered."""
    covered = {op.name for op in OPS if theory_name in op.theories}
    missing = _variant_names(_THEORIES[theory_name]) - covered
    assert not missing, (
        f"{theory_name} exposes {sorted(missing)} with no row in "
        "zrth/lean/ops.py. Add one, saying for each of "
        f"{list(COLUMNS)} either an emitter or `Unsupported(<why not>)`."
    )


def test_no_row_outlives_its_theory():
    """A row keyed to a variant no theory defines can never fire.

    Dispatch goes through `itype_name`, so such a row is dead code. This is
    the old per-table dead-key check, now covering all four columns at once.
    """
    for op in OPS:
        for theory_name in op.theories:
            assert op.name in _variant_names(_THEORIES[theory_name]), (
                f"ops.py has a row for {theory_name}.{op.name}, which that "
                "theory does not define"
            )


# `IType`-era names that survive as rows with no theory behind them. They
# keep their emitters because *removing* them is the open half of
# KNOWN_ISSUES #23; listing them here is what tells "dead" from "missing".
_KNOWN_DEAD = {"Tensor", "TensorGet", "ToUnsigned"}


def test_the_dead_rows_are_the_known_ones():
    assert {op.name for op in OPS if not op.theories} == _KNOWN_DEAD


def test_a_row_is_reachable_from_its_op():
    """`op_for` resolves a live op to its row, theory included."""
    assert op_for(LIA.Ne()) is _row("LIA", "Ne")
    assert op_for(BV.Ne()) is _row("BV", "Ne")
    assert op_for(LIA.Ne()) is not op_for(BV.Ne())


# ══════════════════════════════════════════════════════════════════════
#  What a cell may say
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("column", COLUMNS)
def test_every_cell_is_an_emitter_or_says_why(column):
    for op in OPS:
        cell = cell_of(op, column)
        if isinstance(cell, Unsupported):
            assert cell.reason.strip(), f"{op.name}.{column}: empty reason"
        elif isinstance(cell, Inline):
            assert cell.site.strip(), f"{op.name}.{column}: empty site"
        elif isinstance(cell, ViaMat):
            assert cell.note.strip(), f"{op.name}.{column}: empty note"
        elif column == "box":
            assert isinstance(cell, str) and cell.startswith("Box.")
        else:
            assert callable(cell), f"{op.name}.{column}: not an emitter"


def test_the_scalar_column_falls_back_only_where_there_is_something_to_fall_back_to():
    """`ViaMat` is a claim about the matrix cell, and `Unsupported` too.

    `_translate_terms_scalar` reaches for the matrix form whenever the
    scalar cell is not an emitter, so a scalar cell that gives up while the
    matrix one does not would be a lie: the matrix form would be used
    anyway. Either the matrix column has an emitter (`ViaMat`) or neither
    does (`Unsupported`).
    """
    for op in OPS:
        if isinstance(op.scalar, ViaMat):
            assert callable(op.mat), f"{op.name}: ViaMat with no matrix form"
        if isinstance(op.scalar, Unsupported):
            assert not callable(op.mat), (
                f"{op.name}: the scalar cell gives up but the matrix one "
                "does not, so the matrix form is what actually runs"
            )


def test_the_smt_column_is_emitters_and_refusals_only():
    """`smt_emitter` assumes it: `translate_terms` dispatches every variant
    through the table, constants and `Linear` included, so there is nothing
    for an `Inline` or `ViaMat` cell to mean there."""
    for op in OPS:
        assert callable(op.smt) or isinstance(op.smt, Unsupported)


# ══════════════════════════════════════════════════════════════════════
#  The gaps, pinned
# ══════════════════════════════════════════════════════════════════════

# Every (theory, variant, column) with no emitter. Closing one of these is a
# line removed here; opening a new one is a line added, which is the point —
# `ops.py` records *why* each is open, and this says how many there are.
_KNOWN_GAPS = {
    # Nondeterministic choice: nullary, and every encoding here is a
    # function of the read wires.
    *(
        (theory, name, column)
        for theory, name in (
            ("LIA", "AnyBool"), ("LIA", "AnyInt"),
            ("LRA", "AnyBool"), ("LRA", "AnyReal"),
            ("BV", "Havoc"),
        )
        for column in COLUMNS
    ),
    # No Lean form for an uninterpreted symbol, and no cvc5 one either
    # (KNOWN_ISSUES #27).
    *(
        (theory, "Uninterpreted", column)
        for theory in ("LIA", "LRA", "BV")
        for column in COLUMNS
    ),
    # Gradient-side ops, marked unstable in theory/src/lra.rs.
    *(
        ("LRA", name, column)
        for name in ("BoolZerograd", "RealZerograd")
        for column in COLUMNS
    ),
    # Encoded for cvc5, never for Lean — the divergence finding C measured.
    *(
        ("BV", name, column)
        for name in (
            "ULt", "ULe", "UGt", "UGe", "SLt", "SLe", "SGt", "SGe",
            "UDiv", "SDiv", "BVToBool",
        )
        for column in ("mat", "scalar", "box")
    ),
    # In no backend at all: width-changing, or open in the theory itself.
    *(
        ("BV", name, column)
        for name in ("BitSelect", "Extend", "Abs")
        for column in COLUMNS
    ),
    # `Xor` reaches Lean only through the NA route, which prints the cvc5
    # term rather than using a Lean emitter.
    *(("LIA", "Xor", c) for c in ("mat", "scalar", "box")),
    *(("LRA", "Xor", c) for c in ("mat", "scalar", "box")),
    # A transpose has no cvc5 term; `check_na_supported` refuses it by name.
    ("LIA", "Transpose", "smt"),
    ("LRA", "Transpose", "smt"),
    # Dead rows keep whatever they had.
    ("", "TensorGet", "box"),
    ("", "ToUnsigned", "box"),
}


def test_the_coverage_gaps_are_the_known_ones():
    gaps = {
        (theory, op.name, column)
        for theory, op in ops.rows()
        for column in COLUMNS
        if isinstance(cell_of(op, column), Unsupported)
    }
    assert gaps == _KNOWN_GAPS, (
        "the coverage matrix moved: "
        f"new gaps {sorted(gaps - _KNOWN_GAPS)}, "
        f"closed gaps {sorted(_KNOWN_GAPS - gaps)}. "
        "Run `python -m zrth.lean.ops` to see the whole matrix."
    )


def test_the_matrix_prints():
    """`python -m zrth.lean.ops` is how the table is meant to be read."""
    text = ops.format_matrix()
    assert "ULt" in text and "gaps:" in text
    for theory in _THEORIES:
        assert theory in text


# ══════════════════════════════════════════════════════════════════════
#  A gap, end to end
# ══════════════════════════════════════════════════════════════════════


def _bv_ult_module() -> Module:
    """The two-wire module finding C measured: cvc5 has `ULt`, Lean has not."""
    s = Var(BitVec(8, [1, 1]))
    flag = Var(BitVec(1, [1, 1]))
    init = [
        Term(BV.Const(torch.tensor([[3]])), [X(s)]),
        Term(BV.Const(torch.tensor([[0]])), [X(flag)]),
    ]
    update = [Term(BV.Id(), [X(s)], [s]), Term(BV.ULt(), [X(flag)], [s, s])]
    return Module.sequential([s, flag], init, update)


def test_an_op_only_cvc5_has_still_encodes_for_cvc5():
    import cvc5

    from zrth.lean.smt_module import ModuleSMT

    msmt = ModuleSMT(tm=cvc5.TermManager(), module=_bv_ult_module())
    state = msmt.update_state(
        msmt.fresh_ctrl(), msmt.fresh_extl_l(), msmt.fresh_extl_n()
    )
    assert "bvult" in str(state[1])


@pytest.mark.parametrize(
    "encoding", ["to_lean_functional", "to_lean_scalar", "to_lean_circ"]
)
def test_a_missing_lean_form_is_refused_with_the_recorded_reason(encoding):
    """The failure is the same as before; the message is not.

    "No Lean expression mapping for: ULt" said nothing about whether the op
    was forgotten, unimplementable, or handled somewhere else. The table's
    cell says which, and that reason is what reaches the caller.
    """
    m2l = ModuleToLean4(_bv_ult_module())
    with pytest.raises(ValueError) as excinfo:
        getattr(m2l, encoding)()
    message = str(excinfo.value)
    assert message.startswith("No Lean expression mapping for: ULt")
    assert "encoded for SMT, never for Lean" in message


def test_an_op_no_backend_has_names_its_own_reason():
    """`Uninterpreted` is the limit matrix's only generation failure."""
    with pytest.raises(ValueError, match="KNOWN_ISSUES #27"):
        mat_emitter(LIA.Uninterpreted("f"))


# ══════════════════════════════════════════════════════════════════════
#  Dispatch: the cases the six tables got wrong
# ══════════════════════════════════════════════════════════════════════


def test_ne_is_dispatchable():
    """`!=` is spelled `Ne` in every theory — it used to be keyed as `Neq`."""
    assert itype_name(LIA.Ne()) == "Ne"
    assert callable(mat_emitter(LIA.Ne()))
    assert callable(scalar_emitter(LIA.Ne()))
    assert box_for(LIA.Ne()) == "Box.neq"


def _make_ne_module():
    """Sequential module whose update compares two counters with `!=`."""
    x = Var(Int([1, 1]))
    y = Var(Int([1, 1]))
    differ = Var(Bool([1, 1]))

    init = [
        Term(LIA.Int(torch.tensor([[0]])), [X(x)]),
        Term(LIA.Int(torch.tensor([[1]])), [X(y)]),
        Term(LIA.Bool(torch.tensor([[False]])), [X(differ)]),
    ]
    update = [
        Term(LIA.Id(), [X(x)], [x]),
        Term(LIA.Id(), [X(y)], [y]),
        Term(LIA.Ne(), [X(differ)], [x, y]),
    ]
    return Module.sequential([x, y, differ], init, update)


def test_ne_emits_in_functional_encoding():
    lean = ModuleToLean4(_make_ne_module()).to_lean_functional()
    assert "≠" in lean, "Ne did not reach the functional encoding"


def test_ne_emits_in_scalar_encoding():
    lean = ModuleToLean4(_make_ne_module()).to_lean_scalar()
    assert "≠" in lean, "Ne did not reach the scalar encoding"


def test_ne_emits_in_circuit_encoding():
    lean = ModuleToLean4(_make_ne_module()).to_lean_circ()
    assert "Box.neq" in lean, "Ne did not reach the circuit (Box) encoding"


# ──────────────────────────────────────────────────────────────
# BV uses BitVec operators, never the Boolean ones
# ──────────────────────────────────────────────────────────────

# BV has no Bool wires at all: an `Ite` condition and an `Eq` result are both
# `BitVec 1` (the theory rejects a Bool there). So the Boolean forms cannot
# serve it, and a 1-bit condition is compared against 1 directly rather than
# converted through `BV.BVToBool`.
_BV_DIVERGENT = ["Not", "And", "Or", "Xor", "Ite", "Eq", "Ne"]


def test_bv_theory_has_no_bool_wires():
    """Pins the premise: BV's condition and comparison sorts are BitVec."""
    with pytest.raises(Exception):
        Term(BV.Ite(), [Wire(BitVec(8, [1, 1]))],
             [Wire(Bool([1, 1])), Wire(BitVec(8, [1, 1])), Wire(BitVec(8, [1, 1]))])
    with pytest.raises(Exception):
        Term(BV.Eq(), [Wire(Bool([1, 1]))],
             [Wire(BitVec(8, [1, 1])), Wire(BitVec(8, [1, 1]))])


@pytest.mark.parametrize("name", _BV_DIVERGENT)
def test_bv_gets_its_own_row_where_it_diverges(name):
    """Two rows, one per theory group, and no Lean cell shared between them.

    A single row keyed by name — which is what the tables were before the
    BV ones were split off — would emit `!`/`&&`/`||` against `BitVec 1`.
    """
    bv = _row("BV", name)
    if name == "Xor":
        # LIA/LRA have `Xor` with no Lean form at all; BV has all three.
        assert isinstance(_row("LIA", "Xor").mat, Unsupported)
    else:
        core = _row("LIA", name)
        assert bv is not core
        for column in ("mat", "scalar", "box"):
            assert cell_of(bv, column) is not cell_of(core, column)
    for column in ("mat", "scalar", "box"):
        assert not isinstance(cell_of(bv, column), Unsupported)
    assert box_for(getattr(BV, name)()).startswith("Box.bv")


def _twobit():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    try:
        import twobit
    finally:
        sys.path.pop(0)
    return twobit.module()


def test_bv_module_emits_bitvec_operators_not_boolean_ones():
    """`!`/`&&`/`||` and a bare `if c then` do not elaborate on `BitVec 1`."""
    lean = ModuleToLean4(_twobit()).to_lean_functional()
    assert "~~~" in lean, "BV Not did not use BitVec complement"
    assert "&&&" in lean, "BV And did not use BitVec conjunction"
    assert "= 1 then" in lean, "BV Ite did not compare its 1-bit condition"
    for boolean in (" && ", " || ", "=> !("):
        assert boolean not in lean, f"Boolean form {boolean!r} emitted for BV"
    assert "BVToBool" not in lean, "should not route through BVToBool"


def test_bv_module_emits_bitvec_boxes():
    lean = ModuleToLean4(_twobit()).to_lean_circ()
    assert "Box.bvNot" in lean and "Box.bvAnd" in lean and "Box.bvIte" in lean
    # Only the composed layers matter: `_LAYER_SIMP` lists every Box name as a
    # simp lemma regardless of which ones this module uses.
    layers = [l for l in lean.splitlines() if "⊗" in l or ("Box." in l and ":=" in l)]
    assert layers, "no composed layers emitted"
    for boolean in ("Box.not", "Box.and", "Box.or", "Box.ite", "Box.eq", "Box.neq"):
        for line in layers:
            assert boolean not in line, (
                f"Boolean box {boolean} emitted for BV in: {line.strip()[:90]}"
            )


@pytest.mark.parametrize("name", ["UMod", "SMod"])
def test_bv_modulo_is_dispatchable(name):
    """BV is the only theory with modulo, and it has two of them.

    The tables carried one `"Mod"` key, which no theory defines: LIA and LRA
    have no modulo at all, and BV spells its `UMod`/`SMod`.
    """
    assert getattr(BV, name, None) is not None, f"BV.{name} does not exist"
    op = getattr(BV, name)()
    assert callable(mat_emitter(op)) and callable(scalar_emitter(op))
    assert box_for(op) == f"Box.bv{name}"
    assert not any(op.name == "Mod" for op in OPS), "the dead integer Mod key is back"


def test_mat_add_is_not_a_separate_key():
    """`MatAdd` duplicated `Add`'s mapping under a name no theory defines."""
    assert not any(op.name == "MatAdd" for op in OPS)
    assert box_for(BV.Add()) == "Box.add"


def _bv_mod_module(op):
    """BV module whose update takes a modulo of its state."""
    s = Var(BitVec(8, [1, 1]))
    d = Var(BitVec(8, [1, 1]))
    init = [
        Term(BV.Const(torch.tensor([[7]])), [X(s)]),
        Term(BV.Const(torch.tensor([[3]])), [X(d)]),
    ]
    update = [Term(BV.Id(), [X(d)], [d]), Term(op, [X(s)], [s, d])]
    return Module.sequential([s, d], init, update)


@pytest.mark.parametrize(
    "op,expected", [(BV.UMod(), "BitVec.umod"), (BV.SMod(), "BitVec.smod")]
)
def test_bv_modulo_emits_the_right_lean(op, expected):
    lean = ModuleToLean4(_bv_mod_module(op)).to_lean_functional()
    assert expected in lean
