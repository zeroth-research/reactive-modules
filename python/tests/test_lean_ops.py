"""Coverage of the op tables that drive Lean emission.

The Lean emitters dispatch on the *stripped* variant name that ``itype_name``
returns (``LIA_Ne`` -> ``Ne``). A key that does not match a real variant is
silently dead: the operator it was meant to serve falls through to the "no
mapping" path instead. These tests pin the tables to the variant names the
theories actually expose.
"""

import torch
import pytest
from zrth import Wire, Term, Module, Bool, Int, BitVec, LIA, LRA, BV, Var, X
from zrth.lean import ModuleToLean4
from zrth.lean.common import itype_name
from zrth.lean.native import _LEAN_OP, _SCALAR_OP
from zrth.lean.circ import _LEAN_OP_BOX

# Keys left over from the pre-`Sort` era, when ops lived in a flat `IType`
# namespace. They dispatch on names no theory defines any more, so each is
# dead code awaiting either a real variant or removal. Listed explicitly so
# that a *newly* introduced dead key still fails the test below.
_KNOWN_DEAD = {
    # `TensorGet`/`ToUnsigned` are `IType`-era names with no modern
    # equivalent, so they need a decision (remove, or implement the ops)
    # rather than a rename. `Mod` and `MatAdd` are gone: BV's `UMod`/`SMod`
    # replaced the first, and `Add` already covered the second.
    "_LEAN_OP": {"TensorGet", "ToUnsigned"},
    "_SCALAR_OP": {"TensorGet", "ToUnsigned"},
    "_LEAN_OP_BOX": set(),
}

_TABLES = [
    ("_LEAN_OP", _LEAN_OP),
    ("_SCALAR_OP", _SCALAR_OP),
    ("_LEAN_OP_BOX", _LEAN_OP_BOX),
]


def _variant_names():
    """Every variant name the three theories expose, as itype_name sees it."""
    names = set()
    for theory in (LIA, LRA, BV):
        names |= {n for n in dir(theory) if not n.startswith("_")}
    return names


@pytest.mark.parametrize("table_name,table", _TABLES)
def test_op_table_has_no_new_dead_keys(table_name, table):
    """A key matching no theory variant can never fire; don't add more."""
    dead = set(table) - _variant_names() - _KNOWN_DEAD[table_name]
    assert not dead, (
        f"{table_name} has keys matching no theory variant: {sorted(dead)}. "
        "Dispatch goes through itype_name(), so these can never fire."
    )


@pytest.mark.parametrize("table_name,table", _TABLES)
def test_ne_is_dispatchable(table_name, table):
    """`!=` is spelled `Ne` in every theory — it used to be keyed as `Neq`."""
    assert itype_name(LIA.Ne()) == "Ne"
    assert "Ne" in table, f"{table_name} cannot emit Ne"


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
    from zrth import Wire, BitVec, Bool

    with pytest.raises(Exception):
        Term(BV.Ite(), [Wire(BitVec(8, [1, 1]))],
             [Wire(Bool([1, 1])), Wire(BitVec(8, [1, 1])), Wire(BitVec(8, [1, 1]))])
    with pytest.raises(Exception):
        Term(BV.Eq(), [Wire(Bool([1, 1]))],
             [Wire(BitVec(8, [1, 1])), Wire(BitVec(8, [1, 1]))])


@pytest.mark.parametrize("name", _BV_DIVERGENT)
def test_bv_overrides_the_boolean_form(name):
    """Each divergent op has a BitVec form in all three emitter tables."""
    from zrth.lean.native import _BV_LEAN_OP, _BV_SCALAR_OP
    from zrth.lean.circ import _BV_LEAN_OP_BOX

    assert name in _BV_LEAN_OP, f"{name} has no BitVec matrix form"
    assert name in _BV_SCALAR_OP, f"{name} has no BitVec scalar form"
    assert name in _BV_LEAN_OP_BOX, f"{name} has no BitVec Box"


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
    from zrth.lean.native import _BV_LEAN_OP, _BV_SCALAR_OP
    from zrth.lean.circ import _BV_LEAN_OP_BOX

    assert getattr(BV, name, None) is not None, f"BV.{name} does not exist"
    assert name in _BV_LEAN_OP and name in _BV_SCALAR_OP and name in _BV_LEAN_OP_BOX
    assert "Mod" not in _LEAN_OP and "Mod" not in _SCALAR_OP, (
        "the dead integer Mod key is back"
    )


def test_mat_add_is_not_a_separate_key():
    """`MatAdd` duplicated `Add`'s mapping under a name no theory defines."""
    assert "MatAdd" not in _LEAN_OP_BOX
    assert _LEAN_OP_BOX["Add"] == "Box.add"


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
