"""Coverage of the op tables that drive Lean emission.

The Lean emitters dispatch on the *stripped* variant name that ``itype_name``
returns (``LIA_Ne`` -> ``Ne``). A key that does not match a real variant is
silently dead: the operator it was meant to serve falls through to the "no
mapping" path instead. These tests pin the tables to the variant names the
theories actually expose.
"""

import torch
import pytest
from zrth import Wire, Term, Module, Bool, Int, LIA, LRA, BV, Var, X
from zrth.lean import ModuleToLean4
from zrth.lean.common import itype_name
from zrth.lean.native import _LEAN_OP, _SCALAR_OP
from zrth.lean.circ import _LEAN_OP_BOX

# Keys left over from the pre-`Sort` era, when ops lived in a flat `IType`
# namespace. They dispatch on names no theory defines any more, so each is
# dead code awaiting either a real variant or removal. Listed explicitly so
# that a *newly* introduced dead key still fails the test below.
_KNOWN_DEAD = {
    "_LEAN_OP": {"Mod", "TensorGet", "ToUnsigned"},
    "_SCALAR_OP": {"Mod", "TensorGet", "ToUnsigned"},
    "_LEAN_OP_BOX": {"MatAdd"},
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
