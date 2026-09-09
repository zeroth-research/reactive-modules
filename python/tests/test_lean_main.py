"""`generate_main_lean` must emit a Main.lean that matches what it calls.

The executable project (`verith -x`) is generated from two places that have
to agree: `atom_to_lean_functional` emits `init`/`update`, and
`generate_main_lean` emits the IO loop that drives them. Nothing checked
that agreement, so the loop drifted from the signatures it calls.
"""

import torch
import pytest

from zrth import Term, Module, Int, Bool, BitVec, Real, LIA, LRA, BV, Var, X
from zrth.lean.project import generate_main_lean
from zrth.lean.translate import ModuleToLean4


def _module_with_extl():
    """State plus a genuine external input, so extl_l/extl_n are non-trivial."""
    s = Var(Int([1, 1]))
    e = Var(Int([1, 1]))
    init = [Term(LIA.Linear(torch.tensor([[1]]), torch.tensor([[0]])), [X(s)], [X(e)])]
    update = [Term(LIA.Add(), [X(s)], [s, X(e)])]
    return Module.sequential([s, e], init, update)


def _module_without_extl():
    s = Var(Int([1, 1]))
    return Module.sequential(
        [s],
        [Term(LIA.Int(torch.tensor([[0]])), [X(s)])],
        [Term(LIA.Id(), [X(s)], [s])],
    )


def _two_ctrl_module():
    """Two ctrl components, to expose ordering in the show/parse helpers."""
    a = Var(Int([1, 1]))
    b = Var(Int([1, 1]))
    init = [
        Term(LIA.Int(torch.tensor([[1]])), [X(a)]),
        Term(LIA.Int(torch.tensor([[2]])), [X(b)]),
    ]
    update = [Term(LIA.Id(), [X(a)], [a]), Term(LIA.Id(), [X(b)], [b])]
    return Module.sequential([a, b], init, update)


@pytest.mark.parametrize(
    "make", [_module_with_extl, _module_without_extl, _two_ctrl_module]
)
def test_main_applies_update_curried(make):
    """`update` takes (ctrl) (extl_l) (extl_n) — three groups, not one tuple."""
    module = make()
    functional = ModuleToLean4(module).to_lean_functional()
    assert "def update (ctrl:" in functional, "signature shape changed; update this test"

    main = generate_main_lean("Rea", module, "Rea")
    assert "update state extlPrev extl" in main, (
        "Main.lean must apply update's three parameter groups separately"
    )
    assert "update (state" not in main, "update is called with a merged tuple"


@pytest.mark.parametrize(
    "make", [_module_with_extl, _module_without_extl, _two_ctrl_module]
)
def test_main_threads_the_previous_external_input(make):
    """`extl_l` is the previous step's input, so the loop has to carry it.

    It previously had no source at all in `main`.
    """
    main = generate_main_lean("Rea", make(), "Rea")
    assert "let mut extlPrev := extl0" in main, "extl_l is never initialised"
    assert "extlPrev := extl" in main, "extl_l is never advanced"


# ──────────────────────────────────────────────────────────────
# Component order: the literals must match `_product_type`
# ──────────────────────────────────────────────────────────────


def _heterogeneous_module():
    """Bool then Int ctrl, so a swapped order is a type error, not a silent one."""
    b = Var(Bool([1, 1]))
    n = Var(Int([1, 1]))
    init = [
        Term(LIA.Bool(torch.tensor([[True]])), [X(b)]),
        Term(LIA.Int(torch.tensor([[7]])), [X(n)]),
    ]
    update = [Term(LIA.Id(), [X(b)], [b]), Term(LIA.Id(), [X(n)], [n])]
    return Module.sequential([b, n], init, update)


def test_show_ctrl_destructures_in_declaration_order():
    """`v{i}` must bind ctrl[i]: the formatter per slot is chosen from its type.

    `_product_type` and `_accessor` both order components as declared, but
    the pattern was built reversed, so every slot got the wrong formatter.
    """
    main = generate_main_lean("Rea", _heterogeneous_module(), "Rea")
    assert "let (v0, v1) := v" in main
    assert "let (v1, v0) := v" not in main, "destructuring is reversed"
    # and the ctrl type ascription is in the same order
    assert "showCtrl (v : (Mat Bool 1 1) × (Mat Int 1 1))" in main


def test_parse_extl_builds_the_tuple_in_declaration_order():
    """Same for the parsed input tuple, against `parseExtl`'s return type."""
    e0 = Var(Bool([1, 1]))
    e1 = Var(Int([1, 1]))
    s = Var(Int([1, 1]))
    init = [Term(LIA.Int(torch.tensor([[0]])), [X(s)])]
    update = [Term(LIA.Ite(), [X(s)], [X(e0), X(e1), s])]
    module = Module.sequential([s, e0, e1], init, update)

    main = generate_main_lean("Rea", module, "Rea")
    assert "pure (e0, e1)" in main
    assert "pure (e1, e0)" not in main, "parsed tuple is reversed"


def test_single_component_needs_no_tuple():
    """One wire: the value itself, matching `_accessor`'s total==1 case."""
    main = generate_main_lean("Rea", _module_without_extl(), "Rea")
    assert "let v0 := v" in main
    assert "let (v0) := v" not in main
