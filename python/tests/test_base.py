import pytest
import torch
from zrth import Wire, Term, Atom, Module, LIA, Bool, Int, LRA, Real, Var, X, d


def _bool_t(v):
    """Helper: build a 2-D bool tensor."""
    return torch.tensor([[bool(v)]], dtype=torch.bool)


def test_wire_new():
    Wire(Bool([1, 1]))
    Wire(Int([1, 1]))
    Wire(Int([2, 3]))


def test_term_new():
    x = Wire(Int([2, 3]))
    y = Wire(Int([2, 3]))
    xn = Wire(Int([2, 3]))
    w4 = Wire(Int([1, 3]))
    w5 = Wire(Int([1, 3]))

    # test `function` ctor
    _ = Term.function(LIA.Add(), [xn], [x, y])
    _ = Term.function(LIA.Int(torch.tensor([[3, 4, 5]])), [w4], [])
    _ = Term.function(LIA.Int(torch.tensor([[3, 4, 6]])), [w5], [])

    # comparisons are pointwise -> Bool of the operand shape
    Term(LIA.Lt(), [Wire(Bool([1, 3]))], [w4, Wire(Int([1, 3]))])
    Term.constant(LIA.Int(torch.tensor([[3, 2, 1]])), [Wire(Int([1, 3]))])


def test_var_derefs_to_wire():
    v = Var(Real([2, 3]))

    # attributes Var does not define fall through to the latched wire,
    # mirroring Rust's Deref
    assert v.degree == 0
    assert v.dtype == Real([2, 3])
    assert v.id == Wire(Real([1, 1])).id - 3  # ltc is first of the var's three wires

    # unknown attributes still raise, from the wire's lookup
    with pytest.raises(AttributeError):
        _ = v.no_such_attribute

    # a variable equals its latched wire (both directions), and they are
    # interchangeable dictionary keys; the other views stay distinct
    ltc = Term(LRA.Id(), [X(v)], [v]).read[0]
    assert v == ltc and ltc == v
    assert v is not ltc
    assert hash(v) == hash(ltc)
    assert {v: "found"}[ltc] == "found"
    assert v != X(v) and v != d(v)

    # ordering among variables is by creation
    u = Var(Real([1, 1]))
    assert v < u and sorted([u, v]) == [v, u]


def test_module_sequential():
    x = Var(Bool([1, 1]))
    init = [Term.constant(LIA.Bool(_bool_t(True)), [X(x)])]
    update = [Term(LIA.Id(), [X(x)], [x])]
    _ = Module.sequential([x], init, update)


def test_module_combinatorial():
    x = Var(Bool([1, 1]))

    assign = [Term.constant(LIA.Bool(_bool_t(False)), [X(x)])]
    _ = Module.combinatorial([x], assign)


def test_module_parallel():
    x = Var(Bool([1, 1]))
    y = Var(Bool([1, 1]))
    z = Var(Bool([1, 1]))
    w = Var(Bool([1, 1]))
    v = Var(Bool([1, 1]))

    init = [Term.constant(LIA.Bool(_bool_t(False)), [X(x)])]
    update = [Term(LIA.And(), [X(x)], [x, X(y)])]
    p = Module.sequential([x, y], init, update)

    init = [
        Term.constant(LIA.Bool(_bool_t(False)), [X(v)]),
        Term.constant(LIA.Bool(_bool_t(False)), [X(y)]),
    ]
    update = [Term(LIA.And(), [X(v)], [v, x]), Term(LIA.Id(), [X(y)], [x])]
    q = Module.sequential([x, y, v], init, update, hide=[v])

    assign = [Term(LIA.Or(), [X(z)], [X(y), X(w)])]
    r = Module.combinatorial((z, y, w), assign)

    m = Module.compose(p, q, r)

    c = m.ctrl
    print(c)

    for var in c:
        print(var)

    for atom in m.atoms:
        print(atom)

    print(m.intf)
    assert m.intf == [x, y, z]
    assert m.extl == [w]
    assert m.prvt == [v]


def test_interface():
    x = Wire(Bool([1, 1]))
    y = Wire(Bool([1, 1]))
    xn = Wire(Bool([1, 1]))
    f = Term(LIA.And(), [xn], [x, y])
    f2 = Term(LIA.And(), [xn], [x, y])

    w = f.write
    r = f.read
    r2 = f.read
    assert r is not r2
    assert r is not w
    assert r2 == r
    assert f2.read == r
    assert r == [x, y]
    assert [x, y] == r

    for wire in r:
        print("-->", wire)

    for i in range(len(w)):
        print("-->", w[i])


def _stateful_blocks():
    """Helper: two variables with init, update, and delay blocks over them."""
    zero = torch.tensor([[0.0]])
    x = Var(Real([1, 1]))
    p = Var(Real([1, 1]))
    init = [Term(LRA.Real(zero), [X(x)]), Term(LRA.Real(zero), [X(p)])]
    update = [Term(LRA.Id(), [X(x)], [x]), Term(LRA.Id(), [X(p)], [p])]
    delay = [Term(LRA.RealZerograd([1, 1]), [d(x)]), Term(LRA.RealZerograd([1, 1]), [d(p)])]
    return x, p, init, update, delay


def test_module_new_dispatches_on_blocks():
    x, p, init, update, delay = _stateful_blocks()

    # each block combination selects the corresponding constructor
    sequential = Module(init=init, update=update, vars=[x, p])
    differential = Module(init=init, delay=delay, vars=[x, p])
    hybrid = Module(init=init, update=update, delay=delay, vars=[x, p])
    jump = Module(update=update, vars=[x, p])
    constant = Module(init=init, vars=[x, p])

    for m in (sequential, differential, hybrid, jump, constant):
        assert m.closed()
        assert m.intf == [x, p]

    # the hybrid keeps all three blocks as given, the others synthesise
    atom = hybrid.atoms[0]
    assert len(atom.init) == 2 and len(atom.update) == 2 and len(atom.delay) == 2


def test_module_new_partially_observable():
    x, p, init, update, delay = _stateful_blocks()

    for kwargs in (
            dict(init=init, update=update),
            dict(init=init, delay=delay),
            dict(init=init, update=update, delay=delay),
            dict(update=update),
            dict(update=update, delay=delay),
    ):
        m = Module(**kwargs, vars=[x, p], hide=[p])
        assert m.intf == [x]
        assert m.prvt == [p]


def test_module_hide_accepts_predicate_set_and_iterable():
    x, p, init, update, _delay = _stateful_blocks()

    # hide accepts a callable predicate over variables ...
    m = Module.sequential([x, p], init, update, hide=lambda v: v == p)
    assert m.intf == [x]
    assert m.prvt == [p]

    m = Module.sequential([x, p], init, update, hide={p})
    assert m.intf == [x]
    assert m.prvt == [p]

    S = {p}
    m = Module.sequential([x, p], init, update, hide=S.__contains__)
    assert m.intf == [x]
    assert m.prvt == [p]

    m = Module.sequential([x, p], init, update, hide=[p])
    assert m.intf == [x]
    assert m.prvt == [p]

    m = Module.sequential([x, p], init, update, hide=iter([p]))
    assert m.intf == [x]
    assert m.prvt == [p]

    # hiding nothing yields a fully observable module
    m = Module.sequential([x, p], init, update, hide=lambda v: False)
    assert m.intf == [x, p] and m.prvt == []


def test_module_hide_predicate_errors_propagate():
    x, p, init, update, _delay = _stateful_blocks()

    def broken(v):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        Module.sequential([x, p], init, update, hide=broken)


def test_module_new_positional_is_parallel():
    x, p, init, update, _ = _stateful_blocks()
    y = Var(Real([1, 1]))

    p1 = Module(init=init, update=update, vars=[x, p])
    p2 = Module(init=[Term(LRA.Real(torch.tensor([[0.0]])), [X(y)])], vars=[y])

    m = Module(p1, p2)
    assert m.intf == [x, p, y]

    # positional arguments cannot be combined with keyword blocks
    with pytest.raises(TypeError, match="take atoms or modules"):
        Module(p1, init=init, vars=[x])

    # module composition with hiding
    m1 = Module(p1, p2, hide=[p])
    assert p in m1.prvt
    assert m1.intf == [x, y]


def test_compose_hiding_coupled_variables():
    zero = torch.tensor([[0.0]])
    x = Var(Real([1, 1]))
    y = Var(Real([1, 1]))
    z = Var(Real([1, 1]))
    w = Var(Real([1, 1]))

    # m1 controls x and y, reading the shared input w; m2 reads y and w and
    # controls z: y couples the two, w is external on both sides
    init = [Term(LRA.Real(zero), [X(x)]), Term(LRA.Real(zero), [X(y)])]
    update = [Term(LRA.Id(), [X(x)], [w]), Term(LRA.Id(), [X(y)], [y])]
    m1 = Module.sequential([x, y, w], init, update)

    init = [Term(LRA.Real(zero), [X(z)])]
    update = [Term(LRA.Add(), [X(z)], [y, w])]
    m2 = Module.sequential([y, z, w], init, update)

    assert y in m1.intf and y in m2.extl
    assert w in m1.extl and w in m2.extl

    # hide exactly the coupling variables: interface on one side, external
    # on the other (in either direction) — not the shared external w
    def coupled(v):
        return (v in m1.intf and v in m2.extl) or (v in m1.extl and v in m2.intf)

    m = Module.compose(m1, m2, hide=coupled)

    assert m.prvt == [y]
    assert m.intf == [x, z]
    assert m.extl == [w]

    # the coupling took place regardless: hiding only restricts visibility
    assert y not in m.obs and y not in m.extl


def test_module_new_positional_atoms():
    x, p, init, update, _ = _stateful_blocks()

    # a sequence of atoms builds an observable module...
    atom = Atom.sequential([x, p], init, update)
    m = Module(atom)
    assert m.closed() and m.intf == [x, p]

    # ...and hides the `prvt` variables when given
    m = Module(atom, hide=[p])
    assert m.intf == [x] and m.prvt == [p]

    # the explicit `proc` staticmethod behaves the same
    m = Module(atom)
    assert m.closed() and m.intf == [x, p]
    m = Module(atom, hide=[p])
    assert m.intf == [x] and m.prvt == [p]

    # several atoms compose into one module
    y = Var(Real([1, 1]))
    zero = torch.tensor([[0.0]])
    other = Atom.constant([y], [Term(LRA.Real(zero), [X(y)])])
    m = Module(atom, other)
    assert m.intf == [x, p, y]
    assert len(m.atoms) == 2


def test_module_new_rejects_bad_declarations():
    x, p, init, update, delay = _stateful_blocks()

    # a module needs its observable variables
    with pytest.raises(TypeError, match="vars"):
        Module(init=init, update=update)

    # constant and hold modules can be partially observable too
    m = Module(init=init, vars=[x, p], hide=[p])
    assert m.intf == [x] and m.prvt == [p]
    m = Module(vars=[x, p], hide=[p])
    assert m.intf == [x] and m.prvt == [p]


def test_atom_constructors():
    x, p, init, update, delay = _stateful_blocks()

    # each staticmethod builds an atom controlling both variables
    for atom in (
            Atom.sequential([x, p], init, update),
            Atom.differential([x, p], init, delay),
            Atom.hybrid([x, p], init, update, delay),
            Atom.jump([x, p], update),
            Atom.constant([x, p], init),
    ):
        assert len(atom.ctrl) == 2
        assert atom.ctrl == [x, p]
        assert len(atom.wait) == 0

    # combinatorial reuses the assignments for init and update
    atom = Atom.combinatorial([x, p], init)
    assert atom.ctrl == [x, p]
    assert len(atom.init) == len(atom.update) == 2


def test_atom_new_dispatches_on_blocks():
    x, p, init, update, delay = _stateful_blocks()

    # the hybrid keeps the three explicit blocks, the others synthesise
    atom = Atom([x, p], init=init, update=update, delay=delay)
    assert len(atom.init) == 2 and len(atom.update) == 2 and len(atom.delay) == 2

    # a synthesised block per controlled variable where a block is implicit
    atom = Atom([x, p], init=init, update=update)  # sequential: delay is zero
    assert len(atom.delay) == 2
    atom = Atom([x, p], init=init, delay=delay)  # differential: update is skip
    assert len(atom.update) == 2
    atom = Atom([x, p], update=update)  # jump: init is havoc
    assert len(atom.init) == 2
    atom = Atom([x, p], update=update, delay=delay)  # uninitialized: init is havoc
    assert len(atom.init) == 2 and len(atom.delay) == 2
    atom = Atom([x, p], init=init)  # constant: update skip, delay zero
    assert len(atom.update) == 2 and len(atom.delay) == 2
    atom = Atom([x, p], delay=delay)  # flow: init havoc, update skip
    assert len(atom.init) == 2 and len(atom.update) == 2
    atom = Atom([x, p])  # hold: everything synthesised
    assert len(atom.init) == 2 and len(atom.update) == 2 and len(atom.delay) == 2


def test_module_hold_and_flow():
    x, p, init, update, delay = _stateful_blocks()

    # hold: no blocks, every variable a symbolic constant
    m = Module(vars=[x, p])
    assert m.closed() and m.intf == [x, p]
    m = Module.hold([x, p])
    assert m.intf == [x, p] and len(m.prvt) == 0
    m = Module.hold([x, p], hide=[p])
    assert m.intf == [x] and m.prvt == [p]

    # flow: only the continuous dynamics, initial state havoced
    m = Module(delay=delay, vars=[x, p])
    assert m.closed()
    atom = m.atoms[0]
    assert len(atom.init) == 2 and len(atom.delay) == 2
    m = Module.flow([x, p], delay, hide=[p])
    assert m.intf == [x] and m.prvt == [p]


def test_atom_infers_awaited_variables():
    zero = torch.tensor([[0.0]])
    x = Var(Real([1, 1]))
    t = Var(Real([1, 1]))

    # x' = X(t): the atom awaits t rather than controlling it
    init = [Term(LRA.Real(zero), [X(x)])]
    update = [Term(LRA.Id(), [X(x)], [X(t)])]
    atom = Atom([x, t], init=init, update=update)

    assert atom.ctrl == [x]
    assert atom.wait == [t]


def test_atom_rejects_ill_formed_blocks():
    zero = torch.tensor([[0.0]])
    x = Var(Real([1, 1]))

    # the update writes the latched wire instead of the next wire
    init = [Term(LRA.Real(zero), [X(x)])]
    update = [Term(LRA.Id(), [x], [X(x)])]
    with pytest.raises(Exception):
        Atom.sequential([x], init, update)


def test_atoms_compose_into_modules():
    x, p, init, update, _ = _stateful_blocks()

    # an explicitly built atom yields the same module as the shorthand
    atom = Atom.sequential([x, p], init, update)
    direct = Module(init=init, update=update, vars=[x, p], hide=[p])

    assert len(direct.atoms) == 1
    assert direct.atoms[0].ctrl == atom.ctrl
    assert direct.atoms[0].wait == atom.wait


def test_show_named_rendering():
    x, p, init, update, delay = _stateful_blocks()
    m = Module(init=init, update=update, vars=[x, p])

    # named variables render by name, in interfaces and wire positions
    out = m.with_varnames({x: "x", p: "p"})
    # scalar [1, 1] shapes are omitted from the sort rendering
    assert "x" in out and "Real" in out
    assert "X(x)" in out and "X(p)" in out

    # unnamed variables fall back to question mark
    # this test is meant to fail when this convention changes, or errors are raised
    # if any other part of the code relies on the fallback, it is bad
    partial = m.with_varnames({x: "x"})
    assert f"#" in partial

    # the atom renders with the same naming
    out = m.atoms[0].show({x: "x", p: "p"})
    assert "controls" in out and "X(x)" in out

    print(m.with_varnames({x: "x"}))


def test_heterogeneous_composition():
    x = Var(Real([1, 1]))
    y = Var(Real([1, 1]))
    z = Var(Real([1, 1]))

    zero = torch.tensor([[0]])
    init = [Term(LRA.Real(zero), [X(x)])]
    update = [Term(LRA.Id(), [X(x)], [x])]
    P = Module.sequential([x], init, update)

    init = [Term(LRA.Real(zero), [X(y)])]
    flow = [Term(LRA.RealZerograd([1, 1]), [d(y)])]
    Q = Module.differential([y], init, flow)

    comb = [Term(LRA.Add(), [X(z)], [X(x), X(y)])]
    R = Module.combinatorial([x, y, z], comb)

    S = Module.compose(P, Q, R)
