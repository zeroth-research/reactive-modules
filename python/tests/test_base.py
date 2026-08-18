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
    _ = Term.function(LIA.Const(torch.tensor([[3, 4, 5]])), [w4], [])
    _ = Term.function(LIA.Const(torch.tensor([[3, 4, 6]])), [w5], [])

    # comparisons are pointwise -> Bool of the operand shape
    Term(LIA.Lt(), [Wire(Bool([1, 3]))], [w4, Wire(Int([1, 3]))])
    Term.constant(LIA.Const(torch.tensor([[3, 2, 1]])), [Wire(Int([1, 3]))])


def test_module_sequential():
    x = Var(Bool([1, 1]))
    init = [Term.constant(LIA.Const(_bool_t(True)), [X(x)])]
    update = [Term(LIA.Id(), [X(x)], [x])]
    _ = Module.sequential(init, update, [x])


def test_module_combinatorial():
    x = Var(Bool([1, 1]))

    assign = [Term.constant(LIA.Const(_bool_t(False)), [X(x)])]
    _ = Module.combinatorial(assign, [x])


def test_module_parallel():
    x = Var(Bool([1, 1]))
    y = Var(Bool([1, 1]))
    z = Var(Bool([1, 1]))
    w = Var(Bool([1, 1]))
    v = Var(Bool([1, 1]))

    init = [Term.constant(LIA.Const(_bool_t(False)), [X(x)])]
    update = [Term(LIA.And(), [X(x)], [x, X(y)])]
    p = Module.sequential(init, update, obs=[x, y])

    init = [
        Term.constant(LIA.Const(_bool_t(False)), [X(v)]),
        Term.constant(LIA.Const(_bool_t(False)), [X(y)]),
    ]
    update = [Term(LIA.And(), [X(v)], [v, x]), Term(LIA.Id(), [X(y)], [x])]
    q = Module.sequential(init, update, obs=[x, y], prvt=[v])

    assign = [Term(LIA.Or(), [X(z)], [X(y), X(w)])]
    r = Module.combinatorial(assign, obs=(z, y, w))

    m = Module.comp(p, q, r)

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
    init = [Term(LRA.Const(zero), [X(x)]), Term(LRA.Const(zero), [X(p)])]
    update = [Term(LRA.Id(), [X(x)], [x]), Term(LRA.Id(), [X(p)], [p])]
    delay = [Term(LRA.Const(zero), [d(x)]), Term(LRA.Const(zero), [d(p)])]
    return x, p, init, update, delay


def test_module_new_dispatches_on_blocks():
    x, p, init, update, delay = _stateful_blocks()

    # each block combination selects the corresponding constructor
    sequential = Module(init=init, update=update, obs=[x, p])
    differential = Module(init=init, delay=delay, obs=[x, p])
    hybrid = Module(init=init, update=update, delay=delay, obs=[x, p])
    jump = Module(update=update, obs=[x, p])
    constant = Module(init=init, obs=[x, p])

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
        m = Module(**kwargs, obs=[x], prvt=[p])
        assert m.intf == [x]
        assert m.prvt == [p]


def test_module_new_positional_is_parallel():
    x, p, init, update, _ = _stateful_blocks()
    y = Var(Real([1, 1]))

    p1 = Module(init=init, update=update, obs=[x, p])
    p2 = Module(init=[Term(LRA.Const(torch.tensor([[0.0]])), [X(y)])], obs=[y])

    m = Module(p1, p2)
    assert m.intf == [x, p, y]

    # positional arguments cannot be combined with keyword blocks
    with pytest.raises(TypeError, match="take atoms or modules"):
        Module(p1, init=init, obs=[x])

    # module composition takes no prvt
    with pytest.raises(TypeError, match="takes no `prvt`"):
        Module(p1, p2, prvt=[p])


def test_module_new_positional_atoms():
    x, p, init, update, _ = _stateful_blocks()

    # a sequence of atoms builds an observable module...
    atom = Atom.sequential([x, p], init, update)
    m = Module(atom)
    assert m.closed() and m.intf == [x, p]

    # ...and hides the `prvt` variables when given
    m = Module(atom, prvt=[p])
    assert m.intf == [x] and m.prvt == [p]

    # the explicit `proc` staticmethod behaves the same
    m = Module.proc(atom)
    assert m.closed() and m.intf == [x, p]
    m = Module.proc(atom, prvt=[p])
    assert m.intf == [x] and m.prvt == [p]

    # several atoms compose into one module
    y = Var(Real([1, 1]))
    zero = torch.tensor([[0.0]])
    other = Atom.constant([y], [Term(LRA.Const(zero), [X(y)])])
    m = Module(atom, other)
    assert m.intf == [x, p, y]
    assert len(m.atoms) == 2


def test_module_new_rejects_bad_declarations():
    x, p, init, update, delay = _stateful_blocks()

    # a module needs its observable variables
    with pytest.raises(TypeError, match="obs"):
        Module(init=init, update=update)

    # constant and hold modules are fully observable: `prvt` is invalid
    with pytest.raises(TypeError, match="constant modules .* fully observable"):
        Module(init=init, obs=[x], prvt=[p])
    with pytest.raises(TypeError, match="hold modules .* fully observable"):
        Module(obs=[x], prvt=[p])


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
    atom = Atom([x, p], init=init, update=update)   # sequential: delay is zero
    assert len(atom.delay) == 2
    atom = Atom([x, p], init=init, delay=delay)     # differential: update is skip
    assert len(atom.update) == 2
    atom = Atom([x, p], update=update)              # jump: init is havoc
    assert len(atom.init) == 2
    atom = Atom([x, p], update=update, delay=delay)  # uninitialized: init is havoc
    assert len(atom.init) == 2 and len(atom.delay) == 2
    atom = Atom([x, p], init=init)                  # constant: update skip, delay zero
    assert len(atom.update) == 2 and len(atom.delay) == 2
    atom = Atom([x, p], delay=delay)                # flow: init havoc, update skip
    assert len(atom.init) == 2 and len(atom.update) == 2
    atom = Atom([x, p])                             # hold: everything synthesised
    assert len(atom.init) == 2 and len(atom.update) == 2 and len(atom.delay) == 2


def test_module_hold_and_flow():
    x, p, init, update, delay = _stateful_blocks()

    # hold: no blocks, every variable a symbolic constant; fully observable
    m = Module(obs=[x, p])
    assert m.closed() and m.intf == [x, p]
    m = Module.hold([x, p])
    assert m.intf == [x, p] and len(m.prvt) == 0
    with pytest.raises(TypeError, match="fully observable"):
        Module(obs=[x], prvt=[p])

    # flow: only the continuous dynamics, initial state havoced
    m = Module(delay=delay, obs=[x, p])
    assert m.closed()
    atom = m.atoms[0]
    assert len(atom.init) == 2 and len(atom.delay) == 2
    m = Module.flow(delay, [x], prvt=[p])
    assert m.intf == [x] and m.prvt == [p]


def test_atom_infers_awaited_variables():
    zero = torch.tensor([[0.0]])
    x = Var(Real([1, 1]))
    t = Var(Real([1, 1]))

    # x' = X(t): the atom awaits t rather than controlling it
    init = [Term(LRA.Const(zero), [X(x)])]
    update = [Term(LRA.Id(), [X(x)], [X(t)])]
    atom = Atom([x, t], init=init, update=update)

    assert atom.ctrl == [x]
    assert atom.wait == [t]


def test_atom_rejects_ill_formed_blocks():
    zero = torch.tensor([[0.0]])
    x = Var(Real([1, 1]))

    # the update writes the latched wire instead of the next wire
    init = [Term(LRA.Const(zero), [X(x)])]
    update = [Term(LRA.Id(), [x], [X(x)])]
    with pytest.raises(Exception):
        Atom.sequential([x], init, update)


def test_atoms_compose_into_modules():
    x, p, init, update, _ = _stateful_blocks()

    # an explicitly built atom yields the same module as the shorthand
    atom = Atom.sequential([x, p], init, update)
    direct = Module(init=init, update=update, obs=[x], prvt=[p])

    assert len(direct.atoms) == 1
    assert direct.atoms[0].ctrl == atom.ctrl
    assert direct.atoms[0].wait == atom.wait


def test_heterogeneous_composition():
    x = Var(Real([1, 1]))
    y = Var(Real([1, 1]))
    z = Var(Real([1, 1]))

    zero = torch.tensor([[0]])
    init = [Term(LRA.Const(zero), [X(x)])]
    update = [Term(LRA.Id(), [X(x)], [x])]
    P = Module.sequential(init, update, [x])

    init = [Term(LRA.Const(zero), [X(y)])]
    flow = [Term(LRA.Const(zero), [d(y)])]
    Q = Module.differential(init, flow, [y])

    comb = [Term(LRA.Add(), [X(z)], [X(x), X(y)])]
    R = Module.combinatorial(comb, [x, y, z])

    S = Module.comp(P, Q, R)
