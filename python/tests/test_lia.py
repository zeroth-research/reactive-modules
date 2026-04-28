from zrth import IType as it
from zrth.lia import Bool, Int, Wire, Term, Module


def test_import():
    from zrth.lia import Bool, Int, Wire, Term, Atom, Type


def test_bool_shape():
    b = Bool(1, 1)
    assert b.shape == (1, 1)
    b2 = Bool(2, 3)
    assert b2.shape == (2, 3)


def test_int_shape():
    i = Int(1, 1)
    assert i.shape == (1, 1)
    i2 = Int(4, 8)
    assert i2.shape == (4, 8)


def test_wire_new():
    Wire(Bool(1, 1))
    Wire(Int(1, 1))
    Wire(Int(2, 3))


def test_wire_id():
    w1 = Wire(Bool(1, 1))
    w2 = Wire(Bool(1, 1))
    assert w1.id != w2.id


def test_wire_eq():
    w = Wire(Bool(1, 1))
    w2 = Wire(Bool(1, 1))
    assert w == w
    assert w != w2

def test_wire_dtype():
    w = Wire(Bool(1, 1))
    w2 = Wire(Bool(1, 1))
    assert w.dtype == w2.dtype
    w2 = Wire(Bool(2, 1))
    assert w.dtype != w2.dtype
    w2 = Wire(Int(1, 1))
    assert w.dtype != w2.dtype
    assert w.dtype.shape == w2.dtype.shape
    assert w.dtype.is_bool
    assert w2.dtype.is_int


def test_wire_repr():
    w = Wire(Bool(1, 1))
    r = repr(w)
    print(r)
    assert isinstance(r, str) and len(r) > 0


def test_wire_hash():
    w = Wire(Bool(1, 1))
    d = {w: 42}
    assert d[w] == 42


def test_term_constant():
    w = Wire(Bool(1, 1))
    t = Term.constant(it.ConstBool(True), [w])
    assert len(t.write) == 1
    assert t.write[0] == w
    assert len(t.read) == 0


def test_term_function():
    x = Wire(Bool(1, 1))
    y = Wire(Bool(1, 1))
    out = Wire(Bool(1, 1))
    t = Term.function(it.And(), [out], [x, y])
    assert t.write == [out]
    assert t.read == [x, y]


def test_term_new_constant():
    w = Wire(Int(1, 1))
    t = Term(it.ConstInt(0), [w])
    assert len(t.write) == 1
    assert len(t.read) == 0


def test_term_new_function():
    x = Wire(Bool(1, 1))
    out = Wire(Bool(1, 1))
    t = Term(it.Not(), [out], [x])
    assert t.write == [out]
    assert t.read == [x]


def test_term_itype():
    w = Wire(Bool(1, 1))
    t = Term.constant(it.ConstBool(False), [w])
    print(t)
    print(t.itype)
    # not implemented yet
    # assert t.itype == it.ConstBool(False)


def test_term_interface_len_and_index():
    x = Wire(Bool(1, 1))
    y = Wire(Bool(1, 1))
    out = Wire(Bool(1, 1))
    t = Term.function(it.Or(), [out], [x, y])
    assert len(t.read) == 2
    assert t.read[0] == x
    assert t.read[1] == y
    assert len(t.write) == 1
    assert t.write[0] == out

def test_term_interface():
    x = Wire(Bool(1, 1))
    y = Wire(Bool(1, 1))
    out = Wire(Bool(1, 1))
    t = Term(it.Or(), [out], [x, y])
    for n, w in enumerate(t.read):
        print(w)
        if n == 0: assert w == x
        if n == 1: assert w == y
        assert n <= 1
    for n, w in enumerate(t.write):
        print(w)
        if n == 0: assert w == out
        assert n <= 0


def test_term_id():
    x = Wire(Int(1, 1))
    out = Wire(Int(1, 1))
    t = Term.function(it.Id(), [out], [x])
    assert t.read == [x]
    assert t.write == [out]


# Module and Atom tests

def _make_sequential_module():
    x = (Wire(Bool(1, 1)), Wire(Bool(1, 1)))
    init = [Term(it.ConstBool(False), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    return Module.sequential(init, update, obs=[x])


def test_module_sequential():
    m = _make_sequential_module()
    assert m is not None


def test_module_combinatorial():
    x = (Wire(Bool(1, 1)), Wire(Bool(1, 1)))
    assign = [Term(it.ConstBool(True), [x[1]])]
    m = Module.combinatorial(assign, obs=[x])
    assert m is not None


def test_module_atoms_nonempty():
    m = _make_sequential_module()
    assert len(m.atoms) > 0


def test_atom_from_module():
    m = _make_sequential_module()
    atom = m.atoms[0]
    assert atom is not None


def test_iterate_atoms_in_module():
    m = _make_sequential_module()
    count = 0
    for atom in m.atoms:
        assert atom is not None
        count += 1
    assert count == len(m.atoms)


def test_iterate_terms_in_atom_init():
    m = _make_sequential_module()
    atom = m.atoms[0]
    count = 0
    for term in atom.init():
        assert term is not None
        count += 1
    assert count > 0


def test_iterate_terms_in_atom_update():
    m = _make_sequential_module()
    atom = m.atoms[0]
    count = 0
    for term in atom.update():
        assert term is not None
        count += 1
    assert count > 0


def test_atom_init_term_wires():
    m = _make_sequential_module()
    atom = m.atoms[0]
    for term in atom.init():
        assert len(term.write()) > 0


def test_atom_update_term_wires():
    m = _make_sequential_module()
    atom = m.atoms[0]
    for term in atom.update():
        assert len(term.write()) > 0


