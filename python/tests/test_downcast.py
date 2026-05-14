import pytest
from zrth import Wire, Term, Module, IType as it
from zrth import Bool, Int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool_wire():
    """A 1×1 boolean wire pair suitable for LIA."""
    return (Wire(Bool(1, 1)), Wire(Bool(1, 1)))


def _int_wire():
    """A 1×1 integer wire pair suitable for LIA."""
    return (Wire(Int(1, 1)), Wire(Int(1, 1)))


def _simple_bool_module():
    """Sequential module: const-init + identity update over a Bool(1,1) wire."""
    x = _bool_wire()
    init = [Term(it.ConstBool(True), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    return Module.sequential(init, update, [x])


def _simple_int_module():
    """Sequential module: const-init + identity update over an Int(1,1) wire."""
    x = _int_wire()
    init = [Term(it.ConstInt(0), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    return Module.sequential(init, update, [x])


def _logic_module():
    """Sequential module using And/Or over Bool(1,1) wires."""
    x = _bool_wire()
    y = _bool_wire()
    init = [
        Term(it.ConstBool(False), [x[1]]),
        Term(it.ConstBool(True), [y[1]]),
    ]
    update = [
        Term(it.And(), [x[1]], [x[0], y[0]]),
        Term(it.Id(), [y[1]], [y[0]]),
    ]
    return Module.sequential(init, update, obs=[x, y])


# ---------------------------------------------------------------------------
# Conversion succeeds
# ---------------------------------------------------------------------------

def test_try_to_lia_lowercase(capfd):
    m = _simple_bool_module()
    m.try_to("lia")
    out = capfd.readouterr().out
    assert "module" in out


def test_try_to_lia_uppercase(capfd):
    m = _simple_bool_module()
    m.try_to("LIA")
    out = capfd.readouterr().out
    assert "module" in out


def test_try_to_lia_int_wire(capfd):
    m = _simple_int_module()
    m.try_to("lia")
    capfd.readouterr()  # should not raise


def test_try_to_lia_logic_ops(capfd):
    m = _logic_module()
    m.try_to("lia")
    capfd.readouterr()


def test_try_to_lia_combinatorial(capfd):
    x = _bool_wire()
    y = _bool_wire()
    z = _bool_wire()
    # combinatorial: read from next wires (x[1], y[1]), write to z[1]
    assign = [Term(it.Or(), [z[1]], [x[1], y[1]])]
    m = Module.combinatorial(assign, obs=[z, x, y])
    m.try_to("lia")
    capfd.readouterr()


# ---------------------------------------------------------------------------
# Conversion fails: unknown theory
# ---------------------------------------------------------------------------

def test_try_to_unknown_theory():
    m = _simple_bool_module()
    with pytest.raises(Exception, match="unknown theory"):
        m.try_to("bv")


def test_try_to_empty_string():
    m = _simple_bool_module()
    with pytest.raises(Exception):
        m.try_to("")


# ---------------------------------------------------------------------------
# Conversion fails: incompatible dtype (not 2-dimensional)
# ---------------------------------------------------------------------------

def test_try_to_lia_1d_dtype_fails():
    """Bool([3]) has only one dimension; LIA requires exactly two."""
    from zrth import Bool
    x = (Wire(Bool(3)), Wire(Bool(3)))
    init = [Term(it.ConstBool(True), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("lia")


def test_try_to_lia_float_dtype_fails():
    """Float wires have no LIA counterpart."""
    from zrth import Float
    x = (Wire(Float(1, 1)), Wire(Float(1, 1)))
    init = [Term(it.ConstBool(True), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("lia")


# ---------------------------------------------------------------------------
# Conversion fails: incompatible instruction (not in LIA)
# ---------------------------------------------------------------------------

def test_try_to_lia_sub_fails():
    """Sub() has no LIA counterpart."""
    x = _int_wire()
    y = _int_wire()
    init = [
        Term(it.ConstInt(1), [x[1]]),
        Term(it.ConstInt(2), [y[1]]),
    ]
    update = [
        Term(it.Sub(), [x[1]], [x[0], y[0]]),
        Term(it.Id(), [y[1]], [y[0]]),
    ]
    m = Module.sequential(init, update, obs=[x, y])
    with pytest.raises(Exception):
        m.try_to("lia")
