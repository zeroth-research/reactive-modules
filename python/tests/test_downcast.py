import pytest
from zrth import Wire, Term, Module, IType as it
from zrth import Bool, Int, Float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool_wire():
    return (Wire(Bool(1, 1)), Wire(Bool(1, 1)))


def _int_wire():
    return (Wire(Int(1, 1)), Wire(Int(1, 1)))


def _simple_bool_module():
    x = _bool_wire()
    init = [Term(it.ConstBool(True), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    return Module.sequential(init, update, [x])


def _simple_int_module():
    x = _int_wire()
    init = [Term(it.ConstInt(0), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    return Module.sequential(init, update, [x])


def _logic_module():
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
# LIA: conversion succeeds — __str__ contains expected content
# ---------------------------------------------------------------------------

def test_try_to_lia_lowercase():
    m = _simple_bool_module()
    result = m.try_to("lia")
    assert "module" in str(result)


def test_try_to_lia_uppercase():
    m = _simple_bool_module()
    result = m.try_to("LIA")
    assert "module" in str(result)


def test_try_to_lia_int_wire():
    result = _simple_int_module().try_to("lia")
    assert "module" in str(result)


def test_try_to_lia_logic_ops():
    result = _logic_module().try_to("lia")
    assert "module" in str(result)


def test_try_to_lia_combinatorial():
    x = _bool_wire()
    y = _bool_wire()
    z = _bool_wire()
    assign = [Term(it.Or(), [z[1]], [x[1], y[1]])]
    m = Module.combinatorial(assign, obs=[z, x, y])
    result = m.try_to("lia")
    assert "module" in str(result)


def test_try_to_lia_print(capsys):
    result = _simple_bool_module().try_to("lia")
    print(result)
    assert "module" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# LIA: conversion fails: incompatible dtype or instruction
# ---------------------------------------------------------------------------

def test_try_to_lia_1d_dtype_fails():
    x = (Wire(Bool(3)), Wire(Bool(3)))
    init = [Term(it.ConstBool(True), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("lia")


def test_try_to_lia_float_dtype_fails():
    x = (Wire(Float(1, 1)), Wire(Float(1, 1)))
    init = [Term(it.ConstBool(True), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("lia")


def test_try_to_lia_sub_fails():
    x = _int_wire()
    y = _int_wire()
    init = [Term(it.ConstInt(1), [x[1]]), Term(it.ConstInt(2), [y[1]])]
    update = [Term(it.Sub(), [x[1]], [x[0], y[0]]), Term(it.Id(), [y[1]], [y[0]])]
    m = Module.sequential(init, update, obs=[x, y])
    with pytest.raises(Exception):
        m.try_to("lia")


# ---------------------------------------------------------------------------
# RLA: conversion succeeds
# ---------------------------------------------------------------------------

def test_try_to_rla_lowercase():
    result = _simple_bool_module().try_to("rla")
    assert "module" in str(result)


def test_try_to_rla_uppercase():
    result = _simple_bool_module().try_to("RLA")
    assert "module" in str(result)


def test_try_to_rla_int_wire():
    result = _simple_int_module().try_to("rla")
    assert "module" in str(result)


def test_try_to_rla_logic_ops():
    result = _logic_module().try_to("rla")
    assert "module" in str(result)


def test_try_to_rla_print(capsys):
    result = _simple_bool_module().try_to("rla")
    print(result)
    assert "module" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# RLA: conversion fails: incompatible dtype or instruction
# ---------------------------------------------------------------------------

def test_try_to_rla_1d_dtype_fails():
    x = (Wire(Bool(3)), Wire(Bool(3)))
    init = [Term(it.ConstBool(True), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    m = Module.sequential(init, update, [x])
    with pytest.raises(Exception):
        m.try_to("rla")


def test_try_to_rla_sub_fails():
    x = _int_wire()
    y = _int_wire()
    init = [Term(it.ConstInt(1), [x[1]]), Term(it.ConstInt(2), [y[1]])]
    update = [Term(it.Sub(), [x[1]], [x[0], y[0]]), Term(it.Id(), [y[1]], [y[0]])]
    m = Module.sequential(init, update, obs=[x, y])
    with pytest.raises(Exception):
        m.try_to("rla")


# ---------------------------------------------------------------------------
# Unknown theory
# ---------------------------------------------------------------------------

def test_try_to_unknown_theory():
    with pytest.raises(Exception, match="unknown theory"):
        _simple_bool_module().try_to("bv")


def test_try_to_empty_string():
    with pytest.raises(Exception):
        _simple_bool_module().try_to("")
