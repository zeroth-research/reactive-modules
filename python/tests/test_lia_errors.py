import pytest
from zrth import IType as it
from zrth.lia import Bool, Wire, Term, Module


# ── Term-level errors ──────────────────────────────────────────────────────────

def test_term_reads_and_writes_same_wire():
    w = Wire(Bool(1, 1))
    with pytest.raises(Exception, match="reads and writes the same wire"):
        Term(it.Id(), [w], [w])


def test_block_wire_written_twice():
    x = (Wire(Bool(1, 1)), Wire(Bool(1, 1)))
    t1 = Term(it.ConstBool(False), [x[1]])
    t2 = Term(it.ConstBool(True), [x[1]])
    with pytest.raises(Exception, match="written more than once"):
        Module.sequential([t1, t2], [t1], obs=[x])


# ── Atom-level errors: missing writes ─────────────────────────────────────────

def test_controlled_wire_not_written_in_init():
    x = (Wire(Bool(1, 1)), Wire(Bool(1, 1)))
    y = Wire(Bool(1, 1))
    # init writes an unrelated wire instead of x[1]
    init = [Term(it.ConstBool(False), [y])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    with pytest.raises(Exception, match="not written in init"):
        Module.sequential(init, update, obs=[x])


def test_controlled_wire_not_written_in_update():
    x = (Wire(Bool(1, 1)), Wire(Bool(1, 1)))
    y = Wire(Bool(1, 1))
    init = [Term(it.ConstBool(False), [x[1]])]
    # update writes an unrelated wire instead of x[1]
    update = [Term(it.ConstBool(False), [y])]
    with pytest.raises(Exception, match="not written in update"):
        Module.sequential(init, update, obs=[x])


# ── Atom-level errors: wrong wire kind ────────────────────────────────────────

def test_writing_latched_wire_in_init():
    x = (Wire(Bool(1, 1)), Wire(Bool(1, 1)))
    # x[0] is the latched wire; only x[1] (next) may be written
    init = [Term(it.ConstBool(False), [x[0]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    with pytest.raises(Exception, match="[Ll]atched wire"):
        Module.sequential(init, update, obs=[x])


def test_init_reads_latched_wire():
    x = (Wire(Bool(1, 1)), Wire(Bool(1, 1)))
    # init must not read the latched wire x[0]
    init = [Term(it.Id(), [x[1]], [x[0]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    with pytest.raises(Exception, match="[Ii]nit reads latched wire"):
        Module.sequential(init, update, obs=[x])


# ── Module-level errors: private wires ────────────────────────────────────────

def test_private_wire_not_controlled():
    x = (Wire(Bool(1, 1)), Wire(Bool(1, 1)))
    v = (Wire(Bool(1, 1)), Wire(Bool(1, 1)))
    # v is declared private but no term writes v[1]
    init = [Term(it.ConstBool(False), [x[1]])]
    update = [Term(it.Id(), [x[1]], [x[0]])]
    with pytest.raises(Exception, match="not controlled"):
        Module.sequential(init, update, obs=[x], prvt=[v])
