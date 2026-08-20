"""Tests for `sugar.Module` — the subclass-a-Module DSL front-end.

A `sugar.Module` subclass *is* a base Module: pass `theory` and the `ctrl`(/`extl`) wire
pairs and override `init`/`update`, and the sequential module is built in the constructor.
These tests build small modules and step them through `zrth.eval` (mirroring test_eval),
plus check the ctrl/extl partition and the config surface.
"""

import pytest

from zrth import LIA, Module, Sort, Wire, sugar, Int, Var, Real, LRA, X as _X, d as _d, NonLinearError
from zrth.sugar import ite, d, X
from zrth.eval import eval_itype

INT = Int([1, 1])
Real1 = Real([1, 1])


def _pair():
    """A fresh (latched, next) wire pair (test helper)."""
    return Var(INT)


# --- stepping helpers (same shape as test_eval) -----------------------------


def _run_block(m, state, get_block):
    for a in m.atoms:
        for t in get_block(a):
            read = [state[w] for w in t.read]
            out_sort = t.write[0].dtype if len(t.write) else None
            state.update(zip(t.write, eval_itype(t.itype, read, out_sort)))
    return state


def _init(m):
    return _run_block(m, {}, lambda a: a.init)


def _latch(m, state):
    return {var: state[_X(var)] for var in m.ctrl}


def _update(m, state):
    return _run_block(m, state, lambda a: a.update)


def _trace(m, steps, wire):
    """Latched value of `wire` after init and after each of `steps` updates."""
    state = _init(m)
    out = [state[wire].item()]
    for _ in range(steps):
        state = _update(m, _latch(m, state))
        out.append(state[wire].item())
    return out


# --- counter (closed, single var, no extl) ----------------------------------


class Counter(sugar.Module):
    def init(self):
        return 0

    def update(self, cnt):
        return cnt + 1


def test_counter_is_a_closed_base_module():
    m = Counter(theory=LIA, ctrl=(_pair(),))
    assert isinstance(m, Module)
    assert m.closed()


def test_counter_counts():
    x = _pair()
    m = Counter(theory=LIA, ctrl=(x,))
    print(m)
    # (_x_lat, x_nxt) = x
    assert _trace(m, 5, _X(x)) == [0, 1, 2, 3, 4, 5]


# --- multi-var with ite and an unchanged var --------------------------------


class Bounded(sugar.Module):
    def init(self):
        return 0, 3

    def update(self, x, cap):
        return ite(x < cap, x + 1, x), cap  # x climbs to cap, cap unchanged


def test_multivar_ite_and_hold():
    cap, x = _pair(), _pair()
    m = Bounded(theory=LIA, ctrl=(x, cap))
    assert m.closed()
    assert _trace(m, 5, _X(x)) == [0, 1, 2, 3, 3, 3]
    assert _trace(m, 5, _X(cap)) == [3, 3, 3, 3, 3, 3]


# --- extl variables become external inputs (open module) --------------------


class Gate(sugar.Module):
    def init(self, _y):
        return 0

    def update(self, x, y):
        return ite(x < y, x + 1, y)


def test_extl_is_external_and_module_is_open():
    x, y = _pair(), _pair()  # x controlled, y external
    m = Gate(theory=LIA, ctrl=(x,), extl=(y,))
    assert m.open()
    assert len(list(m.ctrl)) == 1
    assert len(list(m.extl)) == 1


# --- config surface ---------------------------------------------------------


def test_builds_from_wire_pairs_passed_directly():
    class C(sugar.Module):
        def init(self):
            return 0

        def update(self, ctrl):
            return ctrl + 1

    m = C(theory=LIA, ctrl=(_pair(),))
    assert m.closed()


def test_missing_config_raises():
    class Bad(sugar.Module):
        def update(self, ctrl):
            return ctrl

    with pytest.raises(TypeError):
        Bad()  # no theory / ctrl


def test_missing_init_raises():
    class NoInit(sugar.Module):
        def update(self, ctrl):
            return ctrl + 1

    # init is required — every ctrl variable needs an initial value
    # not true anymore - this is a jump module
    # with pytest.raises(TypeError):

    m = NoInit(theory=LIA, ctrl=(_pair(),))
    assert m.closed()


def test_subexpression_shared_across_returns():
    # a guard reused in several returned values must be emitted once, not once per
    # value (else its wire is "written more than once")
    class Shared(sugar.Module):
        def init(self):
            return 0, 0

        def update(self, x, y):
            loop = x < y
            return ite(loop, x + 1, x), ite(loop, y, y - 1)

    m = Shared(theory=LIA, ctrl=(_pair(), _pair()))
    assert m.closed()


def test_simple_clock():
    class Simple(sugar.Module):
        def init(self, t):
            return 0

        def update(self, x, t):
            return ite(X(t) == 1, 0, x)

        def delay(self, x, t):
            return 1 * d(t)

    x = Var(Real1)
    t = Var(Real1)
    m = Simple(ctrl=(x,), extl=(t,), theory=LRA)


def test_differential_sugar_module():
    """init + delay (no update) builds a differential module: the update is
    the synthesised skip, and the delay drives the derivative wires."""

    class Drift(sugar.Module):
        def init(self, t):
            return 0

        def delay(self, x, t):
            return d(t) + d(t)  # dx = 2 dt: a drifting clock

    x = Var(Real1)
    t = Var(Real1)
    m = Drift(ctrl=(x,), extl=(t,), theory=LRA)

    assert isinstance(m, Module)
    assert m.open()  # t is inferred external
    assert m.intf == [x]
    assert m.extl == [t]

    a = m.atoms[0]
    # the delay drives d(x); the synthesised skip update drives X(x)
    assert _d(x) in a.delay.write()
    assert _X(x) in a.update.write()
    assert len(a.update) == 1


def test_hybrid_sugar_module():
    """init + update + delay builds a hybrid module: all three blocks are
    explicit, mixing a discrete reset with continuous drift."""

    class Clock(sugar.Module):
        def init(self, t):
            return 0

        def update(self, x, t):
            return ite(X(t) == 1, 0, x)  # discrete reset when t' hits 1

        def delay(self, x, t):
            return 1 * d(t)  # continuous drift with t

    x = Var(Real1)
    t = Var(Real1)
    m = Clock(ctrl=(x,), extl=(t,), theory=LRA)

    assert isinstance(m, Module)
    assert m.open()
    assert m.intf == [x]
    assert m.extl == [t]

    a = m.atoms[0]
    # every block is explicit and drives its wire of x
    assert _X(x) in a.init.write()
    assert _X(x) in a.update.write()
    assert _d(x) in a.delay.write()
    # the update reads the reset guard from the input's next value
    assert _X(t) in a.update.read()


def test_invalid_nonlinear():
    class Simple(sugar.Module):
        def init(self, t):
            return 0

        def update(self, x, t):
            return ite(X(t) == 1, 0, x)

        def delay(self, x, t):
            return x * d(t)

    x = Var(Real1)
    t = Var(Real1)
    with pytest.raises(NonLinearError):
        Simple(ctrl=(x,), extl=(t,), theory=LRA)


def test_useless_module():
    class Useless(sugar.Module):
        def init(self, i):
            return ()

        def update(self, i):
            return ()

    i = Var(Real1)
    m = Useless(ctrl=(), extl=(i,), theory=LRA)
    assert m.closed()  # unused externals and no control -> closed module
    assert len(m.extl) + len(m.intf) + len(m.prvt) == 0


def test_private():
    class DigitalDelay(sugar.Module):
        def update(self, x, _o, i):
            return (i, x)

    i = Var(Real1)
    o = Var(Real1)
    x = Var(Real1)
    m = DigitalDelay(ctrl=(x, o), extl=(i,), hide=frozenset([x]), theory=LRA)

    assert i in m.extl
    assert o in m.intf
    assert x in m.prvt
