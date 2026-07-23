"""Tests for `build.Module` — the subclass-a-Module DSL front-end.

A `build.Module` subclass *is* a base Module: pass `theory` and the `ctrl`(/`extl`) wire
pairs and override `init`/`update`, and the sequential module is built in the constructor.
These tests build small modules and step them through `zrth.eval` (mirroring test_eval),
plus check the ctrl/extl partition and the config surface.
"""

import pytest

from zrth import LIA, Module, Sort, Wire, build
from zrth.build import nxt, ite
from zrth.eval import eval_itype

INT = Sort.Int([1, 1])


def _pair():
    """A fresh (latched, next) wire pair (test helper)."""
    return (Wire(INT), Wire(INT))


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
    return {ltc: state[nxt_w] for (ltc, nxt_w) in m.ctrl}


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


class Counter(build.Module):
    def init(self):
        return 0

    def update(self, ctrl):
        return ctrl + 1


def test_counter_is_a_closed_base_module():
    m = Counter(theory=LIA, ctrl=(_pair(),))
    assert isinstance(m, Module)
    assert m.closed()


def test_counter_counts():
    x = _pair()
    m = Counter(theory=LIA, ctrl=(x,))
    (_x_lat, x_nxt) = x
    assert _trace(m, 5, x_nxt) == [0, 1, 2, 3, 4, 5]


# --- multi-var with ite and an unchanged var --------------------------------


class Bounded(build.Module):
    def init(self):
        return 0, 3

    def update(self, ctrl):
        x, cap = ctrl
        return ite(x < cap, x + 1, x), cap  # x climbs to cap, cap unchanged


def test_multivar_ite_and_hold():
    x, cap = _pair(), _pair()
    m = Bounded(theory=LIA, ctrl=(x, cap))
    assert m.closed()
    assert _trace(m, 5, x[1]) == [0, 1, 2, 3, 3, 3]
    assert _trace(m, 5, cap[1]) == [3, 3, 3, 3, 3, 3]


# --- extl variables become external inputs (open module) --------------------


class Gate(build.Module):
    def init(self):
        return 0

    def update(self, ctrl, extl):
        return ite(ctrl < extl, ctrl + 1, ctrl)


def test_extl_is_external_and_module_is_open():
    x, y = _pair(), _pair()  # x controlled, y external
    m = Gate(theory=LIA, ctrl=(x,), extl=(y,))
    assert m.open()
    assert len(list(m.ctrl)) == 1
    assert len(list(m.extl)) == 1


# --- config surface ---------------------------------------------------------


def test_builds_from_wire_pairs_passed_directly():
    class C(build.Module):
        def init(self):
            return 0

        def update(self, ctrl):
            return ctrl + 1

    m = C(theory=LIA, ctrl=(_pair(),))
    assert m.closed()


def test_missing_config_raises():
    class Bad(build.Module):
        def update(self, ctrl):
            return ctrl

    with pytest.raises(TypeError):
        Bad()  # no theory / ctrl


def test_missing_init_raises():
    class NoInit(build.Module):
        def update(self, ctrl):
            return ctrl + 1

    # init is required — every ctrl variable needs an initial value
    with pytest.raises(TypeError):
        NoInit(theory=LIA, ctrl=(_pair(),))


def test_subexpression_shared_across_returns():
    # a guard reused in several returned values must be emitted once, not once per
    # value (else its wire is "written more than once")
    class Shared(build.Module):
        def init(self):
            return 0, 0

        def update(self, ctrl):
            x, y = ctrl
            loop = x < y
            return ite(loop, x + 1, x), ite(loop, y, y - 1)

    m = Shared(theory=LIA, ctrl=(_pair(), _pair()))
    assert m.closed()
