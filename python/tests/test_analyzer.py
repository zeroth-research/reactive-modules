"""Analyzer extensions — Python patterns the analyzer must translate to terms.

Each feature gets a small gym.Env that exercises it. Tests run the extracted
module either through the IR interpreter (`interpret=True`) when all terms
are interpretable, or via real-env delegation when they aren't (untraced calls)."""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from zrth.gym import Env


# ── boilerplate ───────────────────────────────────────────────────

class _MinEnv(gym.Env):
    """Scalar Box obs, Discrete(1) action. Subclasses override step()."""

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(1,))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return 0.0, 0.0, False, False


# ── analyzer extensions ───────────────────────────────────────────

class _AugAssign(_MinEnv):
    def __init__(self):
        super().__init__(); self.x = 0.0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.x = 0.0
        return self.x, 0.0, False, False
    def step(self, action):
        self.x += 1.5
        self.x *= 2.0
        return self.x, 0.0, False, False

def test_augmented_assignment():
    e = Env(_AugAssign(), interpret=True); e.reset(); e.step(0)
    assert e.x == 3.0   # (0 + 1.5) * 2


class _IfNoElse(_MinEnv):
    def __init__(self):
        super().__init__(); self.x = 0.0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.x = 0.0
        return self.x, 0.0, False, False
    def step(self, action):
        self.x = self.x + 1.0
        if self.x > 2.5:
            self.x = 0.0
        return self.x, 0.0, False, False

def test_if_without_else():
    e = Env(_IfNoElse(), interpret=True); e.reset()
    for _ in range(3):
        e.step(0)
    assert e.x == 0.0   # 1, 2, 3 → (3 > 2.5) resets to 0


class _IfElse(_MinEnv):
    def __init__(self):
        super().__init__(); self.x = 0.0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.x = 0.0
        return self.x, 0.0, False, False
    def step(self, action):
        self.x = self.x + 1.0
        if self.x > 5.0:
            self.x = 0.0
        else:
            self.x = 100.0
        return self.x, 0.0, False, False

def test_if_else():
    e = Env(_IfElse(), interpret=True); e.reset()
    e.step(0); assert e.x == 100.0   # 1 > 5 False → else
    e.step(0); assert e.x == 0.0     # 101 > 5 True → if
    e.step(0); assert e.x == 100.0   # 1 > 5 False → else


class _IfElif(_MinEnv):
    def __init__(self):
        super().__init__(); self.x = 0.0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.x = 0.0
        return self.x, 0.0, False, False
    def step(self, action):
        self.x = self.x + 1.0
        if self.x < 1.5:
            self.x = 0.5
        elif self.x < 2.5:
            self.x = 1.5
        else:
            self.x = 2.5
        return self.x, 0.0, False, False

def test_if_elif_else():
    e = Env(_IfElif(), interpret=True); e.reset()
    e.step(0); assert e.x == 0.5    # 0+1=1 → if-branch
    e.step(0); assert e.x == 1.5    # 0.5+1=1.5 → elif-branch
    e.step(0); assert e.x == 2.5    # 1.5+1=2.5 → else-branch


class _NestedIf(_MinEnv):
    def __init__(self):
        super().__init__(); self.x = 0.0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.x = 0.0
        return self.x, 0.0, False, False
    def step(self, action):
        self.x = self.x + 1.0
        if self.x > 1.5:
            if self.x > 2.5:
                self.x = 0.0
        return self.x, 0.0, False, False

def test_nested_if():
    e = Env(_NestedIf(), interpret=True); e.reset()
    e.step(0); assert e.x == 1.0   # 1>1.5 False → unchanged
    e.step(0); assert e.x == 2.0   # 2>1.5 True, 2>2.5 False → unchanged
    e.step(0); assert e.x == 0.0   # 3>1.5 True, 3>2.5 True → reset


class _BoolChainTernary(_MinEnv):
    """One expression covering boolean ops, a chained comparison, and a ternary."""
    def __init__(self):
        super().__init__(); self.x = 0.0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.x = 0.0
        return self.x, 0.0, False, False
    def step(self, action):
        next_x = self.x + 1.0
        cond = 0.0 < next_x < 10.0 and not (next_x == 5.0)
        self.x = next_x if cond else -1.0
        return self.x, 0.0, False, False

def test_bool_chained_ternary():
    e = Env(_BoolChainTernary(), interpret=True); e.reset()
    for _ in range(4):
        e.step(0)
    assert e.x == 4.0     # 0 < 4 < 10 and not(4==5) → 4
    e.step(0); assert e.x == -1.0   # next_x=5; not(5==5) is False → ternary else


class _InlineMethod(_MinEnv):
    """Inline call to a sibling method — analyzer parses callee's source and inlines
    its return expression. The callee uses self.* (parameter binding isn't supported)."""
    def __init__(self):
        super().__init__(); self.x = 0.0
    def _next_x(self):
        return self.x + 1.5
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.x = 0.0
        return self.x, 0.0, False, False
    def step(self, action):
        self.x = self._next_x()
        return self.x, 0.0, False, False

def test_inline_method():
    e = Env(_InlineMethod(), interpret=True); e.reset()
    e.step(0); assert e.x == 1.5
    e.step(0); assert e.x == 3.0


class _Power(_MinEnv):
    def __init__(self):
        super().__init__(); self.x = 2.0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.x = 2.0
        return self.x, 0.0, False, False
    def step(self, action):
        self.x = self.x ** 3
        return self.x, 0.0, False, False

def test_power_operator():
    e = Env(_Power(), interpret=True); e.reset(); e.step(0)
    assert e.x == 8.0


class _NpClip(_MinEnv):
    def __init__(self):
        super().__init__(); self.x = 0.0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.x = 0.0
        return self.x, 0.0, False, False
    def step(self, action):
        self.x = np.clip(self.x + 5.0, -2.0, 3.0)
        return self.x, 0.0, False, False

def test_np_clip():
    e = Env(_NpClip(), interpret=True); e.reset(); e.step(0)
    assert e.x == 3.0


class _BoolCast(_MinEnv):
    def __init__(self):
        super().__init__(); self.x = 0.0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.x = 0.0
        return self.x, 0.0, False, False
    def step(self, action):
        self.x = self.x + 1.0
        if bool(self.x > 1.5):
            self.x = 100.0
        return self.x, 0.0, False, False

def test_bool_cast():
    e = Env(_BoolCast(), interpret=True); e.reset()
    e.step(0); assert e.x == 1.0
    e.step(0); assert e.x == 100.0   # bool(2 > 1.5) → True branch


def _external_double(x):
    """Top-level helper the analyzer can't trace."""
    return x * 2.0

class _Untraced(_MinEnv):
    def __init__(self):
        super().__init__(); self.x = 1.0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.x = 1.0
        return self.x, 0.0, False, False
    def step(self, action):
        self.x = _external_double(self.x)
        return self.x, 0.0, False, False

def test_untraced_call():
    e = Env(_Untraced())   # real env runs the function via delegation
    labels = [str(t.itype) for atom in e.atoms for t in list(atom.update)]
    assert any('(...)' in lbl for lbl in labels), "expected an Uninterpreted term"
    e.reset(); e.step(0)
    assert e.x == 2.0


class _ListLiteral(_MinEnv):
    def __init__(self):
        super().__init__(); self.state = [1.0, 2.0]
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(2,))
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.state = [1.0, 2.0]
        return np.array(self.state), 0.0, False, False
    def step(self, action):
        self.state = [self.state[1], self.state[0]]   # swap via index access
        return np.array(self.state), 0.0, False, False

def test_list_literal_and_subscript():
    e = Env(_ListLiteral(), interpret=True); e.reset()
    e.step(0)
    assert torch.allclose(e.state, torch.tensor([2.0, 1.0]))


class _StaticAttrs(_MinEnv):
    def __init__(self):
        super().__init__()
        self.label = "default"   # str → static_attrs
        self.parent = None       # None → static_attrs
        self.x = 0.0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.x = 0.0
        return self.x, 0.0, False, False
    def step(self, action):
        if self.label == "default":
            self.x = self.x + 1.0
        if self.parent is None:
            self.x = self.x * 2.0
        return self.x, 0.0, False, False

def test_static_attrs():
    e = Env(_StaticAttrs(), interpret=True); e.reset(); e.step(0)
    assert e.x == 2.0   # both static comparisons resolve True at compile time


def _external_pair():
    return 1.0, 2.0

class _TupleUnpack(_MinEnv):
    def __init__(self):
        super().__init__(); self.x = 0.0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.x = 0.0
        return self.x, 0.0, False, False
    def step(self, action):
        a, b = (3.0, 4.0)         # tuple literal unpacking
        c, d = _external_pair()   # untraced call returning a tuple
        self.x = a + b + c + d
        return self.x, 0.0, False, False

def test_tuple_unpacking():
    e = Env(_TupleUnpack())   # real env (untraced call)
    e.reset(); e.step(0)
    assert e.x == 10.0   # 3 + 4 + 1 + 2


class _Reassign(_MinEnv):
    def __init__(self):
        super().__init__(); self.x = 0.0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.x = 0.0
        return self.x, 0.0, False, False
    def step(self, action):
        self.x = 1.0
        self.x = 2.0           # re-assignment in the same body
        self.x = self.x + 5.0
        return self.x, 0.0, False, False

def test_reassignment():
    e = Env(_Reassign(), interpret=True); e.reset(); e.step(0)
    assert e.x == 7.0


class _ComplexSig(_MinEnv):
    def __init__(self):
        super().__init__(); self.x = 0.0
    def reset(self, *args, seed: int = None, **kwargs) -> tuple:
        super().reset(seed=seed); self.x = 0.0
        return self.x, 0.0, False, False
    def step(self, action):
        self.x = self.x + 1.0
        return self.x, 0.0, False, False

def test_complex_signature_and_varargs():
    """Abstract interpreter must accept *args/**kwargs and skip complex annotations."""
    e = Env(_ComplexSig(), interpret=True); e.reset(); e.step(0)
    assert e.x == 1.0
