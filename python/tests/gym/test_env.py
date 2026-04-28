"""Tests for the merged Env class — covers the constructor's expanded shape,
the new pure-symbolic runtime branch, composition, and the small public surface
(get, state_dict, get_prvt, attribute access) added in the Wrapper→Env merge."""

import pytest
import torch

from .environments import SimpleEnv, GridWorldEnv
from .qnetworks import SimpleQNet
from zrth.gym import Env
from zrth.torch import Module
from zrth import Wire, DType


# ── Constructor / shape ───────────────────────────────────────────

def test_env_with_gym_env():
    e = Env(SimpleEnv())
    assert e._backing_env is not None
    assert e._env_atom_idx is not None
    assert len(e.obs) == 5  # action, observation, reward, terminated, truncated


def test_env_with_module_only():
    e = Env(Module(SimpleQNet(1, 2, 2)))
    assert e._backing_env is None
    assert e._env_atom_idx is None


def test_env_composes_two_modules():
    e = Env(Module(SimpleQNet(1, 2, 2)), Module(SimpleQNet(1, 2, 2)))
    assert e._backing_env is None
    assert len(e.atoms) == 2


def test_env_extract_and_compose():
    plain = SimpleEnv()
    e = Env(plain, Module(SimpleQNet(1, plain.action_space.n, 2)))
    assert e._backing_env is plain
    assert e._env_atom_idx is not None
    assert len(e.atoms) >= 2


def test_env_unwraps_existing_env():
    plain = SimpleEnv()
    inner = Env(plain)
    outer = Env(inner, Module(SimpleQNet(1, plain.action_space.n, 2)))
    assert outer._backing_env is plain


# ── Errors ────────────────────────────────────────────────────────

def test_env_rejects_multiple_gym_envs():
    with pytest.raises(TypeError, match="at most 1"):
        Env(SimpleEnv(), GridWorldEnv())


def test_env_rejects_wrapped_plus_raw():
    with pytest.raises(TypeError, match="at most 1"):
        Env(Env(SimpleEnv()), GridWorldEnv())


def test_env_rejects_invalid_arg():
    with pytest.raises(TypeError, match="Expected gym.Env or Module"):
        Env(42)


def test_env_rejects_no_args():
    with pytest.raises(TypeError, match="at least 1"):
        Env()


# ── Runtime ───────────────────────────────────────────────────────

def test_pure_symbolic_reset_needs_inputs():
    """Pure-symbolic Env(Module) has no source for the module's input wires;
    reset surfaces this as a KeyError rather than silently producing garbage."""
    e = Env(Module(SimpleQNet(1, 2, 2)))
    with pytest.raises(KeyError):
        e.reset()


def test_step_matches_backing_env():
    raw, wrapped = SimpleEnv(), Env(SimpleEnv())
    raw.reset(seed=42)
    wrapped.reset(seed=42)
    raw.step(torch.tensor([0., 1.]))   # one-hot "right", same as wrapped's _prepare_action(1)
    wrapped.step(1)
    assert raw.state == wrapped.state


def test_discrete_action_one_hot():
    e = Env(GridWorldEnv())
    e.reset(seed=42)
    e.step(2)
    assert torch.allclose(e._state[e._pairs['action'][0]], torch.tensor([0., 0., 1., 0.]))


def test_get_prvt_and_getattr():
    e = Env(SimpleEnv())
    e.reset(seed=42)
    assert e.state == 0  # SimpleEnv.reset sets state=0; routed via getattr_wire
    assert len(e.get_prvt('state')) == 2  # (latched, next)


# ── Composition ───────────────────────────────────────────────────

def test_env_atom_idx_identifies_env_atom():
    e = Env(SimpleEnv(), Module(SimpleQNet(1, 2, 2)))
    obs_next = e._pairs['observation'][1]
    assert obs_next in {w for w in e.atoms[e._env_atom_idx].ctrl}


def test_wire_names_inherited_through_unwrap():
    plain = SimpleEnv()
    inner = Env(plain)
    outer = Env(inner, Module(SimpleQNet(1, plain.action_space.n, 2)))
    assert outer._wire_names == inner._wire_names


# ── Public surface ────────────────────────────────────────────────

def test_get_wire():
    e = Env(SimpleEnv())
    e.reset(seed=42)
    val = e.get(e._pairs['observation'][0])
    assert isinstance(val, torch.Tensor)
    with pytest.raises(RuntimeError, match="not in state"):
        e.get(Wire(DType.Float([1])))


def test_state_dict_is_a_copy():
    e = Env(SimpleEnv())
    e.reset(seed=42)
    sd = e.state_dict()
    sd['fake'] = 'fake'
    assert 'fake' not in e._state


# ── symbolic=True flag ────────────────────────────────────────────

def test_symbolic_flag_strips_backing():
    e = Env(SimpleEnv(), symbolic=True)
    assert e._backing_env is None
    assert e._env_atom_idx is None


def test_symbolic_reset_runs_atoms():
    """symbolic=True executes the env atom's symbolic init terms instead of
    delegating; SimpleEnv's init sets state=0."""
    e = Env(SimpleEnv(), symbolic=True)
    e.reset()
    assert e.state == 0


def test_symbolic_step_matches_real():
    """Cross-check: a delegated Env and a symbolic-mode Env over the same gym.Env
    should agree step-by-step when fed identical actions."""
    real = Env(SimpleEnv())
    sim = Env(SimpleEnv(), symbolic=True)
    real.reset(seed=42)
    sim.reset()
    one_hot_right = torch.tensor([0., 1.])
    for _ in range(3):
        real.step(one_hot_right)
        sim.step(one_hot_right)
        assert real.state == sim.state
