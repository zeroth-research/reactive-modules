"""Tests for the Env class: constructor shape, runtime delegation vs.
interpretation, composition, error paths, and the small public surface
(`get`, `state_dict`, `get_prvt`, attribute access)."""

import pytest
import torch
import gymnasium as gym
from gymnasium import spaces

from .environments import SimpleEnv
from .qnetworks import SimpleQNet
from zrth.gym import Env
from zrth.torch import Module
from zrth import Wire, Real, Int, Var, X

# SimpleEnv's only private state is `state`; its sort must be declared explicitly.
STATE = Real([1, 1])
real = Real([1, 1])


# ── Constructor / shape ───────────────────────────────────────────

def test_env_with_gym_env():
    e = Env(SimpleEnv(), attrs=STATE)
    assert e._backing_env is not None
    assert e._env_atom_idx is not None
    assert len(e.obs) == 5  # action, observation, reward, terminated, truncated


def test_env_with_module_only():
    e = Env(Module(SimpleQNet(1, 2, 2), combinatorial=True))
    assert e._backing_env is None
    assert e._env_atom_idx is None


def test_env_composes_two_modules():
    e = Env(Module(SimpleQNet(1, 2, 2), combinatorial=True), Module(SimpleQNet(1, 2, 2), combinatorial=True))
    assert e._backing_env is None
    assert len(e.atoms) == 2


def test_env_extract_and_compose():
    plain = SimpleEnv()
    e = Env(plain, Module(SimpleQNet(1, plain.action_space.n, 2), combinatorial=True), attrs=STATE)
    assert e._backing_env is plain
    assert e._env_atom_idx is not None
    assert len(e.atoms) >= 2


def test_env_unwraps_existing_env():
    plain = SimpleEnv()
    inner = Env(plain, attrs=STATE)
    outer = Env(inner, Module(SimpleQNet(1, plain.action_space.n, 2), combinatorial=True))
    assert outer._backing_env is plain


# ── Errors ────────────────────────────────────────────────────────

def test_env_rejects_multiple_gym_envs():
    with pytest.raises(ValueError, match="at most 1"):
        Env(SimpleEnv(), SimpleEnv())


def test_env_rejects_wrapped_plus_raw():
    with pytest.raises(ValueError, match="at most 1"):
        Env(Env(SimpleEnv(), attrs=STATE), SimpleEnv())


def test_env_rejects_invalid_arg():
    with pytest.raises(TypeError, match="Expected gym.Env or Module"):
        Env(42)


def test_env_rejects_no_args():
    with pytest.raises(TypeError, match="at least 1"):
        Env()


# ── attrs override (private-state sorts/wires) ────────────────────

def test_attrs_dict_sort_builds():
    e = Env(SimpleEnv(), attrs={"state": Real([1, 1])})
    assert e.get_prvt("state").dtype == Real([1, 1])


def test_attrs_single_sort_applies_to_all():
    e = Env(SimpleEnv(), attrs=Real([1, 1]))
    assert e.get_prvt("state").dtype == Real([1, 1])


def test_attrs_wire_pair_used_verbatim():
    pair = Var(real)
    e = Env(SimpleEnv(), attrs={"state": pair})
    got = e.get_prvt("state")
    assert got == pair and X(got) == X(pair)


def test_attrs_rejects_unknown_name():
    with pytest.raises(ValueError, match="not private-state"):
        Env(SimpleEnv(), attrs={"not_an_attr": Real([1, 1])})


def test_attrs_rejects_bad_spec_type():
    with pytest.raises(TypeError, match="must be a Sort"):
        Env(SimpleEnv(), attrs={"state": 42})


def test_attrs_without_gym_env_rejected():
    with pytest.raises(TypeError, match="only applies when extracting"):
        Env(Module(SimpleQNet(1, 2, 2), combinatorial=True), attrs=Real([1, 1]))


def test_undeclared_private_state_raises():
    # Private-state sorts are never inferred: omitting them (or attrs entirely) errors.
    with pytest.raises(ValueError, match="must be declared"):
        Env(SimpleEnv())
    with pytest.raises(ValueError, match="must be declared"):
        Env(SimpleEnv(), attrs={})


# ── Runtime ───────────────────────────────────────────────────────

def test_pure_symbolic_reset_needs_inputs():
    """Pure-symbolic Env(Module) has no source for the module's input wires;
    reset surfaces this as a KeyError rather than silently producing garbage."""
    e = Env(Module(SimpleQNet(1, 2, 2), combinatorial=True))
    with pytest.raises(KeyError):
        e.reset()


def test_step_matches_backing_env():
    raw, wrapped = SimpleEnv(), Env(SimpleEnv(), attrs=STATE)
    raw.reset(seed=42)
    wrapped.reset(seed=42)
    raw.step(torch.tensor([0., 1.]))  # one-hot "right", same as wrapped's _prepare_action(1)
    wrapped.step(1)
    assert raw.state == wrapped.state


def test_discrete_action_one_hot():
    e = Env(SimpleEnv(), attrs=STATE)
    e.reset(seed=42)
    e.step(1)
    assert torch.allclose(e._state[e._pairs['action']], torch.tensor([0., 1.]))


def test_get_prvt_and_getattr():
    e = Env(SimpleEnv(), attrs=STATE)
    e.reset(seed=42)
    assert e.state == 0  # SimpleEnv.reset sets state=0; routed via getattr_wire


# ── Composition ───────────────────────────────────────────────────

def test_env_atom_idx_identifies_env_atom():
    e = Env(SimpleEnv(), Module(SimpleQNet(1, 2, 2), combinatorial=True), attrs=STATE)
    obs_next = X(e._pairs['observation'])
    assert obs_next in {X(w) for w in e.atoms[e._env_atom_idx].ctrl}


def test_wire_names_inherited_through_unwrap():
    plain = SimpleEnv()
    inner = Env(plain, attrs=STATE)
    outer = Env(inner, Module(SimpleQNet(1, plain.action_space.n, 2), combinatorial=True))
    assert outer._wire_names == inner._wire_names


# ── Public surface ────────────────────────────────────────────────

def test_get_wire():
    e = Env(SimpleEnv(), attrs=STATE)
    e.reset(seed=42)
    val = e.get(e._pairs['observation'])
    assert isinstance(val, torch.Tensor)
    with pytest.raises(RuntimeError, match="not in state"):
        e.get(Wire(real))


def test_state_dict_is_a_copy():
    e = Env(SimpleEnv(), attrs=STATE)
    e.reset(seed=42)
    sd = e.state_dict()
    sd['fake'] = 'fake'
    assert 'fake' not in e._state


# ── interpret=True flag ───────────────────────────────────────────

def test_interpret_flag_strips_backing():
    e = Env(SimpleEnv(), attrs=STATE, interpret=True)
    assert e._backing_env is None
    assert e._env_atom_idx is None


def test_interpret_reset_runs_atoms():
    """interpret=True walks the env atom's init terms through the IR interpreter
    instead of delegating; SimpleEnv's init sets state=0."""
    e = Env(SimpleEnv(), attrs=STATE, interpret=True)
    e.reset()
    assert e.state == 0


def test_interpret_step_matches_real():
    """Cross-check: a delegated Env and an interpreted Env over the same gym.Env
    should agree step-by-step when fed identical actions."""
    real = Env(SimpleEnv(), attrs=STATE)
    sim = Env(SimpleEnv(), attrs=STATE, interpret=True)
    real.reset(seed=42)
    sim.reset()
    one_hot_right = torch.tensor([0., 1.])
    for _ in range(3):
        real.step(one_hot_right)
        sim.step(one_hot_right)
        assert real.state == sim.state


# ── _action_driven (closed-loop runtime) ──────────────────────────

class _BoxEnv(gym.Env):
    """Box obs (matches Linear input) + Discrete(2) action — for closed-loop tests."""

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(1,))
        self.x = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed);
        self.x = 0.0
        return self.x, 0.0, False, False

    def step(self, action):
        idx = action.argmax().item()
        self.x = self.x + (1.0 if idx == 1 else -1.0)
        return self.x, 0.0, False, False


def _make_closed_loop():
    obs = Var(Real([1, 1]))
    act = Var(Real([1, 2]))
    backed = Env(_BoxEnv(), observation=obs, action=act, attrs=Real([1, 1]))
    qnet = Module(SimpleQNet(1, 2, 2), combinatorial=True, extl=obs, ctrl=act)
    return Env(backed, qnet)


def test_action_driven_true_with_controller():
    env = _make_closed_loop()
    assert env._action_driven is True


def test_action_driven_false_solo():
    assert Env(SimpleEnv(), attrs=STATE)._action_driven is False


def test_step_ignores_action_when_driven():
    """Different junk actions should produce identical trajectories — the composed
    controller drives, the user-passed action argument is ignored."""
    a = _make_closed_loop();
    a.reset(seed=42)
    b = _make_closed_loop();
    b.reset(seed=42)
    for _ in range(3):
        a.step(torch.tensor([99.0, -99.0]))
        b.step(torch.tensor([-99.0, 99.0]))
        assert a.x == b.x
