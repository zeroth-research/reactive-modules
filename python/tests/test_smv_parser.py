"""Tests for the pure-Python SMV parser."""

from pathlib import Path
import pytest

from zrth import Wire, DType, IType, Module, parse_smv

FIXTURES = Path(__file__).parent / "smv_fixtures"


def test_counter_basic():
    """Parse counter.smv — 5 variables (3 VAR + 2 IVAR), check obs pairs."""
    text = (FIXTURES / "counter.smv").read_text()
    module, name_map = parse_smv(text)

    assert isinstance(module, Module)
    assert len(name_map) == 5  # x, y, z, y0, z0

    # All 5 variables appear as obs pairs
    assert len(module.obs) == 5

    # VAR names
    for v in ("x", "y", "z"):
        assert v in name_map
        latched, nxt = name_map[v]
        assert isinstance(latched, Wire)
        assert isinstance(nxt, Wire)
        assert latched != nxt

    # IVAR names
    for v in ("y0", "z0"):
        assert v in name_map


def test_counter_init_terms():
    """counter.smv: init(x) := 0 should produce a ConstInt(0) term."""
    text = (FIXTURES / "counter.smv").read_text()
    module, name_map = parse_smv(text)

    # The module has atoms; first atom covers init/update
    atom = module.atoms[0]
    init_terms = atom.init
    assert len(init_terms) > 0

    # At least one init term should be a zero-valued BV constant for x.
    const_zero_found = any(
        t.itype.op_name == "Const" and int(t.itype.const_data.item()) == 0
        for t in init_terms
    )
    assert const_zero_found, "Expected a BV.Const(0) init term for x"


def test_test_itypes():
    """Parse test_itypes.smv — 6 variables (3 VAR + 3 IVAR)."""
    text = (FIXTURES / "test_itypes.smv").read_text()
    module, name_map = parse_smv(text)

    assert isinstance(module, Module)
    assert len(name_map) == 6  # i, j, a, i0, j0, a0
    assert len(module.obs) == 6


def test_hrm_word_level():
    """Smoke test for the large word-level HRM file with INVAR/TRANS."""
    text = (FIXTURES / "hrm_word_level_no_prop.smv").read_text()
    module, name_map = parse_smv(text)

    assert isinstance(module, Module)
    # Should have many variables
    assert len(name_map) > 50
    assert len(module.obs) == len(name_map)


def test_wire_overrides():
    """Overrides should reuse provided wires instead of creating new ones."""
    text = (FIXTURES / "counter.smv").read_text()

    # Create override wires for 'x'
    # The SMV parser is BV-only; overrides must match.
    dtype = DType.BV(32)
    ov_l = Wire(dtype)
    ov_n = Wire(dtype)

    module, name_map = parse_smv(text, overrides={"x": (ov_l, ov_n)})

    assert name_map["x"] == (ov_l, ov_n)
    # Other variables should get fresh wires
    assert name_map["y"][0] != ov_l


def test_parse_error():
    """Invalid input should raise an error."""
    with pytest.raises(Exception):
        parse_smv("THIS IS NOT VALID SMV")


def test_define_expansion():
    """DEFINE macros should be expanded inline."""
    text = """
    MODULE main
    VAR x : integer;
    DEFINE doubled := x + x;
    ASSIGN
        init(x) := 0;
        next(x) := doubled;
    """
    module, name_map = parse_smv(text)
    assert isinstance(module, Module)
    assert "x" in name_map

    # The update terms should contain Add (from doubled := x + x)
    atom = module.atoms[0]
    update_itypes = [str(t.itype) for t in atom.update]
    assert any("Add" in it for it in update_itypes), f"Expected Add in update terms, got {update_itypes}"


def test_enum_type():
    """Enum types should map to integer, and enum values should be resolvable."""
    text = """
    MODULE main
    VAR state : {idle, running, stopped};
    ASSIGN
        init(state) := idle;
        next(state) := case
            state = idle    : running;
            state = running : stopped;
            TRUE            : state;
        esac;
    """
    module, name_map = parse_smv(text)
    assert isinstance(module, Module)
    assert "state" in name_map
    # state should have TensorInt dtype (enum mapped to int)
    latched, _ = name_map["state"]
    # SMV parser maps `integer` / enum types to BV<32>.
    assert latched.dtype.is_bv()


def test_frozen_var():
    """FROZENVAR should get identity update (next := current)."""
    text = """
    MODULE main
    FROZENVAR c : integer;
    VAR x : integer;
    ASSIGN
        init(x) := c;
        next(x) := x + c;
    """
    module, name_map = parse_smv(text)
    assert isinstance(module, Module)
    assert "c" in name_map
    assert "x" in name_map

    # Should have an Id() term in update for the frozen var
    atom = module.atoms[0]
    update_itypes = [str(t.itype) for t in atom.update]
    assert any("Id" in it for it in update_itypes)
