"""The input adapters: `zrth.lean.inputs`."""

from pathlib import Path

import pytest

from zrth import Module
from zrth.lean import inputs

FIXTURES = Path(__file__).parent / "fixtures"


def test_a_python_file_loads_through_its_adapter():
    path = FIXTURES / "counter.py"
    problem = inputs.load(path)
    assert isinstance(problem.module, Module)
    assert problem.source == inputs.SourceText("python", path, path.read_text(), ".py")


def test_a_missing_file_is_named():
    with pytest.raises(FileNotFoundError, match="nope.py"):
        inputs.load(FIXTURES / "nope.py")


def test_an_unknown_format_is_refused_with_the_known_ones():
    with pytest.raises(KeyError, match="have: python"):
        inputs.adapter_by_name("cobol")


def test_the_project_keeps_the_input_under_its_formats_suffix(tmp_path):
    """`dbg/` holds the input the module was read from, named for its
    format rather than assumed to be Python."""
    from zrth.lean.codegen.context import LeanContext
    from zrth.lean.project.scaffold import create_project
    from zrth.lean.proof.render import certificate_files

    module = inputs.load(FIXTURES / "counter.py").module
    source = inputs.SourceText("smv", tmp_path / "m.txt", "MODULE main\n", ".smv")
    project = create_project(tmp_path, module, "Rea", source=source,
                             certificate=certificate_files(LeanContext(module)))
    assert (project / "dbg" / "system.smv").read_text() == "MODULE main\n"
    assert not (project / "dbg" / "system.py").exists()


SMV = Path(__file__).parent / "smv_fixtures"


def test_an_smv_model_loads_with_its_names():
    problem = inputs.load(SMV / "mod10.smv", format="smv")
    assert problem.names == inputs.Names(("x",), ("inc",))
    assert problem.source.suffix == ".smv"
    assert inputs.adapter_for("m.smv").name == "smv"


def test_a_model_smv_would_read_otherwise_is_an_input_error(tmp_path):
    free = tmp_path / "free.smv"
    free.write_text("MODULE main\nVAR x : boolean;\nASSIGN init(x) := TRUE;\n")
    with pytest.raises(inputs.InputError, match="`x` has no next value"):
        inputs.load(free, format="smv")
    garbage = tmp_path / "garbage.smv"
    garbage.write_text("this is not SMV\n")
    with pytest.raises(inputs.InputError, match="is not SMV verith reads"):
        inputs.load(garbage, format="smv")
