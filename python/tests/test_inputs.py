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
    from zrth.lean.project import create_project

    module = inputs.load(FIXTURES / "counter.py").module
    source = inputs.SourceText("smv", tmp_path / "m.txt", "MODULE main\n", ".smv")
    project = create_project(tmp_path, module, "Rea", source=source)
    assert (project / "dbg" / "system.smv").read_text() == "MODULE main\n"
    assert not (project / "dbg" / "system.py").exists()
