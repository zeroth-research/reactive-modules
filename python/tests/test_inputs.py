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
    assert problem.source == inputs.SourceText("python", path, path.read_text())


def test_a_missing_file_is_named():
    with pytest.raises(FileNotFoundError, match="nope.py"):
        inputs.load(FIXTURES / "nope.py")


def test_an_unknown_format_is_refused_with_the_known_ones():
    with pytest.raises(KeyError, match="have: python"):
        inputs.adapter_by_name("cobol")
