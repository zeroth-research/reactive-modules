"""`zrth.lean.project.names` is what the project is written under: every
module the generated Lean imports from the project, the bridge's
included, is a file the project has."""

import re
from pathlib import Path

from zrth.lean.codegen.context import LeanContext
from zrth.lean.inputs.python import load_module
from zrth.lean.project.names import CERT_LIB, SYSTEM_LAYOUT
from zrth.lean.project.scaffold import create_project
from zrth.lean.infer.fbk_proveit.na import check_na_supported
from zrth.lean.infer.fbk_proveit.bridge import atom_to_lean_fbk_bridge

FIXTURES = Path(__file__).parent / "fixtures"


def _own_imports(text: str) -> list[str]:
    mods = re.findall(r"^import (\S+)", text, re.M)
    return [m for m in mods
            if m.startswith(SYSTEM_LAYOUT.prefix) or m.startswith(f"{CERT_LIB}.")]


def test_every_project_module_imported_is_a_file_the_project_has(tmp_path):
    module = load_module(FIXTURES / "counter.py")
    project = create_project(tmp_path, module, "Rea")
    ctx = LeanContext(module)
    bridge = atom_to_lean_fbk_bridge(ctx, na_module="ReaNA",
                                     bodies=check_na_supported(ctx))
    texts = {
        "bridge": bridge,
        "certificate": (project / CERT_LIB / "Certificate.lean").read_text(),
        "root": (project / f"{CERT_LIB}.lean").read_text(),
    }
    for who, text in texts.items():
        mods = _own_imports(text)
        assert mods or who == "root", f"{who} imports nothing of the project"
        for mod in mods:
            path = project / (mod.replace(".", "/") + ".lean")
            assert path.exists(), f"{who} imports {mod}, which is not written"


def test_the_lakefile_builds_the_certificate_library(tmp_path):
    module = load_module(FIXTURES / "counter.py")
    project = create_project(tmp_path, module, "Rea")
    lakefile = (project / "lakefile.toml").read_text()
    assert f'name = "{CERT_LIB}"' in lakefile
    assert re.search(rf'defaultTargets = \[[^\]]*"{CERT_LIB}"', lakefile)
