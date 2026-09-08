"""The template tree must have exactly one copy of every Lean source.

`templates/` used to carry a second, byte-identical set of the Core files
(`Mat.lean`, `Box.lean`, `Basic.lean`, `LTL.lean` and the LeanAI tree)
alongside the ones under `templates/static/`. Only the `static/` copies are
ever read — `create_project` resolves them through `STATIC_DIR`, and the
tests/lean fixture syncs from the same place — so edits to the top-level
copies silently did nothing. That is what these tests exist to prevent.
"""

from pathlib import Path

import pytest

from zrth.lean.project import CORE_FILES, LEAN_AI_FILES, TEMPLATE_DIR
from zrth.lean.template_env import TEMPLATES_DIR, STATIC_DIR, PROJECT_TEMPLATES_DIR


def test_every_lean_template_is_reachable_by_the_generator():
    """A `.lean` file outside static/ and project/ can never be emitted."""
    reachable = (STATIC_DIR, PROJECT_TEMPLATES_DIR)
    stray = [
        p.relative_to(TEMPLATES_DIR)
        for p in TEMPLATES_DIR.rglob("*.lean")
        if not any(p.is_relative_to(d) for d in reachable)
    ]
    assert not stray, (
        "these templates are read by nothing, so editing them has no effect: "
        f"{sorted(map(str, stray))}"
    )


@pytest.mark.parametrize("name", CORE_FILES)
def test_core_file_exists_exactly_once(name):
    """One source of truth per Core file, under static/Core/."""
    found = sorted(p.relative_to(TEMPLATES_DIR) for p in TEMPLATES_DIR.rglob(name))
    assert found == [Path("static") / "Core" / name], (
        f"{name} should exist only at static/Core/, found {list(map(str, found))}"
    )


@pytest.mark.parametrize("name", LEAN_AI_FILES)
def test_lean_ai_entry_exists_exactly_once(name):
    """Same for the LeanAI file and directory the generator copies."""
    found = sorted(
        p.relative_to(TEMPLATES_DIR)
        for p in TEMPLATES_DIR.rglob(name)
        if p.name == name
    )
    assert found == [Path("static") / name], (
        f"{name} should exist only under static/, found {list(map(str, found))}"
    )


def test_generator_reads_the_static_tree():
    """Pins the assumption the tests above rely on."""
    assert TEMPLATE_DIR == STATIC_DIR
    for name in CORE_FILES:
        assert (TEMPLATE_DIR / "Core" / name).is_file(), f"Core/{name} missing"
    for name in LEAN_AI_FILES:
        assert (TEMPLATE_DIR / name).exists(), f"{name} missing"
