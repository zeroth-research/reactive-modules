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


def test_every_project_template_is_rendered():
    """A `.j2` that no `render()` call names drifts from the code that prints it."""
    lean_pkg = TEMPLATES_DIR.parent
    src = "\n".join(p.read_text() for p in lean_pkg.rglob("*.py"))
    unused = sorted(
        rel
        for p in PROJECT_TEMPLATES_DIR.rglob("*.j2")
        if f'"{(rel := p.relative_to(TEMPLATES_DIR).as_posix())}"' not in src
    )
    assert not unused, f"these templates are rendered by nothing: {unused}"


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


# ══════════════════════════════════════════════════════════════════════
#  The reductions have to stay computable under the certificate's `simp`
# ══════════════════════════════════════════════════════════════════════

REDUCTIONS = ("matMin", "matMax", "argmax_1d", "argmax")


def _mat_source() -> str:
    return (STATIC_DIR / "Core" / "Mat.lean").read_text()


def test_the_reductions_enumerate_with_ofFn_not_finRange():
    """`List.finRange` is what made these unprovable for a year.

    `simp_mat` carries `List.ofFn_succ` and `List.ofFn_zero` and no lemma
    that computes `List.finRange`, so a `finRange` fold stays stuck in the
    goal at every concrete shape and every closer fails on it; `be70a43`
    moved them to `List.ofFn`. Rewriting one of these back to `finRange`
    would look like a simplification and would silently cost twelve
    certificates.
    """
    src = _mat_source()
    for name in REDUCTIONS:
        body = src.split(f"def {name} ", 1)[1].split("\ndef ", 1)[0]
        assert "List.ofFn" in body, f"{name} no longer enumerates with List.ofFn"
        assert "List.finRange" not in body, (
            f"{name} enumerates with List.finRange, which `simp_mat` cannot "
            f"compute (fixed in be70a43)"
        )


def _counter_ctx():
    from zrth.lean.context import LeanContext
    from zrth.lean.inputs.python import load_module

    fixture = Path(__file__).parent / "fixtures" / "counter.py"
    return LeanContext(load_module(fixture))


def test_the_certificate_simp_set_unfolds_them():
    """Computing the fold is no use if the definition never opens."""
    from zrth.lean.proof.render import generate_certificate_lean

    cert = generate_certificate_lean(_counter_ctx())
    simp_mat = [ln for ln in cert.splitlines()
                if "simp [" in ln and "MatAdd_apply" in ln]
    assert simp_mat, "the simp_mat macro moved"
    for name in REDUCTIONS:
        assert name in simp_mat[0], f"{name} is not in simp_mat"


def test_the_fbk_bridge_simp_set_agrees_with_it():
    """The bridge proves the NA model is the module, over the same terms."""
    from zrth.lean.translate.fbk import check_na_supported
    from zrth.lean.translate.fbk_bridge import atom_to_lean_fbk_bridge

    ctx = _counter_ctx()
    bridge = atom_to_lean_fbk_bridge(ctx, na_module="NA",
                                     bodies=check_na_supported(ctx))
    cascade = [ln for ln in bridge.splitlines() if "MatAdd_apply" in ln]
    assert cascade, "the bridge's cascade has no matrix simp set"
    for name in REDUCTIONS:
        assert name in cascade[0], f"{name} is not in the fbk bridge's simp set"
