"""Check that every measured cell of the matrix reaches a column on the page.

This is the one thing the renderer cannot be allowed to get wrong quietly. A
route that fails is a cell the page shows; a *column* that fails to render is
a cell the page does not show at all, and nothing about the output says so --
`houdini-vampire` was 155 measured cells and 112 `VERIFIED` dropped on the
floor for exactly as long as it took someone to go looking for them.

Two things had to agree and did not. A cell is looked up by its key suffix,
but the column list came from `zrth.lean.infer_route.ROUTES` and the presence
test from the `route` field recorded beside each run -- and neither of those
is the column: one route measured over two solvers is two columns sharing a
route name. So the checks here are that the column list covers the harness's
own columns, and that it covers whatever a results file actually holds.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render                                                   # noqa: E402
from suites import column_of, routes                            # noqa: E402


def a_run(route: str) -> dict:
    """The little of a run that `columns` reads."""
    return dict(route=route, verdict="VERIFIED")


def test_every_column_the_harness_measures_has_one_on_the_page():
    names = [r.name for r in routes(proveit_dir="/x", ic3ia="/y", vampire="/z")]
    assert "houdini-vampire" in names and "houdini" in names
    assert set(names) <= set(render.columns({}))


def test_two_solvers_of_one_route_are_two_columns_that_share_its_description():
    docs = {c["name"]: c for c in render.column_docs()}
    cvc5, vamp = docs["houdini"], docs["houdini-vampire"]
    assert cvc5["summary"] == vamp["summary"]       # one route, one description
    assert cvc5["note"] != vamp["note"]             # and each says which half
    assert "vampire" in vamp["note"].lower()


def test_a_run_recorded_under_the_route_name_still_lands_in_its_column():
    # What the stale half of `results.json` looks like: the key says which
    # column, the field beside it says `vampire` because that is what the
    # route was called when the run was made.
    runs = {"limits/m_countdown/Countdown::houdini-vampire": a_run("vampire")}
    assert column_of(next(iter(runs))) == "houdini-vampire"
    assert "houdini-vampire" in render.columns(runs)


def test_a_column_the_harness_no_longer_declares_keeps_its_cells():
    runs = {"limits/m_countdown/Countdown::retired-route": a_run("retired-route")}
    assert "retired-route" in render.columns(runs)


def row_html(run: dict, column: str) -> str:
    out: list = []
    selects = {c["name"]: c["infer"] for c in render.column_docs()}[column]
    render.method_row(out.append, column, run, None, selects)
    return "".join(out)


def a_cell(command: str) -> dict:
    return dict(verdict="VERIFIED", command=command,
                gen=dict(secs=1.0), build=dict(secs=1.0, errors=[]))


def test_a_command_recorded_under_an_old_flag_says_it_no_longer_reproduces():
    # What the 155 cells carry: `--infer vampire` meant houdini-over-Vampire
    # when they were measured, and means a different route now.
    html = row_html(a_cell("uv run verith m.py --buchi --infer vampire "
                           "--vampire /bin/vampire"), "houdini-vampire")
    assert "Recorded before the flags changed" in html
    assert "--infer houdini" in html


def test_a_command_that_still_selects_its_column_is_left_alone():
    html = row_html(a_cell("uv run verith m.py --buchi --infer houdini "
                           "--houdini-solver vampire"), "houdini-vampire")
    assert "Recorded before the flags changed" not in html


def test_a_column_is_named_once_and_selects_one_route():
    cols = routes(proveit_dir="/x", ic3ia="/y", vampire="/z")
    assert len({c.name for c in cols}) == len(cols)
    from zrth.lean.infer_route import ROUTES

    known = {r.name for r in ROUTES}
    for c in cols:
        assert c.infer in known, f"{c.name} selects --infer {c.infer}, which is not a route"
