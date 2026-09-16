"""The inference-route table (`zrth.lean.infer_route`) and the CLI over it.

Two things are checked here. The table is checked for *completeness*, the way
`test_lean_ops.py` checks the op matrix: a row that declares something the
rest of the package cannot honour fails here rather than four steps later in
generated code. And the grammar is checked for *preservation*: every
cross-flag rule `main.py` spells out by hand has a case below, so the table
driving them is the same grammar and not merely a similar one.

Nothing here runs a route. Constructing the certificate is the routes' own
tests; this is about what the CLI accepts and why it refuses.
"""

from __future__ import annotations

import pytest

from zrth.lean.cli import build_parser, parse_args
from zrth.lean.common import Refused
from zrth.lean.infer_route import (
    ROUTES,
    InferResult,
    ProjectHandle,
    all_route_aliases,
    all_route_options,
    route_by_name,
    route_names,
)
from zrth.lean.project import ENCODINGS

KINDS = {"safety", "buchi"}
RETURNS = {"smt", "lean", "installed"}
SEEDS = {"inv", "ranking", "pre"}


def err(argv: list[str]) -> str:
    """The message `parse_args` exits with, or "" if it accepted the line."""
    parser_err: list[str] = []

    class Caught(SystemExit):
        pass

    import argparse

    original = argparse.ArgumentParser.error

    def capture(self, message):          # noqa: ANN001
        parser_err.append(message)
        raise Caught(2)

    argparse.ArgumentParser.error = capture
    try:
        parse_args(argv)
        return ""
    except SystemExit:
        return parser_err[0] if parser_err else ""
    finally:
        argparse.ArgumentParser.error = original


# ══════════════════════════════════════════════════════════════════════════
# The table
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("route", ROUTES, ids=route_names())
def test_row_is_well_formed(route):
    assert route.run is not None, "a row with no `run` is a route with no route"
    assert route.kinds and route.kinds <= KINDS
    assert route.returns in RETURNS
    assert route.seeds <= SEEDS
    assert route.summary and not route.summary.endswith(".")


@pytest.mark.parametrize("route", ROUTES, ids=route_names())
def test_a_partial_row_says_why_it_is_partial(route):
    """A route that refuses a kind or a seed has to have prose for it.

    The check is generic; the reason cannot be. A row that declares
    `kinds={"buchi"}` with no `kinds_refusal` would refuse `--safety` with a
    sentence assembled from set arithmetic, which is exactly the quality the
    hand-written errors have and the table must not lose.
    """
    if route.kinds != KINDS:
        assert route.kinds_refusal, "refusing a proof rule needs a reason"
    if route.seeds != SEEDS:
        assert route.seeds_refusal, "discarding a supplied predicate needs a reason"


@pytest.mark.parametrize("route", ROUTES, ids=route_names())
def test_row_reads_and_owns_are_real(route):
    suffixes = {e.suffix for e in ENCODINGS}
    assert set(route.reads) <= suffixes, "an encoding no table row emits"
    for owned in route.owns:
        assert not owned.startswith("/") and ".." not in owned, (
            "an owned path is project-relative"
        )


def test_row_promises_match_what_it_returns():
    """`returns="installed"` is the only shape that writes its own certificate."""
    for route in ROUTES:
        if route.returns == "installed":
            assert route.owns, "a route that installs a certificate writes files"
        else:
            assert not route.lakefile, (
                "a route that hands back predicates needs no lakefile of its own"
            )


def test_route_options_do_not_collide():
    dests = [o.dest for _, o in all_route_options()]
    assert len(dests) == len(set(dests)), "two routes claiming one flag"


def test_parser_offers_exactly_the_rows():
    action = next(
        a for a in build_parser()._actions if "--infer" in a.option_strings
    )
    assert tuple(action.choices) == route_names()


def test_every_route_option_is_reachable():
    """Each route-scoped flag parses when its route is selected."""
    parser = build_parser()
    known = {a.dest for a in parser._actions}
    for _, opt in all_route_options():
        assert opt.dest in known


# ══════════════════════════════════════════════════════════════════════════
# The rules `main.py` writes by hand
# ══════════════════════════════════════════════════════════════════════════


def test_safety_and_buchi_are_the_choice():
    assert "mutually exclusive" in err(
        ["m.py", "--safety", "(= s0 0)", "--buchi", "(= s0 0)"]
    )


def test_ranking_is_meaningless_with_safety():
    assert "--ranking is meaningless with --safety" in err(
        ["m.py", "--safety", "(= s0 0)", "--ranking", "s0"]
    )


def test_a_blank_option_is_the_error():
    assert "empty value" in err(["m.py", "--buchi", ""])


@pytest.mark.parametrize("flag", ["--smt-timeout", "--smt-budget"])
def test_a_budget_is_positive(flag):
    assert "positive number of milliseconds" in err(
        ["m.py", "--buchi", "(= s0 0)", flag, "0"]
    )


@pytest.mark.parametrize("route", ROUTES, ids=route_names())
def test_a_route_needs_a_property_and_names_the_kinds_it_takes(route):
    """A route that only certifies `G P` must not ask for `--buchi` and then
    refuse it."""
    argv = ["m.py", "--infer", route.name]
    for opt in route.options:
        if opt.required:
            argv += [opt.flags[-1], "/nowhere"]
    message = err(argv)
    assert f"--infer {route.name} requires" in message
    for kind in KINDS:
        assert (f"--{kind}" in message) == (kind in route.kinds)


def test_ai_does_not_infer_a_safety_certificate():
    """`main.py:616`, from the row's `kinds` + `kinds_refusal`."""
    message = err(["m.py", "--safety", "(= s0 0)", "--infer", "ai"])
    assert "rule_globally" in message and "ai-cegar" in message


@pytest.mark.parametrize(
    "flag,value",
    [("--invariant", "(= s0 1)"), ("--ranking", "s0"), ("--pre", "true")],
)
def test_nuterm_refuses_what_it_would_discard(flag, value):
    """`main.py:627`, from the row's `seeds` + `seeds_refusal`."""
    message = err(["m.py", "--buchi", "(= s0 0)", "--infer", "nuterm", flag, value])
    assert flag in message and "comes from the learner" in message


@pytest.mark.parametrize(
    "extra", [["--model", "qwen3"], ["--base-url", "http://localhost:11434/v1"]]
)
def test_the_llm_flags_belong_to_the_llm_routes(extra):
    message = err(["m.py", "--buchi", "(= s0 0)", "--infer", "nuterm", *extra])
    assert "for the LLM routes" in message


def test_fbk_proveit_is_a_safety_question():
    """`main.py:523`, now the row's `kinds_refusal`."""
    message = err(
        [
            "m.py",
            "--buchi",
            "(= s0 0)",
            "--infer",
            "fbk-proveit",
            "--proveit-dir",
            "/nowhere",
        ]
    )
    assert "ic3ia" in message and "safety question" in message


def test_fbk_proveit_refuses_a_supplied_invariant():
    message = err(
        [
            "m.py",
            "--safety",
            "(= s0 0)",
            "--infer",
            "fbk-proveit",
            "--proveit-dir",
            "/nowhere",
            "--invariant",
            "(= s0 1)",
        ]
    )
    assert "--invariant" in message and "comes from ic3ia" in message


def test_fbk_proveit_needs_a_lean_identifier_for_p():
    """`main.py:530` -- the row's `extra_check`, not a cell of the table."""
    message = err(
        [
            "m.py",
            "--safety",
            "(= s0 0)",
            "-p",
            "1proj",
            "--infer",
            "fbk-proveit",
            "--proveit-dir",
            "/nowhere",
        ]
    )
    assert "Lean identifier" in message and "1projNA" in message


def test_fbk_proveit_needs_its_checkout():
    assert "--proveit-dir" in err(["m.py", "--safety", "(= s0 0)", "--infer", "fbk-proveit"])


@pytest.mark.parametrize(
    "extra",
    [
        ["--ic3ia", "/bin/true"],
        ["--fbk-simplify", "none"],
        ["--fbk-equiv", "none"],
        ["--proveit-dir", "/nowhere"],
    ],
)
def test_a_route_option_without_its_route(extra):
    """`main.py:561` -- one rule for every row's options, not three by hand."""
    message = err(["m.py", "--buchi", "(= s0 0)", *extra])
    assert "only meaningful together with --infer fbk-proveit" in message


@pytest.mark.parametrize("route", ROUTES, ids=route_names())
def test_no_route_writes_a_standalone_certificate(route):
    """`--cert-file` writes a file and returns before inference could run.

    True of every route, including one that declares it neither reads nor
    writes the project: `_standalone` returns first either way, so a route
    allowed through here would be silently skipped.
    """
    kind = "safety" if "safety" in route.kinds else "buchi"
    argv = ["m.py", f"--{kind}", "(= s0 0)", "--infer", route.name,
            "--cert-file", "C.lean"]
    for opt in route.options:
        if opt.required:
            argv += [opt.flags[-1], "/nowhere"]
    assert "--cert-file" in err(argv)


def test_build_cert_needs_a_certificate():
    assert "--build-cert needs a certificate" in err(
        ["m.py", "--buchi", "(= s0 0)", "--build-cert"]
    )


def test_build_cert_is_satisfied_by_supplied_predicates():
    settings = parse_args(
        [
            "m.py",
            "--buchi",
            "(= s0 0)",
            "--invariant",
            "(<= s0 100)",
            "--ranking",
            "s0",
            "--build-cert",
        ]
    )
    assert settings.supplied_complete and settings.route is None


def test_build_cert_with_a_file_route():
    assert "--build-cert is incompatible with --cert-file" in err(
        ["m.py", "--buchi", "(= s0 0)", "--cert-file", "C.lean", "--build-cert"]
    )


def test_hammer_file_needs_no_module():
    settings = parse_args(["--hammer-file", "H.lean"])
    assert settings.module_file is None and settings.hammer_file == "H.lean"


def test_a_module_file_is_otherwise_required():
    assert "module_file is required" in err(["--buchi", "(= s0 0)"])


# ══════════════════════════════════════════════════════════════════════════
# The legacy spelling
# ══════════════════════════════════════════════════════════════════════════


def _checkout(tmp_path):
    """A directory `resolve_project` accepts."""
    for rel in ("proveit.py", "vmt2lean.py", "lakefile.toml"):
        (tmp_path / rel).write_text("")
    (tmp_path / "LTLCertifying").mkdir()
    (tmp_path / "LTLCertifying" / "lean2vmt.lean").write_text("")
    return tmp_path


def test_fbk_proveit_flag_still_selects_the_route(tmp_path):
    root = _checkout(tmp_path)
    settings = parse_args(
        ["m.py", "--safety", "(= s0 0)", "--fbk-proveit", str(root)]
    )
    assert settings.route.name == "fbk-proveit"
    assert settings.route_opts["proveit_dir"] == str(root)
    # `resolve` ran at parse time: the lakefile written during generation
    # names this path, so a checkout that is not one fails the CLI.
    assert settings.route_config == root.resolve()


def test_the_legacy_flag_and_a_different_route_conflict(tmp_path):
    message = err(
        [
            "m.py",
            "--safety",
            "(= s0 0)",
            "--infer",
            "ai-cegar",
            "--fbk-proveit",
            str(_checkout(tmp_path)),
        ]
    )
    assert "a run takes one route" in message


def test_a_bad_checkout_fails_the_cli(tmp_path):
    message = err(
        ["m.py", "--safety", "(= s0 0)", "--fbk-proveit", str(tmp_path / "nope")]
    )
    assert "not a directory" in message


def test_route_option_defaults_survive(tmp_path):
    settings = parse_args(
        ["m.py", "--safety", "(= s0 0)", "--fbk-proveit", str(_checkout(tmp_path))]
    )
    assert settings.route_opts["fbk_simplify"] == "cvc5"
    assert settings.route_opts["fbk_equiv"] == "lean"
    assert settings.route_opts["ic3ia"] is None


# ══════════════════════════════════════════════════════════════════════════
# The handle
# ══════════════════════════════════════════════════════════════════════════


def _store(tmp_path, **kw):
    from zrth.lean.artifacts import ArtifactStore

    kw.setdefault("module_digest", "aaaaaaaaaaaa")
    kw.setdefault("kind", "buchi")
    kw.setdefault("prp", "(= s0 0)")
    kw.setdefault("producer", "nuterm")
    return ArtifactStore(dir=tmp_path / "artifacts", **kw)


def _handle(tmp_path, reads=(), owns=(), resumes=(), store=None):
    return ProjectHandle(
        dir=tmp_path,
        name="Rea",
        module=None,
        ctx=None,
        reads=frozenset(reads),
        owns=tuple(owns),
        resumes=frozenset(resumes),
        artifacts=store or _store(tmp_path),
    )


def test_an_undeclared_encoding_is_refused(tmp_path):
    with pytest.raises(Refused, match="did not declare"):
        _handle(tmp_path).encoding("")


def test_an_encoding_that_does_not_exist_is_refused(tmp_path):
    with pytest.raises(Refused, match="no 'Nope' encoding"):
        _handle(tmp_path, reads=("Nope",)).encoding("Nope")


def test_a_declared_encoding_is_read_by_encoding_not_by_path(tmp_path):
    (tmp_path / "System").mkdir()
    (tmp_path / "System" / "System.lean").write_text("-- functional\n")
    assert "functional" in _handle(tmp_path, reads=("",)).encoding("")


def test_an_undeclared_path_is_refused(tmp_path):
    with pytest.raises(Refused, match="not among this route's declared paths"):
        _handle(tmp_path, owns=("ProveIt/",)).path("Certificate/Certificate.lean")


def test_a_declared_path_is_made(tmp_path):
    handle = _handle(tmp_path, owns=("ProveIt/", "Certificate/Certificate.lean"))
    out = handle.path("ProveIt/ReaNA.lean")
    assert out.parent.is_dir() and out.parent == tmp_path / "ProveIt"
    assert handle.path("Certificate/Certificate.lean").parent.is_dir()


def test_the_declared_prefix_is_a_directory_not_a_string(tmp_path):
    """`owns=("ProveIt/",)` must not admit `ProveItElse/x`."""
    with pytest.raises(Refused):
        _handle(tmp_path, owns=("ProveIt/",)).path("ProveItElse/x.lean")


# ══════════════════════════════════════════════════════════════════════════
# The result
# ══════════════════════════════════════════════════════════════════════════


def test_an_empty_result_is_what_an_installing_route_returns():
    """`empty` is what `main` checks the row's `returns` against."""
    assert InferResult().empty
    assert not InferResult(inv_smt="(= s0 0)").empty
    assert not InferResult(inv_lean="fun s => s.1 = 0").empty


def _run_infer(monkeypatch, tmp_path, result, *, returns="smt", kind="buchi"):
    """Put one synthetic row through `main._infer`, counting renders."""
    from zrth.lean import main as m
    from zrth.lean.cert import CertificateData
    from zrth.lean.infer_route import InferRoute

    rendered = []

    def _render(cd, module, **kw):
        rendered.append(cd)
        return CertificateData(inv="RENDERED_INV", ranking="RENDERED_RANK")

    monkeypatch.setattr(m, "smt_predicates_to_lean", _render)
    settings = parse_args(["m.py", f"--{kind}", "(= s0 0)"])
    settings.route = InferRoute(
        name="probe",
        summary="a synthetic row",
        kinds=frozenset({kind}),
        kinds_refusal="",
        seeds=frozenset(),
        returns=returns,
        run=lambda inp: result,
    )
    out = m._infer(
        settings,
        None,
        _handle(tmp_path),
        CertificateData(prp="(= s0 0)", kind=kind),
        None,
    )
    return out, rendered


def test_lean_that_came_back_is_not_rendered_again(monkeypatch, tmp_path):
    """Rendering encodes the module into cvc5 again, so a route that already
    has the Lean -- both `ai-cegar` and `nuterm` do -- is taken at its word.
    """
    out, rendered = _run_infer(
        monkeypatch,
        tmp_path,
        InferResult(
            inv_smt="(= s0 0)",
            ranking_smt="s0",
            inv_lean="fun s => s.1 = 0",
            ranking_lean="fun s => s.1",
        ),
    )
    assert rendered == [], "the pipeline re-rendered Lean it was handed"
    assert out.inv == "fun s => s.1 = 0" and out.ranking == "fun s => s.1"
    # cvc5's own input is kept either way: nothing parses the Lean back.
    assert out.inv_smt == "(= s0 0)" and out.ranking_smt == "s0"


def test_smt_with_no_lean_is_rendered_once(monkeypatch, tmp_path):
    out, rendered = _run_infer(
        monkeypatch, tmp_path, InferResult(inv_smt="(= s0 0)", ranking_smt="s0")
    )
    assert len(rendered) == 1
    assert out.inv == "RENDERED_INV" and out.ranking == "RENDERED_RANK"


def test_an_installing_route_hands_back_no_certificate(monkeypatch, tmp_path):
    out, rendered = _run_infer(
        monkeypatch, tmp_path, InferResult(), returns="installed", kind="safety"
    )
    assert out is None and rendered == []


def test_a_row_and_its_result_have_to_agree(monkeypatch, tmp_path):
    """The row declares what comes back; nothing else has to guess which of
    the two to trust."""
    with pytest.raises(SystemExit, match="declares returns='smt'"):
        _run_infer(monkeypatch, tmp_path, InferResult(), returns="smt")
    with pytest.raises(SystemExit, match="declares returns='installed'"):
        _run_infer(
            monkeypatch,
            tmp_path,
            InferResult(inv_lean="x"),
            returns="installed",
            kind="safety",
        )


@pytest.mark.parametrize("route", ROUTES, ids=route_names())
def test_declared_artifact_roles_are_real(route):
    from zrth.lean.artifacts import ROLES

    assert route.reads_artifacts <= set(ROLES)


def test_artifacts_flag_needs_a_project():
    for flag in ("--cert-file", "--hammer-file"):
        message = err(
            ["m.py", "--buchi", "(= s0 0)", flag, "out.lean", "--artifacts", "reset"]
        )
        assert "belongs to a generated project" in message


def test_artifacts_defaults_to_use():
    assert parse_args(["m.py", "--buchi", "(= s0 0)"]).artifacts == "use"


# --- resuming, as `ai-cegar` does it -------------------------------------


def _resume_input(tmp_path, *, kind="buchi", supplied_inv=None):
    """`_resume_inv`'s answer for a store that already holds artifacts.

    The store does the role/language/status filtering (it is the half that
    knows what a stale artifact is and reports what it dropped); what is
    exercised here is the one rule left to the route.
    """
    from zrth.lean.cert import CertificateData
    from zrth.lean.infer_route import InferInput, _resume_inv

    handle = _handle(tmp_path, resumes=("inv",), store=_store(tmp_path, kind=kind))
    return _resume_inv(
        InferInput(
            project=handle,
            module=None,
            cert_data=CertificateData(prp="(= s0 0)", kind=kind, inv=supplied_inv),
            opts={},
        )
    )


def test_an_inductive_invariant_is_taken_as_given(tmp_path):
    """The user's case: infer only the ranking function, over a fixed inv."""
    a = _store(tmp_path).put(
        "inv", "(<= s0 10)", status="too_weak", why="does not imply P"
    )
    assert _resume_input(tmp_path) == a


def test_a_too_weak_invariant_is_not_a_safety_invariant(tmp_path):
    """`too_weak` means "inductive but does not imply P" -- which is all
    `rule_buchi` ever wanted, and exactly what `rule_globally` needs."""
    _store(tmp_path, kind="safety").put("inv", "(<= s0 10)", status="too_weak")
    assert _resume_input(tmp_path, kind="safety") is None


def test_a_refuted_invariant_is_not_taken_as_given(tmp_path):
    _store(tmp_path).put(
        "inv", "(= s0 3)", status="not_inductive", why="fails at the step"
    )
    assert _resume_input(tmp_path) is None


def test_an_explicit_invariant_wins_over_an_inherited_one(tmp_path):
    _store(tmp_path).put("inv", "(<= s0 10)", status="proved")
    assert _resume_input(tmp_path, supplied_inv="(<= s0 5)") is None


def test_the_registry_rejects_two_rows_with_one_name():
    """`ops.py` guards its op table the same way: a shadowed row would also
    show the name twice in `--infer`'s choices."""
    from zrth.lean.infer_route import index_routes

    assert index_routes(ROUTES).keys() == set(route_names())
    with pytest.raises(RuntimeError, match="two route-table rows named"):
        index_routes(ROUTES + ROUTES[:1])


def test_an_unknown_route_name_says_where_to_add_one():
    # `abduction` is `SMT_ASSIST.md` item 6, which is planned and not built;
    # it was `sygus` until that one became a row, which is the point -- the
    # name in this test has to be one the table really does not have.
    with pytest.raises(KeyError, match="infer_route.ROUTES"):
        route_by_name("abduction")


def test_every_alias_fills_an_option_of_the_route_it_selects():
    """An alias supplies one of its route's options, so a renamed `dest`
    cannot leave the deprecated flag a no-op that drops the route."""
    for route, alias in all_route_aliases():
        assert alias.fills in {o.dest for o in route.options}, (
            f"{alias.flag} fills {alias.fills!r}, which is not an option of "
            f"--infer {route.name}"
        )
        assert alias.flag.startswith("--") and alias.help


def test_the_owned_proveit_directory_is_the_one_the_route_writes():
    """`owns` is the truth about what a route writes, and `path()` refuses
    anything undeclared -- so a renamed `PROVEIT_DIR` must not leave the row
    declaring a directory `fbk_proveit` no longer uses."""
    from zrth.lean.project import PROVEIT_DIR

    owns = route_by_name("fbk-proveit").owns
    assert f"{PROVEIT_DIR}/" in owns


def test_the_workspace_is_not_a_route_owned_path():
    """`artifacts/` outlives the run, so `create_project`'s cleanup of
    route-written files must never reach it."""
    from zrth.lean.artifacts import ARTIFACTS_DIR

    for route in ROUTES:
        assert not any(o.startswith(ARTIFACTS_DIR) for o in route.owns)
