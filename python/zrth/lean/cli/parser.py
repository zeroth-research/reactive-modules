"""The flags: `build_parser`, with the help each route and input format
contributes assembled from their rows.
"""

from __future__ import annotations

import argparse

from ..artifacts import ARTIFACTS_DIR
from ..inputs import ADAPTERS
from ..infer import (
    DEFAULT_ROUTE,
    ROUTES,
    all_route_aliases,
    route_names,
    shared_route_options,
)
from ..infer.common.llm import DEFAULT_MODEL
from ..smt.limits import DEFAULT_CALL_MS, DEFAULT_PHASE_MS


_EPILOG = """\
examples:
  # bare project — no property, so `P` and `ranking` are sorry
  uv run verith mymodule.py -o out/ -p MyProject

  # a Buchi property, `G (F (= (X @intf.0) 0))` (certificate stub with known P)
  uv run verith mymodule.py --buchi "(= (X @intf.0) 0)" -o out/ -p MyProject

  # a safety property, `G (<= (X @intf.0) 100)`: invariant only, no ranking function
  uv run verith mymodule.py --safety "(<= (X @intf.0) 100)" -o out/ -p MyProject

  # AI inference with Claude (requires ANTHROPIC_API_KEY + pip install zrth[ai])
  uv run verith mymodule.py --buchi "(= (X @intf.0) 0)" --infer -o out/ -p MyProject

  # ... and the same loop for a safety property (cvc5-checked route only)
  uv run verith mymodule.py --safety "(<= (X @intf.0) 100)" --infer ai-cegis -o out/ -p MyProject

  # no LLM: learn a ranking function and certify it before it is offered
  uv run verith mymodule.py --buchi "(= (X @intf.0) 0)" --infer nuterm -o out/ -p MyProject

  # no LLM: synthesise the invariant, congruences included (`x` even), and
  # leave it in artifacts/ for the next run to take as given
  uv run verith mymodule.py --safety "(not (= (X @intf.0) 1))" --infer sygus -o out/ -p MyProject

  # no LLM: one query per shape -- and a refuted shape is a *proof* that no
  # certificate of that shape exists, which --infer ai-cegis reads next run
  uv run verith mymodule.py --buchi "(= (X @intf.0) 0)" --infer smt-linear -o out/ -p MyProject

  # no LLM: Houdini and a ranking search, each candidate decided by cvc5 --
  # a candidate it cannot prove comes back with the counterexample that kills it
  uv run verith mymodule.py --buchi "(= (X @intf.0) 0)" --infer houdini -o out/ -p MyProject

  # ... the same search, refuted by the Vampire prover instead
  uv run verith mymodule.py --buchi "(= (X @intf.0) 0)" --infer houdini \\
      --houdini-solver vampire --vampire ~/vampire/vampire -o out/ -p MyProject

  # no LLM, and nothing proposed: the obligations are stated with the invariant
  # left open and Vampire's answer literals return its coefficients
  uv run verith mymodule.py --safety "(<= (X @intf.0) 100)" --infer vampire \\
      --vampire ~/vampire/vampire -o out/ -p MyProject

  # AI inference with Ollama (requires pip install zrth[ai-local])
  uv run verith mymodule.py --buchi "(= (X @intf.0) 0)" --infer \\
      --model qwen3-coder --base-url http://localhost:11434/v1 -o out/ -p MyProject

  # ask cvc5 whether the obligations are true before spending a lake build on them
  uv run verith mymodule.py --buchi "(= (X @intf.0) 0)" --invariant "(<= (X @intf.0) 100)" --ranking "(X @intf.0)" \\
      --pre-check cvc5 -o out/ -p MyProject

  # certify through lean-ltl-certifying's proveit.py (lean2vmt -> ic3ia -> vmt2lean)
  uv run verith mymodule.py --safety "(not (= (X @intf.0) 15))" -o out/ -p MyProject \\
      --infer fbk-proveit --proveit-dir ~/lean-ltl-certifying --ic3ia ~/ic3ia/build/ic3ia

  # build the certificate of any route that fills it: `lake build Certificate`
  uv run verith mymodule.py --buchi "(= (X @intf.0) 0)" --invariant "(<= (X @intf.0) 100)" --ranking "(X @intf.0)" \\
      --build-cert -o out/ -p MyProject
"""


def _infer_help() -> str:
    """`--infer`'s help, assembled from the rows."""
    lines = [
        "Infer the certificate for --safety or --buchi: the invariant, plus "
        "the ranking function --buchi needs. One of:",
    ]
    for r in ROUTES:
        kinds = " and ".join(f"--{k}" for k in sorted(r.kinds))
        lines.append(f"`{r.name}` ({kinds}) -- {r.summary}.")
    lines.append(f"Passed without a value, `{DEFAULT_ROUTE}`.")
    return " ".join(lines)


def _format_help() -> str:
    """`--format`'s help, assembled from the rows."""
    rows = " ".join(
        f"`{a.name}` ({', '.join(a.suffixes)}) -- {a.summary}." for a in ADAPTERS
    )
    return (
        "The input's format; by default the one its suffix names. One of: "
        + rows
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verith",
        description=(
            "Generate a Lean4 certificate project from a reactive module, read "
            "from a Python file or another format --format names. "
            "The property goes under --safety (`G FORMULA`: every reachable "
            "state) or --buchi (`G (F FORMULA)`: infinitely often); with --infer "
            "the invariant is searched for -- by an LLM, by learning a ranking "
            "function and certifying it, or by an external prover -- along with "
            "the ranking function a Buchi certificate also needs."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "module_file",
        nargs="?",
        help=(
            "The input: a file stating the module, in the format --format "
            "names (by default the one its suffix names; a Python file must "
            "contain a callable that returns a Module). Optional when "
            "--hammer-file is used alone."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory where the Lean project will be created (default: current directory).",
    )
    parser.add_argument(
        "-p",
        "--project-name",
        default="Rea",
        help="Name of the Lean package / library (default: Rea).",
    )
    parser.add_argument(
        "--format",
        choices=[a.name for a in ADAPTERS],
        default=None,
        help=_format_help(),
    )
    # A format's own options, one group per format, parsed `default=None`
    # so "given" is distinguishable from "not given" -- as the routes' are.
    for adapter in ADAPTERS:
        if adapter.options:
            group = parser.add_argument_group(f"--format {adapter.name}")
            for opt in adapter.options:
                group.add_argument(*opt.flags, **{**opt.kwargs, "default": None})
    parser.add_argument(
        "-x",
        "--executable",
        action="store_true",
        help="Generate Main.lean and add [[lean_exe]] to lakefile for a runnable binary.",
    )
    parser.add_argument(
        "--safety",
        default=None,
        metavar="FORMULA",
        help=(
            "Safety property `G FORMULA`: an SMT-LIB 2 Bool expression that "
            "must hold in *every* reachable state, over the state after each "
            "round: `(X x)` for a variable the input names `x`, `(X @intf.N)` "
            "for interface variable N. Certified with `rule_globally`, which "
            "needs an invariant implying FORMULA and no ranking function at "
            "all. Example: '(not (= (X @intf.0) 15))'."
        ),
    )
    parser.add_argument(
        "--buchi",
        default=None,
        metavar="FORMULA",
        help=(
            "Buchi property `G (F FORMULA)`: an SMT-LIB 2 Bool expression "
            "over the state after each round (`(X x)`, `(X @intf.N)`) that "
            "must hold infinitely often. "
            "Certified with `rule_buchi`, which needs an invariant *and* a "
            "ranking function that decreases in every state where FORMULA "
            "is false. Example: '(= (X @intf.0) 0)'."
        ),
    )
    parser.add_argument(
        "--pre",
        default=None,
        help=(
            "Precondition as an SMT-LIB 2 Bool expression over the inputs "
            "of a round: `(X u)` / `(X @extl.N)` for the input read in it, "
            "`u` / `@extl.N` for the one read in the round before. Tuple "
            "(matrix) elements use '((_ tuple.select k) (X u))'. "
            "Applied to both init_pre and update_pre. "
            "Example: '(>= ((_ tuple.select 0) (X @extl.0)) 1.0)'."
        ),
    )
    parser.add_argument(
        "--invariant",
        default=None,
        help=(
            "Invariant as an SMT-LIB 2 Bool expression over the state, "
            "named as --safety names it. With a route that seeds from it, "
            "only what is left of the certificate is inferred. Example: "
            "'(= (X @intf.0) (X @intf.5))'."
        ),
    )
    parser.add_argument(
        "--ranking",
        default=None,
        help=(
            "Ranking as an SMT-LIB 2 Int expression over the state, named "
            "as --buchi names it, for --buchi only. With a route that seeds "
            "from it, only the invariant is inferred. Must satisfy `ranking "
            ">= 0` under the invariant. Example: "
            "'(ite (X @intf.5) (ite (X @intf.4) 1 2) 0)'."
        ),
    )
    parser.add_argument(
        "--infer",
        nargs="?",
        const=DEFAULT_ROUTE,
        default=None,
        choices=list(route_names()),
        help=_infer_help(),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model to use for inference (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "OpenAI-compatible base URL, e.g. http://localhost:11434/v1 for Ollama "
            "or https://openrouter.ai/api/v1 for OpenRouter. Hosted providers read "
            "the API key from OPENROUTER_API_KEY or OPENAI_API_KEY; local servers "
            "need no key."
        ),
    )

    # One group per option, named for every route that declares it, so
    # `--help` shows whose flag is whose and says when a flag is two routes'.
    groups: dict = {}
    for opt, owners in shared_route_options():
        title = ", ".join(f"--infer {r.name}" for r in owners)
        if title not in groups:
            groups[title] = parser.add_argument_group(title)
        # `default=None` so that "given" is distinguishable from "not
        # given" for the stray-option check; the row's own default is
        # put back in `check_route_options` once the route is known.
        kwargs = dict(opt.kwargs)
        kwargs["default"] = None
        groups[title].add_argument(*opt.flags, **kwargs)

    # Deprecated spellings, from the rows that own them: each selects its
    # route and fills one of that route's options, which is why it cannot be
    # an `Opt` of its own.
    for route, alias in all_route_aliases():
        parser.add_argument(
            alias.flag, default=None, metavar="DIR", help=alias.help
        )

    parser.add_argument(
        "--cert-file",
        default=None,
        help=(
            "Write a standalone, self-contained certificate .lean file to this path "
            "instead of creating a full project.  The file inlines init/update and "
            "imports zeroth_hammer from ZerothHammer.  The other encodings of the "
            "same module are written beside it under the same stem (Rel, Scalar, "
            "ScalarRel), as they are in a project."
        ),
    )
    parser.add_argument(
        "--hammer-file",
        default=None,
        help="Write a standalone ZerothHammer.lean to this path.",
    )
    parser.add_argument(
        "--build-cert",
        action="store_true",
        help=(
            "Build the generated certificate: `lake update` then `lake "
            "build Certificate` in the project. Needs a certificate with "
            "predicates in it, so pass --invariant and --ranking, or "
            "--infer; an obligation left as `sorry` is an error, as is one "
            "lake cannot close. Off by default because the build resolves "
            "and compiles cslib, Mathlib, lean-smt and cvc5 -- minutes "
            "against a warm package cache, much longer against a cold one."
        ),
    )
    parser.add_argument(
        "--artifacts",
        default="use",
        choices=["use", "ignore", "reset"],
        help=(
            f"What to do with the project's `{ARTIFACTS_DIR}/` directory, "
            f"which is how a sequence of runs in one -o continues rather "
            f"than restarts: a predicate a previous run found but could not "
            f"prove enough of is left there, and a route that understands it "
            f"starts from it. `use` (default) lets the route read it -- an "
            f"artifact written for a different module, property or proof "
            f"rule is skipped, and what is picked up is printed, because two "
            f"identical command lines must not quietly mean different "
            f"things. `ignore` searches from scratch and still records what "
            f"this run finds. `reset` empties it first, which is what to "
            f"pass when a bad artifact is being inherited by every run."
        ),
    )
    parser.add_argument(
        "--pre-check",
        default="none",
        choices=["none", "cvc5"],
        help=(
            "Before generating, ask cvc5 whether the certificate obligations "
            "are actually true. A refuted obligation means the invariant or "
            "ranking is wrong, which a failing `lake build` cannot tell you "
            "apart from tactics that are merely too weak. With --infer the "
            "check runs after inference, on the certificate that was "
            "inferred -- before it there is no invariant to check. Bounded "
            "by --smt-timeout / --smt-budget; anything cvc5 cannot answer is "
            "reported as unknown and changes nothing."
        ),
    )
    parser.add_argument(
        "--smt-tactics",
        default="none",
        choices=["none", "cvc5"],
        help=(
            "Let cvc5 inform the generated tactics: discharge the branch "
            "conditions the invariant already settles instead of splitting "
            "on them, hand `nlinarith` the products cvc5's own refutation "
            "needed, and drop a precondition no refutation used. Bounded by "
            "--smt-timeout / --smt-budget; strictly additive, so anything "
            "cvc5 cannot answer leaves the plan as it would have been."
        ),
    )
    parser.add_argument(
        "--smt-timeout",
        type=int,
        default=DEFAULT_CALL_MS,
        metavar="MS",
        help=(
            f"Wall-clock limit for a single SMT query, in ms "
            f"(default: {DEFAULT_CALL_MS})."
        ),
    )
    parser.add_argument(
        "--smt-budget",
        type=int,
        default=DEFAULT_PHASE_MS,
        metavar="MS",
        help=(
            f"Wall-clock limit for one phase of SMT queries, in ms "
            f"(default: {DEFAULT_PHASE_MS}). A phase that fans out over many "
            f"small queries still cannot exceed this."
        ),
    )
    return parser
