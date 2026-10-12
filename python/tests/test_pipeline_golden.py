"""The certificate files and `verith` transcripts, pinned byte for byte.

Two layers: the codegen entry points on a fixed set of modules and
predicates, and whole in-process `verith` runs (stdout, the two certificate
files, `artifacts/index.json`). Both are compared with the snapshots under
`tests/golden/pipeline/`; set `ZRTH_UPDATE_GOLDEN=1` to rewrite them, and
only for a change to the output that is intended.

No run reaches an LLM: `_make_client` is replaced in both modules that build
one, and the `ai-cegis` run has every predicate fixed, so it never asks.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest
import torch

from zrth import Module, Term, Var, Wire, X, LIA, LRA, Int, Real, Bool
from zrth.lean.cert import (
    LeanCertificate,
    PlanInputs,
    generate_certificate_lean,
    generate_data_lean,
    predicate_terms,
    render_certificate,
)
from zrth.lean.certificate import Certificate, Claim
from zrth.lean.common import LeanContext
from zrth.lean.project import generate_standalone_cert_lean, load_module_from_file

pytest.importorskip("cvc5")

GOLDEN = Path(__file__).parent / "golden" / "pipeline"
FIXTURES = Path(__file__).parent / "fixtures"
UPDATE = os.environ.get("ZRTH_UPDATE_GOLDEN") == "1"


def _check(name: str, text: str) -> None:
    path = GOLDEN / name
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return
    assert path.exists(), f"no snapshot {path}; run with ZRTH_UPDATE_GOLDEN=1"
    assert text == path.read_text(), f"{name} differs from its snapshot"


# ══════════════════════════════════════════════════════════════════════════
# Codegen
# ══════════════════════════════════════════════════════════════════════════


def _fixture(name: str) -> Module:
    return load_module_from_file(str(FIXTURES / f"{name}.py"))


def _vector_counter() -> Module:
    """3x1 state over a 2x1 input, as `tests/lean/conftest.py` builds it."""
    state = Var(Int([3, 1]))
    extl = Var(Int([2, 1]))
    zero31 = torch.zeros((3, 1), dtype=torch.int64)
    zero11 = torch.zeros((1, 1), dtype=torch.int64)
    a = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int64)
    init = [Term(LIA.Linear(a, zero31), [X(state)], [X(extl)])]
    x, y, z = Wire(Int([1, 1])), Wire(Int([1, 1])), Wire(Int([1, 1]))
    x_lt_y, x_lt_z, cond = Wire(Bool([1, 1])), Wire(Bool([1, 1])), Wire(Bool([1, 1]))
    yes, no = Wire(Int([3, 1])), Wire(Int([3, 1]))
    row = lambda i: torch.tensor([[int(j == i) for j in range(3)]], dtype=torch.int64)
    e1 = torch.tensor([[1], [0], [0]], dtype=torch.int64)
    diag_yz = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.int64)
    update = [
        Term(LIA.Linear(row(0), zero11), [x], [state]),
        Term(LIA.Linear(row(1), zero11), [y], [state]),
        Term(LIA.Linear(row(2), zero11), [z], [state]),
        Term(LIA.Lt(), [x_lt_y], [x, y]),
        Term(LIA.Lt(), [x_lt_z], [x, z]),
        Term(LIA.Or(), [cond], [x_lt_y, x_lt_z]),
        Term(LIA.Linear(torch.eye(3, dtype=torch.int64), e1), [yes], [state]),
        Term(LIA.Linear(diag_yz, zero31), [no], [state]),
        Term(LIA.Ite(), [X(state)], [cond, yes, no]),
    ]
    return Module.sequential([state, extl], init, update)


def _real_countdown() -> Module:
    x = Var(Real([1, 1]))
    one = Wire(Real([1, 1]))
    return Module.sequential(
        [x],
        [Term(LRA.Real(torch.tensor([[5.0]])), [X(x)])],
        [
            Term(LRA.Real(torch.tensor([[1.0]])), [one]),
            Term(LRA.Sub(), [X(x)], [x, one]),
        ],
    )


def _relu(e: str) -> str:
    return f"(ite (> {e} 0) {e} 0)"


def _dense(layers: int) -> str:
    """A ranking whose printed form shares subterms behind `let`."""
    a, b = "s0", "s0"
    for _ in range(layers):
        a, b = _relu(f"(+ {a} {b})"), _relu(f"(+ {a} (- {b}))")
    return f"(+ {a} {b})"


COUNTDOWN_INV = "(and (>= s0 0) (<= s0 100))"

# (name, module, claim, certificate, ask cvc5 for hints)
CODEGEN = [
    ("bare", lambda: _fixture("counter"), None, None, False),
    ("countdown", lambda: _fixture("svcomp_countdown"), Claim(prp="(= s0 0)"),
     Certificate(COUNTDOWN_INV, "(ite (= s0 0) 0 s0)"), False),
    ("countdown_safe", lambda: _fixture("svcomp_countdown"),
     Claim(prp="(<= s0 100)", kind="safety"), Certificate(COUNTDOWN_INV),
     False),
    ("property_only", lambda: _fixture("svcomp_countdown"),
     Claim(prp="(= s0 0)"), Certificate(), False),
    ("twovars", lambda: _fixture("svcomp_twovars"), Claim(prp="(= s0 s1)"),
     Certificate("(and (>= s0 0) (<= s0 s1) (= s1 10))",
                 "(ite (= s0 s1) 0 (- s1 s0))"), False),
    ("vector_counter_pre", _vector_counter,
     Claim(prp="s[0][0] == 0",
           init_pre="And(e[0][0] >= 0, e[1][0] >= 0)",
           update_pre="e[0][0] >= 0"),
     Certificate(
         "And(s[0][0] >= 0, Or(s[0][0] <= s[1][0], s[0][0] <= s[2][0]))",
         "Ite(s[0][0] == 0, 0, (Ite(s[1][0] >= s[2][0], s[1][0], "
         "s[2][0]) - s[0][0]) + 1)"), False),
    ("shared_let", lambda: _fixture("svcomp_countdown"), Claim(prp="(= s0 0)"),
     Certificate(COUNTDOWN_INV, _dense(3)), False),
    ("hints_settle_a_branch", lambda: _fixture("svcomp_countdown"),
     Claim(prp="(= s0 0)"),
     Certificate(COUNTDOWN_INV, "(ite (<= s0 200) s0 999)"), True),
    ("nonlinear_hints", lambda: _fixture("svcomp_countdown"),
     Claim(prp="(= s0 0)"), Certificate(COUNTDOWN_INV, "(* s0 s0)"), True),
    ("real", _real_countdown, Claim(prp="(<= s0 0.0)"),
     Certificate("(<= s0 5.0)", "(to_int s0)"), False),
    ("real_safety", _real_countdown, Claim(prp="(<= s0 5.0)", kind="safety"),
     Certificate("(<= s0 5.0)"), False),
]


def _codegen(module: Module, claim: "Claim | None", cert: "Certificate | None",
             hints: bool) -> str:
    """Data.lean, Certificate.lean and the standalone certificate, as one text."""
    from zrth.lean.smt_query import SmtBudget, predicate_facts, solver_hints

    lean = None
    if claim is not None:
        q = predicate_terms(module, claim, cert)
        h = solver_hints(q, SmtBudget()) if hints else None
        lean = LeanCertificate(
            render_certificate(q, share=not (h and h.determined)),
            PlanInputs(predicate_facts(q), h),
        )
    ctx = LeanContext(module)
    return "\n".join([
        "=== Certificate/Data.lean",
        generate_data_lean(ctx, lean),
        "=== Certificate/Certificate.lean",
        generate_certificate_lean(ctx, lean),
        "=== standalone",
        generate_standalone_cert_lean(module, lean),
    ])


@pytest.mark.parametrize("name,make,claim,cert,hints", CODEGEN,
                         ids=[c[0] for c in CODEGEN])
def test_codegen_is_unchanged(name, make, claim, cert, hints):
    _check(f"codegen-{name}.golden", _codegen(make(), claim, cert, hints))


# ══════════════════════════════════════════════════════════════════════════
# `verith`, in process
# ══════════════════════════════════════════════════════════════════════════

COUNTER = "counter"
COUNTDOWN = "svcomp_countdown"
BOUNDED = "svcomp_bounded_input"

_AI_REPLIES = {
    "generate": "INVARIANT: fun s => (s 0 0) ≥ 0 ∧ (s 0 0) ≤ 9\n"
                "RANKING: fun s => (if (s 0 0) = 0 then 0 else 10 - (s 0 0)).toNat",
    "verify": "CORRECT",
}

# (name, fixture, arguments). `-o` and `-p` are added.
RUNS = [
    ("bare", COUNTER, []),
    ("supplied", COUNTER, [
        "--buchi", "(= s0 0)",
        "--invariant", "(and (>= s0 0) (<= s0 9))",
        "--ranking", "(ite (<= s0 200) (ite (= s0 0) 0 (- 10 s0)) 999)",
        "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
    ("supplied_pre", BOUNDED, [
        "--safety", "(>= s0 0)", "--pre", "(>= e0 0)",
        "--invariant", "(>= s0 0)",
        "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
    ("cert_file", COUNTER, [
        "--buchi", "(= s0 0)",
        "--invariant", "(and (>= s0 0) (<= s0 9))",
        "--ranking", "(ite (= s0 0) 0 (- 10 s0))",
        "--cert-file", "{out}/Cert.lean",
    ]),
    ("unreadable_property", COUNTER, ["--buchi", "(= s0"]),
    ("smt_linear", COUNTDOWN, [
        "--buchi", "(= s0 0)", "--invariant", "(and (>= s0 0) (<= s0 100))",
        "--infer", "smt-linear", "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
    ("houdini", COUNTDOWN, [
        "--safety", "(<= s0 100)", "--infer", "houdini",
        "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
    ("nuterm", COUNTDOWN, [
        "--buchi", "(= s0 0)", "--infer", "nuterm",
        "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
    ("nuterm_pre", BOUNDED, [
        "--safety", "(>= s0 0)", "--pre", "(>= e0 0)", "--infer", "nuterm",
        "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
    ("ai_cegis_fixed", COUNTER, [
        "--buchi", "(= s0 0)",
        "--invariant", "(and (>= s0 0) (<= s0 9))",
        "--ranking", "(ite (<= s0 200) (ite (= s0 0) 0 (- 10 s0)) 999)",
        "--infer", "ai-cegis", "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
    ("ai", COUNTER, [
        "--buchi", "(= s0 0)", "--infer", "ai",
        "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
    ("sygus", COUNTDOWN, [
        "--safety", "(<= s0 100)", "--infer", "sygus",
        "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
    # A short declared limit: Vampire answers this one in about a second
    # under 10 and in about six under the default 120.
    ("vampire", COUNTDOWN, [
        "--safety", "(<= s0 100)", "--infer", "vampire",
        "--vampire-timeout", "10",
        "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
]

# `ModuleSMT`s built per run. The pre-check and both renderings share the
# run's `smt.run.lazy` encoding; a route that searches still builds its own.
ENCODINGS = {
    "bare": 0, "supplied": 1, "supplied_pre": 1, "cert_file": 1,
    "unreadable_property": 1, "smt_linear": 2, "houdini": 2, "nuterm": 1,
    "nuterm_pre": 1, "ai_cegis_fixed": 2, "ai": 1, "sygus": 2, "vampire": 2,
}


def _has_vampire() -> bool:
    from zrth.lean.common import Refused
    from zrth.lean.smt.vampire import resolve_vampire

    try:
        resolve_vampire(None)
    except Refused:
        return False
    return True


def _runs() -> list:
    out = []
    for name, fixture, args in RUNS:
        marks = ()
        if "vampire" in args:
            marks = pytest.mark.skipif(
                not _has_vampire(),
                reason="no Vampire binary: set $VAMPIRE or put `vampire` on PATH")
        out.append(pytest.param(name, fixture, args, id=name, marks=marks))
    return out


def _offline(monkeypatch) -> None:
    """Both LLM routes answer from `_AI_REPLIES` and never open a client."""
    from zrth.lean.magic import ai, cegar

    def client(base_url, model):
        def chat(system: str, user: str) -> str:
            return _AI_REPLIES[
                "verify" if system is ai.VERIFY_SYSTEM else "generate"]
        return chat

    def no_client(base_url, model):
        def chat(system: str, user: str) -> str:
            raise AssertionError("ai-cegis with every predicate fixed asked")
        return chat

    monkeypatch.setattr(ai, "_make_client", client)
    monkeypatch.setattr(cegar, "_make_client", no_client)


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _normalise(text: str, out: Path) -> str:
    text = _ANSI.sub("", text)
    text = text.replace(str(out), "<OUT>").replace(str(FIXTURES), "<FIX>")
    text = re.sub(r"#\d+", "#N", text)                     # wire ids
    text = re.sub(r"\s+\d+ ms\b", " N ms", text)            # verdict timings
    text = re.sub(r"\b\d+(\.\d+)? s\b", "N s", text)        # route timings
    return text


def _index(path: Path) -> str:
    if not path.exists():
        return "(none)\n"
    entries = json.loads(path.read_text())
    for e in entries:
        e.pop("created", None)
        e["module_digest"] = "<digest>"      # hashes wire ids
    return json.dumps(entries, indent=2, sort_keys=True) + "\n"


def _transcript(monkeypatch, capsys, tmp_path, fixture, args) -> str:
    from zrth.lean.main import main

    out = tmp_path / "out"
    argv = [a.replace("{out}", str(out)) for a in args]
    monkeypatch.setattr(sys, "argv", [
        "verith", str(FIXTURES / f"{fixture}.py"), *argv,
        "-o", str(out), "-p", "G",
    ])
    _offline(monkeypatch)
    try:
        main()
        status = "exit 0"
    except SystemExit as e:
        status = f"exit {e.code}"
    captured = capsys.readouterr()
    project = out / "G"

    def read(rel: str) -> str:
        p = project / rel
        return p.read_text() if p.exists() else "(none)\n"

    parts = [
        f"=== {status}",
        "=== stdout", captured.out,
        "=== stderr", captured.err,
        "=== Certificate/Data.lean", read("Certificate/Data.lean"),
        "=== Certificate/Certificate.lean", read("Certificate/Certificate.lean"),
        "=== artifacts/index.json", _index(project / "artifacts" / "index.json"),
    ]
    if (out / "Cert.lean").exists():
        parts += ["=== Cert.lean", (out / "Cert.lean").read_text()]
    # Every file the run left in `artifacts/` but the two the index already
    # pins: the SMT-LIB it asked (`system.smt2`, `obligation-*.smt2`), what
    # it found and the notes it left.
    store = project / "artifacts"
    if store.is_dir():
        for path in sorted(store.iterdir()):
            if path.name not in ("index.json", "README.md"):
                parts += [f"=== artifacts/{path.name}", path.read_text()]
    return _normalise("\n".join(parts), out)


def _count_encodings(monkeypatch) -> list:
    """Counts each `ModuleSMT` built from here on, in a one-element list."""
    from zrth.lean.smt_module import ModuleSMT

    built = [0]
    post_init = ModuleSMT.__post_init__

    def counted(self):
        built[0] += 1
        post_init(self)

    monkeypatch.setattr(ModuleSMT, "__post_init__", counted)
    return built


@pytest.mark.parametrize("name,fixture,args", _runs())
def test_verith_run_is_unchanged(name, fixture, args, monkeypatch, capsys,
                                 tmp_path):
    built = _count_encodings(monkeypatch)
    _check(f"verith-{name}.golden",
           _transcript(monkeypatch, capsys, tmp_path, fixture, args))
    assert built[0] == ENCODINGS.get(name), (
        f"{name} encoded the module {built[0]} times")
