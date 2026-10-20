"""The SMT-LIB that is handed to a solver outside this process, pinned.

`--infer houdini` on Vampire, `--infer vampire` and the modes' z3 round
each print a script from cvc5 terms. A change to how one is printed is a
change to the question the prover answers, so the text is compared with the
snapshots under `tests/golden/scripts/`; set `ZRTH_UPDATE_GOLDEN=1` to
rewrite them, and only for a change that is intended. Nothing here runs a
prover. The obligation files `--pre-check` leaves are pinned by
`test_pipeline_golden.py`.
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import pytest

from zrth.lean.certificate import Claim

pytest.importorskip("cvc5")

GOLDEN = Path(__file__).parent / "golden" / "scripts"
LIMITS = Path(__file__).parent / "limits" / "mods"
HYBRID = Path(__file__).parent / "bench_matrix" / "hybrid"
UPDATE = os.environ.get("ZRTH_UPDATE_GOLDEN") == "1"


def _check(name: str, text: str) -> None:
    path = GOLDEN / name
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return
    assert path.exists(), f"no snapshot {path}; run with ZRTH_UPDATE_GOLDEN=1"
    assert text == path.read_text(), f"{name} differs from its snapshot"


def _module(path: Path):
    spec = importlib.util.spec_from_file_location(f"_m_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.module()


def _context(path: Path, kind: str, prp: str, route: str):
    from zrth.lean.smt.obligations import Obligations
    from zrth.lean.smt_synth import SynthContext

    ctx = SynthContext.build(_module(path), Claim(prp=prp, kind=kind),
                             route=route,
                             takes=("int", "bool", "bv", "real", "tuple"))
    return ctx, Obligations(ctx)


def _candidates(ctx, srcs):
    from zrth.lean.smt.obligations import Candidate

    return [Candidate(s, ctx.env.parse_expr(s)) for s in srcs]


# ══════════════════════════════════════════════════════════════════════════
# `--houdini-solver vampire`: one query per script
# ══════════════════════════════════════════════════════════════════════════

# (name, module, kind, property, facts, ranking or None)
HOUDINI = [
    ("countdown", LIMITS / "m_countdown.py", "buchi", "(= s0 0)",
     ["(<= 0 s0)", "(<= s0 100)"], "s0"),
    ("lra_lin", LIMITS / "m_lra_lin.py", "buchi", "(= s0 0.0)",
     ["(<= 0.0 s0)", "(<= s0 5.0)"], "(to_int s0)"),
    ("thermostat", HYBRID / "m_thermostat.py", "safety",
     "(and (>= s0 15.0) (<= s0 25.0))",
     ["(=> s1 (<= s0 23.0))", "(=> (not s1) (<= 17.0 s0))"], None),
    ("relu_vec", LIMITS / "m_relu_vec.py", "buchi",
     "(= ((_ tuple.select 0) s0) 0)", ["(<= 0 ((_ tuple.select 0) s0))"],
     None),
]


@pytest.mark.parametrize("name,path,kind,prp,facts,rank", HOUDINI,
                         ids=[c[0] for c in HOUDINI])
def test_houdini_vampire_scripts_are_unchanged(name, path, kind, prp, facts,
                                               rank):
    from zrth.lean.smt.solvers import VampireSolver

    ctx, ob = _context(path, kind, prp, "houdini")
    kept = _candidates(ctx, facts)
    queries = [("entry", ob.holds_at_entry(kept)),
               ("preserved", ob.preserved(kept, kept)),
               ("implies", ob.implies(kept, ctx.prp))]
    if rank is not None:
        queries.append(("drops", ob.drops(kept, _candidates(ctx, [rank])[0])))
    text = "".join(f";; --- {what}\n{VampireSolver._script(q)}"
                   for what, q in queries)
    _check(f"houdini-{name}.golden", text)


# ══════════════════════════════════════════════════════════════════════════
# `--infer vampire`: the conjecture with holes
# ══════════════════════════════════════════════════════════════════════════

# (name, module, kind, property, template, fixed invariant for the rank)
VAMPIRE = [
    ("countdown_safety", LIMITS / "m_countdown.py", "safety", "(<= s0 100)",
     "intervals", None),
    ("countdown_buchi", LIMITS / "m_countdown.py", "buchi", "(= s0 0)",
     "intervals", {"A0": 0, "B0": 100}),
    ("twovars_differences", LIMITS / "m_twovars.py", "safety", "(<= s0 10)",
     "differences", None),
    ("lra_half_buchi", LIMITS / "m_lra_half.py", "buchi", "(<= s0 0.0)",
     "intervals", {"A0": 0, "B0": 10}),
]


@pytest.mark.parametrize("name,path,kind,prp,which,fixed", VAMPIRE,
                         ids=[c[0] for c in VAMPIRE])
def test_vampire_conjectures_are_unchanged(name, path, kind, prp, which,
                                           fixed):
    from zrth.lean.magic.vampire import Question, split, template

    ctx, ob = _context(path, kind, prp, "vampire")
    q = Question(ctx, ob, split(ctx, ob.init, "the initial state"),
                 split(ctx, ob.next, "the round"))
    tpl = template(ctx, which, ranked=kind != "safety")
    if kind == "safety":
        text = q.conjecture(tpl, ctx.prp)
    else:
        text = (q.invariant_conjecture(tpl) + ";; --- rank\n"
                + q.rank_conjecture(tpl, ctx.prp, fixed))
    _check(f"vampire-{name}.golden", text)


# ══════════════════════════════════════════════════════════════════════════
# The modes: the round as z3 reads it
# ══════════════════════════════════════════════════════════════════════════

MODES = [
    ("thermostat", HYBRID / "m_thermostat.py", "(>= s0 21.0)"),
    ("lra_lin", LIMITS / "m_lra_lin.py", "(= s0 0.0)"),
]


@pytest.mark.parametrize("name,path,prp", MODES, ids=[c[0] for c in MODES])
def test_the_round_z3_reads_is_unchanged(name, path, prp, monkeypatch):
    z3 = pytest.importorskip("z3")
    from zrth.lean.magic import modes

    ctx, ob = _context(path, "buchi", prp, "houdini")
    read = []
    parse = z3.parse_smt2_string

    def recording(text, *a, **kw):
        read.append(text)
        return parse(text, *a, **kw)

    monkeypatch.setattr(modes.z3, "parse_smt2_string", recording)
    assert modes._Model.of(ctx, ob, time.monotonic() + 5) is not None
    assert len(read) == 1
    _check(f"modes-{name}.golden", read[0] + "\n")
