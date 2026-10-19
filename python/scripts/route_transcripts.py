"""Transcripts of `verith` runs, to check that a pipeline refactor changes nothing it should not.

Workflow: at the commit before a change, record `OUTDIR=before`; after the
change, record `OUTDIR=after`; then `diff -r before after`. Every difference
must be one the change intends. Run from `python/`:

    uv run python scripts/route_transcripts.py OUTDIR [case ...]
    uv run python scripts/route_transcripts.py --list

Each case is one or more `verith` invocations sharing one output directory;
its transcript (exit status, stdout, stderr, the generated Lean files, the
artifact store, the lakefile and the project's file list) is normalised
(paths, timings, counters) and written to OUTDIR/<case>.txt.

External tools: the fbk-proveit cases need the proof-prototyping checkout
(VERITH_PROVEIT_DIR), ic3ia (VERITH_IC3IA) and mathsat's python bindings
(VERITH_MATHSAT_PY); the cases whose arguments mention vampire need the binary
(VAMPIRE). Cases whose tool is missing are skipped, with a note on stderr.
The sygus, nuterm and houdini cases need cvc5 and the rest of the usual
verith setup. The LLM cases (ai, ai-cegis) use scripted replies: no network,
no API key.
"""

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PY = Path(__file__).resolve().parents[2] / "python"
FIX = PY / "tests" / "fixtures"
MODS = PY / "tests" / "limits" / "mods"
HYBRID = PY / "tests" / "bench_matrix" / "hybrid"


def _tool(var: str, default: str) -> Path:
    return Path(os.environ.get(var) or Path(default).expanduser())


LTL = _tool("VERITH_PROVEIT_DIR", "~/zeroth/proof-prototyping/lean-ltl-certifying")
IC3IA = _tool("VERITH_IC3IA", "~/zeroth/fbk/ic3ia/build")
MATHSAT_PY = _tool("VERITH_MATHSAT_PY", "~/zeroth/fbk/mathsat/python")
VAMPIRE = _tool("VAMPIRE", "~/Downloads/vampire")
os.environ.setdefault("VAMPIRE", str(VAMPIRE))

FBK = ["--infer", "fbk-proveit", "--proveit-dir", str(LTL), "--ic3ia", str(IC3IA)]


def C(mod: Path, args: list[str], llm: str | None = None, project: str = "G"):
    """One step: a module, its verith arguments, a scripted-LLM name, a project name."""
    return (mod, args, llm, project)


# Scripted replies per LLM script: "gen" answers generation requests, "verify"
# the verification ones; the last reply repeats when the queue runs dry.
AI_OK = ("INVARIANT: fun s => (s 0 0) ≥ 0 ∧ (s 0 0) ≤ 9\n"
         "RANKING: fun s => (if (s 0 0) = 0 then 0 else 10 - (s 0 0)).toNat")

LLM = {
    "cegis_buchi": {"gen": [
        "INVARIANT: (>= s0 0)\nRANKING: (- 10 s0)",
        "INVARIANT: (and (>= s0 0) (<= s0 9))\nRANKING: (ite (= s0 0) 0 (- 10 s0))",
    ]},
    "cegis_safety": {"gen": [
        "INVARIANT: (>= s0 0)",
        "INVARIANT: (and (>= s0 0) (<= s0 9))",
    ]},
    "cegis_rank_only": {"gen": ["RANKING: (ite (= s0 0) 0 (- 10 s0))"]},
    "cegis_rank_s0": {"gen": ["RANKING: (- 50 s0)", "RANKING: s0"]},
    "cegis_wrong": {"gen": ["INVARIANT: (>= s0 0)"] * 5},
    "cegis_unparsable": {"gen": ["nonsense", "INVARIANT: (>= s0 0)"] * 3},
    "ai_ok": {"gen": [AI_OK], "verify": ["CORRECT"]},
    "ai_retry": {"gen": [AI_OK, AI_OK],
                 "verify": ["WRONG: the ranking does not drop at 9", "CORRECT"]},
    "ai_empty": {"gen": ["INVARIANT: \nRANKING: "], "verify": ["CORRECT"]},
    "ai_wrong": {"gen": [AI_OK] * 5, "verify": ["WRONG: no"] * 5},
}

CASES = {
    "sygus_countdown": [C(FIX / "svcomp_countdown.py",
                          ["--safety", "(<= (X @intf.0) 100)", "--infer", "sygus",
                           "--pre-check", "cvc5", "--smt-tactics", "cvc5"])],
    "sygus_step2": [C(MODS / "m_step2.py",
                      ["--safety", "(not (= (X @intf.0) 1))", "--infer", "sygus",
                       "--pre-check", "cvc5", "--smt-tactics", "cvc5"])],
    "sygus_none": [C(MODS / "m_step2.py",
                     ["--safety", "(not (= (X @intf.0) 1))", "--infer", "sygus",
                      "--sygus-grammar", "linear", "--sygus-conjuncts", "1"])],
    "sygus_then_linear": [
        C(MODS / "m_countdown.py", ["--safety", "(>= (X @intf.0) 0)", "--infer", "sygus"]),
        C(MODS / "m_countdown.py", ["--safety", "(<= (X @intf.0) 100)", "--infer",
                                    "smt-linear", "--pre-check", "cvc5"]),
    ],
    "linear_buchi_twice": [
        C(MODS / "m_countdown.py", ["--buchi", "(= (X @intf.0) 0)", "--infer", "smt-linear"]),
        C(MODS / "m_countdown.py", ["--buchi", "(= (X @intf.0) 0)", "--infer", "smt-linear",
                                    "--pre-check", "cvc5"]),
    ],
    "linear_safety_given": [C(MODS / "m_countdown.py",
                              ["--safety", "(<= (X @intf.0) 100)", "--invariant", "(>= (X @intf.0) 0)",
                               "--infer", "smt-linear"])],
    "linear_toward5": [C(MODS / "m_toward5.py",
                         ["--buchi", "(= (X @intf.0) 5)", "--infer", "smt-linear"])],
    "houdini_buchi": [C(FIX / "svcomp_countdown.py",
                        ["--buchi", "(= (X @intf.0) 0)", "--infer", "houdini",
                         "--pre-check", "cvc5", "--smt-tactics", "cvc5"])],
    "houdini_twovars": [C(MODS / "m_twovars.py",
                          ["--buchi", "(and (= (X @intf.0) 0) (= (X @intf.1) 0))", "--infer", "houdini"])],
    "vampire_buchi": [C(MODS / "m_countdown.py",
                        ["--buchi", "(= (X @intf.0) 0)", "--infer", "vampire",
                         "--pre-check", "cvc5", "--smt-tactics", "cvc5"])],
    "vampire_safety": [C(MODS / "m_countdown.py",
                         ["--safety", "(<= (X @intf.0) 100)", "--infer", "vampire",
                          "--pre-check", "cvc5"])],
    "nuterm_wrap": [C(FIX / "counter.py",
                      ["--buchi", "(= (X @intf.0) 0)", "--infer", "nuterm",
                       "--pre-check", "cvc5", "--smt-tactics", "cvc5"])],
    "nuterm_safety": [C(MODS / "m_countdown.py",
                        ["--safety", "(<= (X @intf.0) 100)", "--infer", "nuterm",
                         "--pre-check", "cvc5", "--smt-tactics", "cvc5"])],
    "nuterm_refused": [C(MODS / "m_countdown.py",
                         ["--safety", "(<= (X @intf.0) 3)", "--infer", "nuterm"])],
    "nuterm_real": [C(MODS / "m_lra_lin.py",
                      ["--buchi", "(= (X @intf.0) 0.0)", "--infer", "nuterm"])],
    "nuterm_no_prop": [C(MODS / "m_countdown.py", ["--infer", "nuterm"])],
    "nuterm_then_cegis": [
        C(MODS / "m_countdown.py", ["--safety", "(<= (X @intf.0) 100)", "--infer", "nuterm",
                                    "--pre-check", "cvc5"]),
        C(MODS / "m_countdown.py", ["--safety", "(<= (X @intf.0) 100)", "--infer", "ai-cegis",
                                    "--pre-check", "cvc5"], "cegis_wrong"),
    ],
    "linear_then_cegis_buchi": [
        C(MODS / "m_countdown.py", ["--buchi", "(= (X @intf.0) 0)", "--infer", "houdini",
                                    "--pre-check", "cvc5"]),
        C(MODS / "m_countdown.py", ["--buchi", "(= (X @intf.0) 0)", "--infer", "ai-cegis",
                                    "--pre-check", "cvc5"], "cegis_rank_s0"),
    ],
    "cegis_buchi": [C(FIX / "counter.py",
                      ["--buchi", "(= (X @intf.0) 0)", "--infer", "ai-cegis",
                       "--pre-check", "cvc5", "--smt-tactics", "cvc5"], "cegis_buchi")],
    "cegis_safety": [C(FIX / "counter.py",
                       ["--safety", "(<= (X @intf.0) 9)", "--infer", "ai-cegis",
                        "--pre-check", "cvc5", "--smt-tactics", "cvc5"], "cegis_safety")],
    "cegis_inv_given": [C(FIX / "counter.py",
                          ["--buchi", "(= (X @intf.0) 0)", "--invariant",
                           "(and (>= (X @intf.0) 0) (<= (X @intf.0) 9))", "--infer", "ai-cegis",
                           "--pre-check", "cvc5"], "cegis_rank_only")],
    "cegis_pre": [C(FIX / "svcomp_bounded_input.py",
                    ["--safety", "(>= (X @intf.0) 0)", "--pre", "(>= (X @extl.0) 0)", "--infer", "ai-cegis",
                     "--pre-check", "cvc5"], "cegis_safety")],
    "cegis_wrong": [C(FIX / "counter.py",
                      ["--safety", "(<= (X @intf.0) 9)", "--infer", "ai-cegis"], "cegis_wrong")],
    "cegis_unparsable": [C(FIX / "counter.py",
                           ["--safety", "(<= (X @intf.0) 9)", "--infer", "ai-cegis"],
                           "cegis_unparsable")],
    "ai_ok": [C(FIX / "counter.py",
                ["--buchi", "(= (X @intf.0) 0)", "--infer", "ai", "--pre-check", "cvc5",
                 "--smt-tactics", "cvc5"], "ai_ok")],
    "ai_pre": [C(FIX / "svcomp_bounded_input.py",
                 ["--buchi", "(= (X @intf.0) 0)", "--pre", "(>= (X @extl.0) 0)", "--infer", "ai"], "ai_ok")],
    "ai_retry": [C(FIX / "counter.py",
                   ["--buchi", "(= (X @intf.0) 0)", "--infer", "ai"], "ai_retry")],
    "ai_empty": [C(FIX / "counter.py",
                   ["--buchi", "(= (X @intf.0) 0)", "--infer", "ai"], "ai_empty")],
    "ai_wrong": [C(FIX / "counter.py",
                   ["--buchi", "(= (X @intf.0) 0)", "--infer", "ai"], "ai_wrong")],
    "houdini_vampire_buchi": [C(MODS / "m_countdown.py",
                                ["--buchi", "(= (X @intf.0) 0)", "--infer", "houdini",
                                 "--houdini-solver", "vampire", "--vampire-cores", "1",
                                 "--pre-check", "cvc5", "--smt-tactics", "cvc5"])],
    "houdini_vampire_lra": [C(MODS / "m_lra_lin.py",
                              ["--buchi", "(= (X @intf.0) 0.0)", "--infer", "houdini",
                               "--houdini-solver", "vampire", "--vampire-cores", "1"])],
    "houdini_vampire_thermostat": [C(HYBRID / "m_thermostat.py",
                                     ["--safety", "(and (>= (X @intf.0) 15.0) (<= (X @intf.0) 25.0))",
                                      "--infer", "houdini", "--houdini-solver", "vampire",
                                      "--vampire-cores", "1"])],
    "vampire_lra_half": [C(MODS / "m_lra_half.py",
                           ["--buchi", "(<= (X @intf.0) 0.0)", "--infer", "vampire",
                            "--pre-check", "cvc5"])],
    "vampire_twovars": [C(MODS / "m_twovars.py",
                          ["--safety", "(<= (X @intf.0) 10)", "--infer", "vampire"])],
    "fbk_countdown": [C(MODS / "m_countdown.py", ["--safety", "(>= (X @intf.0) 0)", *FBK])],
    "fbk_plain": [C(MODS / "m_countdown.py",
                    ["--safety", "(>= (X @intf.0) 0)", *FBK, "--fbk-equiv", "none",
                     "--fbk-simplify", "none"])],
    "fbk_pre": [C(FIX / "svcomp_bounded_input.py",
                  ["--safety", "(>= (X @intf.0) 0)", "--pre", "(>= (X @extl.0) 0)", *FBK])],
    "fbk_unsafe": [C(MODS / "m_countdown.py", ["--safety", "(<= (X @intf.0) 3)", *FBK])],
    "fbk_alias": [C(MODS / "m_countdown.py",
                    ["--safety", "(>= (X @intf.0) 0)", "--fbk-proveit", str(LTL),
                     "--ic3ia", str(IC3IA)])],
    # a project name that is not a valid Lean identifier
    "fbk_bad_name": [C(MODS / "m_countdown.py", ["--safety", "(>= (X @intf.0) 0)", *FBK],
                       project="1P")],
    "fbk_then_linear": [
        C(MODS / "m_countdown.py", ["--safety", "(>= (X @intf.0) 0)", *FBK]),
        C(MODS / "m_countdown.py", ["--safety", "(<= (X @intf.0) 100)", "--infer", "smt-linear"]),
    ],
    "linear_then_fbk": [
        C(MODS / "m_countdown.py", ["--safety", "(<= (X @intf.0) 100)", "--infer", "smt-linear"]),
        C(MODS / "m_countdown.py", ["--safety", "(>= (X @intf.0) 0)", *FBK]),
    ],
}


def missing_tool(name: str) -> str | None:
    args = [a for _, args, _, _ in CASES[name] for a in args]
    if any("fbk-proveit" in a for a in args) and not LTL.is_dir():
        return f"proveit checkout {LTL} (VERITH_PROVEIT_DIR)"
    if any("vampire" in a for a in args) and not VAMPIRE.exists():
        return f"vampire binary {VAMPIRE} (VAMPIRE)"
    return None


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_TMP = re.compile(r"/(?:private/)?var/folders/\S+?/certify-[A-Za-z0-9_]+-[A-Za-z0-9_]+")


def normalise(text: str, out: Path) -> str:
    text = _ANSI.sub("", text)
    text = text.replace(str(out), "<OUT>").replace(str(PY), "<REPO>")
    text = re.sub(r"#\d+", "#N", text)
    text = re.sub(r"\s+\d+ ms\b", " N ms", text)
    text = re.sub(r"\b\d+(\.\d+)? s\b", "N s", text)
    text = _TMP.sub("<TMP>", text)
    text = text.replace(str(LTL), "<LTL>").replace(str(IC3IA), "<IC3IA>")
    text = text.replace(sys.executable, "<PY>")
    text = re.sub(r"finished in \d+(\.\d+)?s", "finished in N s", text)
    text = re.sub(r"\(\d+(\.\d+)?s\)", "(N s)", text)
    text = re.sub(r"(_time = )\d+(\.\d+)?", r"\1N", text)
    text = re.sub(r"Build completed successfully \(\d+ jobs\)", "Build completed", text)
    return text


def index(path: Path) -> str:
    if not path.exists():
        return "(none)\n"
    entries = json.loads(path.read_text())
    for e in entries:
        e.pop("created", None)
        e["module_digest"] = "<digest>"
    return json.dumps(entries, indent=2, sort_keys=True) + "\n"


def install_llm(script: str) -> None:
    """Replace the LLM client of the ai and cegar routes by scripted replies."""
    from zrth.lean.magic import ai, cegar

    gen = list(LLM[script].get("gen", []))
    ver = list(LLM[script].get("verify", []))

    def client(base_url, model):
        def chat(system, user):
            print(f"[fake llm] {'verify' if system is ai.VERIFY_SYSTEM else 'generate'}")
            q = ver if system is ai.VERIFY_SYSTEM else gen
            return q.pop(0) if len(q) > 1 else q[0]
        return chat

    ai.make_client = client
    cegar.make_client = client


def transcript(mod: Path, args: list[str], out: Path, name: str) -> str:
    from zrth.lean.main import main as verith

    sys.argv = ["verith", str(mod), *args, "-o", str(out), "-p", name]
    so, se = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
        try:
            verith()
            status = "exit 0"
        except SystemExit as e:
            status = f"exit {e.code}"
    project = out / name

    def read(rel):
        p = project / rel
        return p.read_text() if p.exists() else "(none)\n"

    parts = [
        f"=== {status}", "=== stdout", so.getvalue(), "=== stderr", se.getvalue(),
        "=== Data.lean", read("Certificate/Data.lean"),
        "=== Certificate.lean", read("Certificate/Certificate.lean"),
        "=== index.json", index(project / "artifacts" / "index.json"),
    ]
    store = project / "artifacts"
    if store.is_dir():
        for p in sorted(store.iterdir()):
            if p.name not in ("index.json", "README.md"):
                parts += [f"=== artifacts/{p.name}", p.read_text()]
    for rel in ("lakefile.toml", "Certificate/Equivalence.lean"):
        parts += [f"=== {rel}", read(rel)]
    if (project / "ProveIt").is_dir():
        for p in sorted((project / "ProveIt").glob("*.lean")):
            parts += [f"=== ProveIt/{p.name}", p.read_text()]
    files = sorted(
        str(p.relative_to(project)) for p in project.rglob("*")
        if p.is_file() and not any(s in p.parts for s in ("Core", "LeanAI", ".lake"))
    ) if project.is_dir() else []
    parts += ["=== files", "\n".join(files)]
    return "\n".join(parts)


def run_step(name: str, i: int, out: Path, dest: Path) -> None:
    mod, args, llm, project = CASES[name][i]
    if llm:
        install_llm(llm)
    if any("fbk-proveit" in a for a in args):
        os.environ["PYTHONPATH"] = str(MATHSAT_PY)
    dest.write_text(normalise(transcript(mod, args, out, project), out))


def main() -> None:
    argv = sys.argv[1:]
    if argv[:1] == ["--list"]:
        print("\n".join(CASES))
        return
    if argv[:1] == ["--step"]:
        run_step(argv[1], int(argv[2]), Path(argv[3]), Path(argv[4]))
        return
    dest = Path(argv[0])
    dest.mkdir(parents=True, exist_ok=True)
    only = argv[1:]
    for name in only:
        if name not in CASES:
            sys.exit(f"unknown case {name}")
    for name, steps in CASES.items():
        if only and name not in only:
            continue
        if (why := missing_tool(name)):
            print(f"skipping {name}: missing {why}", file=sys.stderr)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            parts = []
            for i in range(len(steps)):
                # one process per step: module-level caches and the LLM
                # monkeypatch must not leak from one step into the next
                part = Path(tmp) / f"step{i}.txt"
                subprocess.run([sys.executable, __file__, "--step", name, str(i),
                                str(out), str(part)], check=True)
                parts.append(part.read_text())
            (dest / f"{name}.txt").write_text("\n#####\n".join(parts))
        print(name, flush=True)


if __name__ == "__main__":
    main()
