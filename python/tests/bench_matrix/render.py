#!/usr/bin/env python3
"""Render `results.json` as one static, local HTML page.

    uv run python tests/bench_matrix/render.py -o matrix.html

No server, no network, no build step: one file that opens from the
filesystem. The route descriptions are read from `zrth.lean.infer_route`
rather than restated, so the page cannot drift from the table that drives
the CLI; everything else comes from the results file, including the machine
the timings were taken on.
"""
from __future__ import annotations

import argparse
import ast
import html
import json
import re
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from pygments.lexers import get_lexer_for_filename
from pygments.lexers.special import TextLexer
from pygments.token import Token
from pygments.util import ClassNotFound

SP = Path(__file__).resolve().parent
PY = SP.parent.parent
sys.path.insert(0, str(PY))
sys.path.insert(0, str(SP))

from merge import load, merge  # noqa: E402

WORK = Path(os.environ.get("VERITH_BENCH_WORK", "/tmp/verith-bench.noindex"))

SUITE_BLURB = {
    "limits": (
        "The limit matrix, <code>tests/limits/cases.py</code>. Its 77 cases vary the "
        "<em>supplied</em> <code>--invariant</code> and <code>--ranking</code> over a much "
        "smaller set of questions, and no <code>--infer</code> route reads a supplied "
        "predicate, so the cases collapse onto their (module, property, precondition) key "
        "the way <code>run_limits.py</code> collapses them: 30 distinct questions. "
        "Every property is a <code>--buchi</code> reachability target."),
    "fbk": (
        "The <code>--fbk-proveit</code> sweep, <code>tests/lean/fbk/probes.py</code>. "
        "39 <code>--safety</code> properties over the same module fixtures. The first "
        "group are the limit matrix's own invariants reused as safety properties, so they "
        "are inductive by construction and measure the plumbing; the rest are net-shaped, "
        "hand-written or deliberately hard."),
    "tests": (
        "The CLI's own fixtures, <code>tests/fixtures/</code>. Each states its "
        "<code>--buchi</code> property in its docstring (&ldquo;Property: x == 0 holds "
        "infinitely often&rdquo;); that line, as SMT-LIB, is what is asked here. "
        "<code>counter</code>, <code>twobit</code> and <code>twobit_lia</code> also appear "
        "in the limit matrix, under other properties."),
    "svcomp": (
        "The SV-COMP <code>termination-crafted-lit</code> corpus, "
        "<code>benchmarks/svcomp/dsl/</code> &mdash; 57 benchmarks, the ones the nuterm "
        "learner is scored on. No property is written down in the corpus, because "
        "these are termination benchmarks and the Farkas pipeline states termination as a "
        "claim over wires rather than as the one-state predicate <code>verith</code> wants. "
        "Each benchmark is asked <code>terminates</code>, read off its loop guard, and the 23 "
        "Houdini finds invariants for are also asked <code>houdini-inv</code>; how is "
        "recorded per property."),
}

VERDICTS = {
    "VERIFIED": ("ok", "Lean discharged every obligation."),
    "UNSUPPORTED": ("na", "The route does not take this question &mdash; this kind of "
                          "property, or a precondition &mdash; and <code>verith</code> "
                          "refused it before generating anything. The panel carries its "
                          "reason."),
    "REFUTED": ("no", "The route did not merely fail to find a certificate &mdash; it "
                       "<em>disproved</em> the property, with a counterexample. Green "
                       "where the property is <em>known false</em>, which is what the "
                       "suites' deliberately false controls are there to produce; red "
                       "where it is known to hold."),
    "NO-CERT": ("no", "The route searched its shape and returned nothing. For "
                      "<code>sygus</code> and <code>smt-linear</code> a bounded shape that "
                      "comes back empty is a <em>proof</em> that it is empty, not a search "
                      "that ran out of time."),
    "GEN-FAIL": ("bad", "<code>verith</code> could not emit a project at all &mdash; a "
                        "codegen gap, or a module shape the route refused up front."),
    "PROOF-FAIL": ("bad", "A certificate was produced and Lean rejected it: either the "
                          "certificate is wrong or the tactics are too weak. "
                          "<code>--pre-check cvc5</code> is what tells those apart."),
    "BUILD-FAIL": ("bad", "The project failed to build outside the certificate &mdash; one "
                          "of the five encodings of the module did not compile."),
    "TIMEOUT": ("bad", "The cell was killed at the clock rather than measured &mdash; "
                       "so it is not evidence either way. The budgets are stated in "
                       "the machine block above."),
    "SORRY": ("open", "The project built with an obligation left as <code>sorry</code>."),
    "SORRY+FAIL": ("open", "An obligation left as <code>sorry</code> <em>and</em> a build "
                           "failure besides it."),
}

def prose(text: str) -> str:
    """A route's `summary`, marked up. It is written as plain text for
    `--help`, where a backtick is a backtick and ` -- ` is an em dash."""
    out, parts = [], esc(text).split("`")
    for i, part in enumerate(parts):
        out.append(f"<code>{part}</code>" if i % 2 else part)
    joined = "".join(out).replace(" -- ", " &mdash; ")
    return re.sub(r"\*(\w[^*]*?)\*", r"<em>\1</em>", joined)


def route_docs():
    """Each route as the CLI's own table describes it."""
    from zrth.lean.cli import _SEED_FLAGS
    from zrth.lean.infer_route import ROUTES

    flag_of = {field: flag for field, _attr, flag in _SEED_FLAGS}

    out = []
    for r in ROUTES:
        out.append(dict(
            name=r.name, summary=r.summary, kinds=sorted(r.kinds), llm=r.uses_llm,
            returns=r.returns, seeds=sorted(flag_of[s] for s in r.seeds),
            opts=[f for o in r.options for f in o.flags],
            kinds_refusal=r.kinds_refusal))
    return out


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def secs(v) -> str:
    if v is None:
        return "&mdash;"
    return f"{v:.1f}" if v >= 0.05 else "&mdash;"


def timing(h: dict) -> str:
    """One half's seconds, carrying how firm the number is.

    With one sample this is the reading. With several it is their mean, and
    the spread is what says whether the mean means anything -- so the cell
    shows it rather than making the reader open the panel."""
    txt = secs(h.get("secs"))
    n = h.get("n", 1)
    if n > 1 and h.get("secs", 0) >= 0.05:
        pm = h.get("spread", 0.0) / 2
        title = f"mean of {n}: {', '.join(f'{v:g}s' for v in h.get('samples', []))}"
        extra = f"&plusmn;{pm:.1f}" if pm >= 0.05 else ""
        return (f'<span class="mean" title="{esc(title)}">{txt}'
                f'<span class="n">{extra}</span></span>')
    return txt


CSS = """
:root{--bg:#fbfbf9;--fg:#1c1b19;--dim:#6b6862;--line:#e2ded6;--card:#fff;
 --ok:#1a7f4b;--okbg:#e7f5ec;--no:#8a6d1f;--nobg:#fbf3dc;--bad:#a8332a;
 --badbg:#fdeceb;--open:#4c5a72;--openbg:#eef1f6;--code:#f3f1ec;--accent:#2f5d8a;
 --band:#e8eff7;--bandline:#bed3e5}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#e9e6e0;--dim:#9d9a93;
 --line:#2e2d29;--card:#1e1e1b;--ok:#6fd39b;--okbg:#14301f;--no:#e0bd63;--nobg:#332a12;
 --bad:#f08b80;--badbg:#3a1a17;--open:#a8b8d4;--openbg:#1c222e;--code:#24241f;
 --accent:#8fb8de;--band:#1b2531;--bandline:#3a5271}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
 -webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:16px}
@media(min-width:760px){.wrap{padding:32px 24px 80px}}
code,pre,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
code{background:var(--code);padding:.1em .35em;border-radius:3px;font-size:.88em}
pre{background:var(--code);padding:12px 14px;border-radius:6px;overflow-x:auto;
 font-size:12.5px;line-height:1.55;margin:.5em 0}
h1{font-size:1.75rem;line-height:1.25;margin:0 0 .3em;letter-spacing:-.02em}
h2{font-size:1.22rem;margin:2.4em 0 .7em;padding-bottom:.3em;
 border-bottom:1px solid var(--line);letter-spacing:-.01em}
h3{font-size:1rem;margin:1.8em 0 .5em}
p{margin:.65em 0}
.lede{color:var(--dim);font-size:1.02rem;max-width:66ch}
.stamp{color:var(--dim);font-size:.83rem;margin-top:.6em}
a{color:var(--accent)}
p.jump{margin:1.3em 0 0}
p.jump a{display:inline-block;padding:7px 13px;border:1px solid var(--bandline);
 border-radius:6px;background:var(--band);font-weight:600;font-size:.92rem;
 text-decoration:none}
p.jump a:hover{border-color:var(--accent);color:var(--accent)}
table.kv{border-collapse:collapse;font-size:.88rem;width:100%}
table.kv td{padding:5px 12px 5px 0;border-bottom:1px solid var(--line);
 vertical-align:top}
table.kv td:first-child{color:var(--dim);white-space:nowrap;width:1%}
.cards{display:grid;gap:12px;margin:1.2em 0}
@media(min-width:720px){.cards{grid-template-columns:1fr 1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.card h4{margin:0 0 .35em;font-size:.97rem;display:flex;align-items:center;
 gap:8px;flex-wrap:wrap}
.card p{margin:.4em 0;font-size:.88rem;color:var(--dim)}
.tag{font-size:10.5px;letter-spacing:.04em;text-transform:uppercase;
 padding:2px 6px;border-radius:3px;background:var(--code);color:var(--dim);
 font-weight:600;white-space:nowrap}
.tag.llm{background:var(--nobg);color:var(--no)}
.v{font-size:11px;font-weight:700;letter-spacing:.03em;padding:2px 7px;
 border-radius:3px;white-space:nowrap;text-align:center}
.v.ok{background:var(--okbg);color:var(--ok)}
.v.no{background:var(--nobg);color:var(--no)}
.v.bad{background:var(--badbg);color:var(--bad)}
.v.open{background:var(--openbg);color:var(--open)}
.v.na{background:transparent;color:var(--dim);border:1px dashed var(--line)}
/* the matrix */
.bench{background:var(--card);border:1px solid var(--line);border-radius:8px;
 margin:26px 0;overflow:clip}
/* The header band -- name, path, description -- is tinted and accented, so a
   benchmark is seen to start; it sticks while its properties scroll past. */
.bench>.bh{padding:11px 16px 9px;background:var(--band);
 box-shadow:inset 3px 0 0 var(--accent);position:sticky;top:0;z-index:2;
 display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.bh .bname{font-weight:700;font-size:1.06rem;letter-spacing:-.01em;
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.bh .path{font-size:11.5px;color:var(--dim)}
.bh .bcount{margin-left:auto;font-size:11px;letter-spacing:.04em;
 text-transform:uppercase;color:var(--dim);font-weight:600}
a.file{color:inherit;text-decoration:underline;text-decoration-style:dotted;
 text-underline-offset:2px;cursor:pointer}
a.file:hover{color:var(--accent);text-decoration-style:solid}
a.file.path{color:var(--dim)}
.bench>.desc{margin:0;padding:7px 16px 8px;font-size:12.5px;color:var(--dim);
 background:var(--band);box-shadow:inset 3px 0 0 var(--accent)}
.bench>.desc code{font-size:11.5px;background:transparent;padding:0}
.bench>.bh,.bench>.desc{border-bottom:1px solid var(--bandline)}
.bench>.bh:not(:has(+ .desc)),.bench>.desc{border-bottom-width:2px}
dialog#srcdlg{width:min(900px,94vw);height:min(80vh,900px);padding:0;border:1px solid var(--line);
 border-radius:8px;background:var(--bg);color:var(--fg)}
dialog#srcdlg::backdrop{background:rgba(0,0,0,.35)}
.dlg-bar{position:sticky;top:0;display:flex;gap:12px;align-items:baseline;padding:9px 14px;
 background:var(--code);border-bottom:1px solid var(--line)}
.dlg-bar .dlg-close{margin-left:auto}
h3.dir{margin:2.2em 0 .7em;font-size:1.05rem}
.tag.suite{background:var(--openbg);color:var(--open)}
.tag.truth.holds{background:var(--okbg);color:var(--ok)}
.tag.truth.fails{background:var(--badbg);color:var(--bad)}
.grid .m.missing{color:var(--dim);border-top:1px solid var(--line)}
dl.suites{font-size:.9rem;margin:.6em 0 1.4em}
dl.suites dt{font-weight:700;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;margin-top:.7em}
dl.suites dd{margin:.15em 0 0 1.4em;color:var(--dim)}
.tocdir{margin:.5em 0;break-inside:avoid}
.tocb{font-size:11.5px;line-height:1.55;margin:.2em 0 0 1em}
.tocb a{display:inline;margin-right:.55em;white-space:nowrap}
.prop{padding:12px 16px;border-bottom:1px solid var(--line)}
.prop:last-child{border-bottom:0}
.prop>.head{display:flex;gap:9px;align-items:baseline;flex-wrap:wrap;
 margin-bottom:3px}
.prop>.head b{font-size:.92rem}
.prop .smt{font-size:11.5px;color:var(--dim);word-break:break-word;
 display:block;margin:.3em 0 .1em;background:transparent;padding:0}
.prop .note{font-size:11.5px;color:var(--dim);margin:.35em 0 .6em}
.grid{border:1px solid var(--line);border-radius:6px;overflow:hidden}
.grid .hd,.grid summary,.grid .m.missing{display:grid;
 grid-template-columns:minmax(88px,1.4fr) 92px 66px 66px 16px;
 gap:8px;align-items:center;padding:6px 10px;font-size:12.5px}
.grid .hd{background:var(--code);color:var(--dim);font-size:10.5px;
 letter-spacing:.04em;text-transform:uppercase;font-weight:600}
.grid .hd span:nth-child(3),.grid .hd span:nth-child(4),
.grid summary .t{text-align:right;font-variant-numeric:tabular-nums}
details.m{border-top:1px solid var(--line)}
details.m:first-of-type{border-top:0}
details.m>summary{cursor:pointer;list-style:none}
details.m>summary::-webkit-details-marker{display:none}
details.m>summary:hover{background:var(--code)}
details.m>summary .name{font-family:ui-monospace,Menlo,monospace;font-weight:600}
details.m>summary .t{color:var(--dim)}
details.m>summary .chev{color:var(--dim);font-size:10px;transition:transform .12s}
details.m[open]>summary .chev{transform:rotate(90deg)}
details.m[open]>summary{background:var(--code)}
.m .body{padding:4px 10px 12px;border-top:1px solid var(--line);
 background:var(--bg)}
.m .body pre{margin:.5em 0 .3em;user-select:all}
.m .body .lbl{font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;
 color:var(--dim);font-weight:600;margin-top:.7em}
.m .body .found{font-size:11.5px;word-break:break-all;color:var(--dim)}
.m .body .err{font-size:11.5px;color:var(--bad);word-break:break-word}
.m .body .lbl .hint{text-transform:none;letter-spacing:0;font-weight:400}
pre.err{color:var(--bad);white-space:pre-wrap;max-height:19em;overflow-y:auto}
.mean{white-space:nowrap}
.mean .n{color:var(--dim);font-size:10px;margin-left:.25em;font-weight:400}
.flag{display:inline-block;margin-left:5px;width:13px;height:13px;line-height:13px;
 text-align:center;border-radius:50%;background:var(--bad);color:#fff;
 font-size:10px;font-weight:700;cursor:help;vertical-align:1px}
.cav{border:1px solid var(--line);border-left:3px solid var(--dim);
 background:var(--card);border-radius:5px;padding:11px 14px;margin:10px 0}
.cav-hi{border-left-color:var(--bad)}
.cav-h{display:flex;align-items:baseline;gap:9px;margin-bottom:5px}
.cav-h code{font-size:12px;font-weight:700}
.cav-n{font-size:11px;color:var(--dim)}
.cav p{margin:0;font-size:13px}
.cav details{margin-top:7px}
.cav summary{font-size:11.5px;color:var(--dim);cursor:pointer}
.cav pre{font-size:11px;max-height:17em;overflow:auto;margin:6px 0 0;
 white-space:pre-wrap;word-break:break-word}
p.note{font-size:12px;color:var(--dim);margin:3px 0 0}
/* summary matrix */
table.sum{border-collapse:collapse;font-size:12.5px;width:100%;
 font-variant-numeric:tabular-nums}
table.sum th,table.sum td{padding:5px 9px;border:1px solid var(--line);
 text-align:right;white-space:nowrap}
table.sum th:first-child,table.sum td:first-child{text-align:left;
 font-family:ui-monospace,Menlo,monospace}
table.sum thead th{background:var(--code);color:var(--dim);font-size:10.5px;
 letter-spacing:.04em;text-transform:uppercase}
table.sum td.n0{color:var(--line)}
table.sum tr.all td{font-weight:700}
.scroll{overflow-x:auto}
.toc{font-size:.9rem;columns:2;column-gap:28px;margin:1em 0}
.toc a{display:block;padding:1px 0}
.warn{background:var(--nobg);border-left:3px solid var(--no);padding:10px 14px;
 border-radius:0 5px 5px 0;font-size:.88rem;margin:1.1em 0}
.warn p{margin:.3em 0}
"""


def samples_cell(meta: dict) -> str:
    """How many passes this page averages, and when each ran.

    A timing is a claim about a machine at a time, so the page states how
    many readings are behind it rather than presenting a mean as a fact."""
    ss = meta.get("samples") or []
    if len(ss) <= 1:
        one = ss[0] if ss else {}
        span = one.get("span")
        return ("one pass"
                + (f', {esc(span[0].replace("T", " "))} &ndash; '
                   f'{esc(span[1].split("T")[-1])}' if span else "")
                + " &mdash; every timing below is a single reading, not a mean")
    out = [f"<b>{len(ss)} passes</b>, averaged:"]
    for one in ss:
        span = one.get("span")
        out.append(f'<br><code>{esc(one["label"])}</code> &middot; {one["runs"]} runs'
                   + (f' &middot; {esc(span[0].replace("T", " "))} &ndash; '
                      f'{esc(span[1].split("T")[-1])}' if span else
                      " &middot; <i>undated</i>"))
    return "".join(out)


def warnings_section(warns: list, w) -> None:
    """What the numbers do not support, said on the page and not only on the
    terminal that generated it.

    A caveat that lives in the generating script's stderr is a caveat nobody
    reading the page ever sees."""
    if not warns:
        return
    high = [x for x in warns if x.severity == "high"]
    w('<h2 id=caveats>What these numbers do not support</h2>')
    w("<p>Raised by <code>merge.py</code> when the passes were folded together. "
      + ("Nothing here invalidates the table, but each is a place where a "
         "reading is softer than it looks." if not high else
         "<b>The first entries are serious</b> &mdash; they describe passes "
         "that should arguably not be averaged at all.")
      + "</p>")
    for x in sorted(warns, key=lambda x: x.severity != "high"):
        w(f'<div class="cav {"cav-hi" if x.severity == "high" else ""}">')
        w(f'<div class="cav-h"><code>{esc(x.kind)}</code>'
          + (f'<span class="cav-n">{len(x.cells)} cell'
             f'{"s" if len(x.cells) != 1 else ""}</span>' if x.cells else "")
          + "</div>")
        w(f"<p>{esc(x.message)}</p>")
        if x.cells:
            w("<details><summary>which</summary><pre>"
              + esc("\n".join(x.cells[:60]))
              + (f"\n... and {len(x.cells) - 60} more" if len(x.cells) > 60 else "")
              + "</pre></details>")
        w("</div>")


# The directories benchmark files live in, in the order the page lists them.
DIR_ORDER = ["tests/fixtures", "tests/limits/mods", "benchmarks/svcomp/dsl"]
SVCOMP_DSL = PY / "benchmarks" / "svcomp" / "dsl"


def _svcomp_files() -> dict:
    """SV-COMP benchmark name -> its file. The rows name a benchmark, and run it
    through the one adapter, so the file the benchmark *is* has to be found by
    the `name=` it declares."""
    out = {}
    for f in sorted(SVCOMP_DSL.glob("*.py")):
        m = re.search(r'\bname\s*=\s*"([^"]+)"', f.read_text())
        if m:
            out[m.group(1)] = f"benchmarks/svcomp/dsl/{f.name}"
    return out


_SVCOMP = None


def source_of(row: dict) -> str:
    """The benchmark file a row asks about, relative to `python/`."""
    global _SVCOMP
    if row["suite"] == "svcomp":
        if _SVCOMP is None:
            _SVCOMP = _svcomp_files()
        return _SVCOMP.get(row["bench"], row["module"])
    return row["module"]


def _c_loop(docstring: str) -> "str | None":
    """The C loop an SV-COMP benchmark's docstring carries, on one line: whole
    when it is short, its header and `{ … }` when it is not."""
    m = re.search(r"\b(?:while|for)\s*\(", docstring)
    if not m:
        return None
    depth, end = 0, m.end() - 1
    for end in range(m.end() - 1, len(docstring)):
        depth += docstring[end] == "("
        depth -= docstring[end] == ")"
        if depth == 0:
            break
    header = " ".join(docstring[m.start():end + 1].split())
    brace = docstring.find("{", end)
    if brace < 0 or docstring[end + 1:brace].strip():
        return header
    depth = 0
    for close in range(brace, len(docstring)):
        depth += docstring[close] == "{"
        depth -= docstring[close] == "}"
        if depth == 0:
            break
    whole = f"{header} {' '.join(docstring[brace:close + 1].split())}"
    return whole if len(whole) <= 90 else f"{header} {{ … }}"


def describe(path: str) -> str:
    """What the benchmark is, in a line, read from its own docstring.

    A module fixture or a limit-matrix module says so in its first paragraph.
    An SV-COMP benchmark's first line is mostly its own name; what it *is*
    is the C loop underneath, so that is shown, with the author's gloss after
    the dash where the first line has one."""
    try:
        doc = ast.get_docstring(ast.parse((PY / path).read_text())) or ""
    except (OSError, SyntaxError):
        return ""
    first = " ".join(doc.split("\n\n")[0].split())
    if path.startswith("benchmarks/svcomp/"):
        gloss = first.split(" — ", 1)[1] if " — " in first else ""
        loop = _c_loop(doc)
        parts = ([f"<code>{esc(loop)}</code>"] if loop else []) + ([prose(gloss)] if gloss else [])
        return " &mdash; ".join(parts)
    return prose(first)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def verdict_class(verdict: str, truth: "str | None") -> str:
    """The label's colour: green for a right answer, red for a wrong one.

    `VERIFIED` is Lean's, so it is right unless the property is known to fail
    -- which would be a soundness bug, and is drawn as one. `REFUTED` is right
    exactly where the property is known to fail, and wrong where it is known
    to hold. Everything else keeps its legend colour."""
    if verdict == "VERIFIED":
        return "bad" if truth == "fails" else "ok"
    if verdict == "REFUTED" and truth:
        return "ok" if truth == "fails" else "bad"
    return VERDICTS.get(verdict, ("", ""))[0]


def method_row(w, rt: str, run: dict, truth: "str | None" = None) -> None:
    """One route's row: the summary line, and the panel it opens."""
    cls = verdict_class(run["verdict"], truth)
    dis = run.get("disagreed")
    w('<details class="m"><summary>')
    w(f'<span class="name">{esc(rt)}</span>')
    w(f'<span class="v {cls}">{esc(run["verdict"])}</span>'
      + (f'<span class="flag" title="{esc("passes disagreed: " + ", ".join(f"{v} x{c}" for v, c in dis.items()))}">!</span>'
         if dis else ""))
    na = run["verdict"] == "UNSUPPORTED"
    w(f'<span class="t">{"" if na else timing(run["gen"])}</span>')
    w(f'<span class="t">{"" if na else timing(run["build"])}</span>')
    w('<span class="chev">&#9656;</span>')
    w("</summary>")
    w('<div class="body">')
    w(f'<pre>{esc(run["command"])}</pre>')
    if dis:
        w('<div class="lbl">the passes disagreed</div>'
          '<p class="note">'
          + esc(", ".join(f"{v} in {c} pass{'es' if c != 1 else ''}"
                          for v, c in sorted(dis.items())))
          + ". The row shows the most common one; its timing "
            "averages only the runs that reached it.</p>")
    found = run["gen"].get("inferred") or {}
    # Recorded before the harness kept the last candidate a route printed, a
    # multi-attempt run's value is attempt 0's -- possibly one it rejected.
    first = run["gen"].get("inferred_rule") != "last" and rt in ("ai", "ai-cegis")
    for lbl in ("inv", "ranking"):
        if found.get(lbl):
            w(f'<div class="lbl">{lbl} {"it printed first" if first else "it found"}'
              + ('<span class="hint" title="recorded from the first candidate this '
                 'route printed; if it took more than one attempt, the certificate '
                 'built was a later one"> (attempt 0 &mdash; may not be the one built)</span>'
                 if first else "") + "</div>"
              f'<div class="found"><code>{esc(found[lbl])}</code></div>')
    if run["gen"].get("err"):
        w(f'<div class="lbl">{"why verith refused" if na else "why it returned nothing"}</div>')
        w(f'<pre class="err">{esc(run["gen"].get("err_full") or run["gen"]["err"])}</pre>')
    errs, seen = [], set()
    for e in run["build"].get("errors", []):
        if e.startswith("(not built") or e.strip() == "build failed":
            continue
        if e not in seen:
            seen.add(e)
            errs.append(e)
    if errs:
        w('<div class="lbl">lake said</div>')
        for e in errs[:4]:
            w(f'<div class="err">{esc(e)}</div>')
    if run["build"].get("sorries"):
        w('<div class="lbl">obligations left open</div>')
        w('<div class="found">' + esc(", ".join(run["build"]["sorries"])) + "</div>")
    w("</div></details>")


# Where a path in the page's prose is relative to: the prose is written from
# `python/`, with a few names local to this directory or the limit matrix's.
LINK_BASES = [PY, PY.parent, SP, PY / "tests" / "limits", WORK]
# Embedded so the popup works from a copied page and without a server; larger
# or non-text files are linked, not embedded.
EMBED_SUFFIXES = {".py", ".md", ".lean", ".toml", ".txt", ".j2"}
EMBED_MAX = 300_000

# The token kinds the source viewer colours, most specific first; names,
# operators and punctuation keep the text colour.
HL_KINDS = [
    (Token.String.Doc, "d"),
    (Token.Comment, "c"),
    (Token.String, "s"),
    (Token.Keyword, "k"),
    (Token.Operator.Word, "k"),
    (Token.Name.Builtin, "b"),
    (Token.Name.Function, "f"),
    (Token.Name.Class, "f"),
    (Token.Name.Decorator, "a"),
    (Token.Number, "n"),
    (Token.Generic.Heading, "h"),
    (Token.Generic.Subheading, "h"),
]
# The whole source viewer -- line numbering as well as colour -- so that the
# in-page dialog and the popup window, which has no other stylesheet, lay a
# file out identically; the popup reads it from the page's `<style id="srccss">`.
SRC_CSS = """
.src{--hk:#7b36a8;--hs:#2f6f3a;--hd:#4d6b3c;--hc:#736f67;--hn:#a14f00;--hf:#245b91;
 --hb:#17707c;--ha:#8a6100;--hln:#9a968d}
@media (prefers-color-scheme:dark){.src{--hk:#c79be6;--hs:#9fcf8c;--hd:#a8bd8a;
 --hc:#8d8a82;--hn:#e6a86e;--hf:#8fb8de;--hb:#79c5cf;--ha:#d9b65e;--hln:#6f6b64}}
pre.src{margin:0;padding:8px 0;counter-reset:ln;background:transparent;
 font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre.src .ln{display:block;padding-right:14px;white-space:pre-wrap;word-break:break-word}
pre.src .ln::before{counter-increment:ln;content:counter(ln);display:inline-block;
 width:3.2em;margin-right:1em;text-align:right;color:var(--hln);user-select:none}
.src .k{color:var(--hk)}.src .s{color:var(--hs)}.src .d{color:var(--hd)}
.src .c{color:var(--hc);font-style:italic}.src .n{color:var(--hn)}
.src .f,.src .h{color:var(--hf)}.src .h{font-weight:600}.src .b{color:var(--hb)}
.src .a{color:var(--ha)}
"""


def highlight(name: str, text: str) -> str:
    """`text` as HTML, its tokens in `<span class=…>` by `HL_KINDS`. No span
    crosses a line, so the viewer numbers lines by splitting on newlines."""
    try:
        lexer = get_lexer_for_filename(name, stripnl=False, ensurenl=False)
    except ClassNotFound:
        lexer = TextLexer(stripnl=False, ensurenl=False)
    out, open_kind = [], None
    for ttype, value in lexer.get_tokens(text):
        kind = next((k for t, k in HL_KINDS if ttype in t), None)
        for i, part in enumerate(value.split("\n")):
            if i:
                if open_kind:
                    out.append("</span>")
                    open_kind = None
                out.append("\n")
            if not part:
                continue
            if kind != open_kind and not (kind is None and part.isspace()):
                if open_kind:
                    out.append("</span>")
                if kind:
                    out.append(f'<span class="{kind}">')
                open_kind = kind
            out.append(html.escape(part, quote=False))
    if open_kind:
        out.append("</span>")
    return "".join(out)


def resolve(text: str) -> "Path | None":
    """The file or directory `text` names, if it names one."""
    text = text.strip()
    if not text or " " in text or len(text) > 200:
        return None
    cand = Path(text)
    if cand.is_absolute():
        return cand if cand.exists() else None
    for base in LINK_BASES:
        full = (base / cand).resolve()
        if full.exists():
            return full
    return None


class Files:
    """The files the page links, and the highlighted text of those it embeds."""

    def __init__(self):
        self.text: dict = {}

    def link(self, full: Path, label: str, cls: str = "file") -> str:
        href = "file://" + str(full) + ("/" if full.is_dir() else "")
        if full.is_file() and full.suffix in EMBED_SUFFIXES and full.stat().st_size <= EMBED_MAX:
            key = str(full.relative_to(PY.parent)) if full.is_relative_to(PY.parent) else str(full)
            if key not in self.text:
                self.text[key] = highlight(full.name, full.read_text(errors="replace"))
            return (f'<a class="{cls}" href="{esc(href)}" data-src="{esc(key)}" '
                    f'title="open {esc(key)}">{label}</a>')
        return (f'<a class="{cls}" href="{esc(href)}" target="_blank" '
                f'title="open {esc(str(full))}">{label}</a>')

    def linkify(self, page: str) -> str:
        """Every `<code>` in `page` whose whole text names a file or directory,
        as a link to it. Commands are in `<pre>`, not `<code>`, so they stay
        pasteable."""
        def one(m):
            full = resolve(html.unescape(m.group(1)))
            return self.link(full, m.group(0)) if full else m.group(0)
        return re.sub(r"<code>([^<]{2,200})</code>", one, page)

    def payload(self) -> str:
        return (json.dumps(self.text, ensure_ascii=False)
                .replace("</", "<\\/").replace("<!--", "<\\!--"))


VIEWER_JS = r"""
(function () {
  var SRC = JSON.parse(document.getElementById('srcs').textContent);
  function esc(t) {
    return t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  // SRC holds each file already highlighted, a span never crossing a line.
  // The spans are blocks and join with nothing between them: a newline here
  // would be one more line box inside the <pre>, doubling every gap.
  function lines(src) {
    return src.replace(/\n$/, '').split('\n').map(function (l) {
      return '<span class="ln">' + l + '</span>';
    }).join('');
  }
  var STYLE =
    ':root{color-scheme:light dark}' +
    'body{margin:0;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;' +
    'background:#fbfbf9;color:#1c1b19}' +
    'header{position:sticky;top:0;padding:9px 14px;background:#f3f1ec;' +
    'border-bottom:1px solid #e2ded6;display:flex;gap:12px;align-items:baseline}' +
    'header b{font-size:13px}header a{font-size:11.5px;color:#2f5d8a}' +
    '@media(prefers-color-scheme:dark){body{background:#161614;color:#e7e4dc}' +
    'header{background:#24241f;border-color:#2e2d29}header a{color:#8fb4dc}}' +
    document.getElementById('srccss').textContent;
  function doc(key, href, text) {
    return '<!doctype html><meta charset="utf-8"><title>' + esc(key) + '</title>' +
      '<style>' + STYLE + '</style><header><b>' + esc(key) + '</b>' +
      '<a href="' + esc(href) + '">open the file itself</a></header>' +
      '<pre class="src">' + lines(text) + '</pre>';
  }
  function inPage(key, href, text) {
    var d = document.getElementById('srcdlg');
    d.querySelector('.dlg-title').textContent = key;
    d.querySelector('.dlg-raw').setAttribute('href', href);
    d.querySelector('pre').innerHTML = lines(text);
    d.showModal();
  }
  document.addEventListener('click', function (e) {
    var a = e.target.closest && e.target.closest('a[data-src]');
    if (!a || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    var key = a.getAttribute('data-src'), text = SRC[key];
    if (text == null) return;
    e.preventDefault();
    var href = a.getAttribute('href'), win = null;
    try {
      win = window.open('', 'verith-source', 'popup,width=860,height=780');
      if (win) {
        win.document.open();
        win.document.write(doc(key, href, text));
        win.document.close();
        win.focus();
        return;
      }
    } catch (err) {
      if (win) { try { win.close(); } catch (_) {} }
    }
    inPage(key, href, text);
  });
  document.getElementById('srcdlg').addEventListener('click', function (e) {
    if (e.target === this || e.target.classList.contains('dlg-close')) this.close();
  });
})();
"""


def render(data: dict, warns: list = ()) -> str:
    meta, rows, runs = data["meta"], data["rows"], data["runs"]
    m = meta["machine"]
    files = Files()
    o = []
    w = o.append

    w("<title>verith --infer route matrix</title>")
    w(f"<style>{CSS}</style>")
    w('<div class="wrap">')
    w("<h1>The <code>--infer</code> route matrix</h1>")
    w('<p class="lede">Every <code>--infer</code> route of <code>verith</code>, '
      "put to every benchmark and property in the tree. One cell is one "
      "<code>uv run verith</code> followed by one <code>lake build</code>, and both are "
      "timed &mdash; click a method to get the command back.</p>")
    w('<p class="lede">A command pasted from a cell <em>generates</em>; building what '
      "it emits needs the shared <code>.lake</code> that keeps Mathlib from being "
      'rebuilt, which is four more lines &mdash; see <a href="#reproduce">Reproducing '
      "this</a>.</p>")
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    done = len(runs)
    w(f'<p class="stamp">Rendered {stamp} &middot; {len(rows)} properties over '
      f'{len({source_of(r) for r in rows.values()})} benchmarks &middot; {done} measured runs</p>')

    w('<p class="jump"><a href="#summary">Jump to the results &darr;</a></p>')

    # ── the routes ──────────────────────────────────────────────────────
    w("<h2 id=methods>The methods</h2>")
    w("<p>A route&rsquo;s own row in <code>zrth/lean/infer_route.py</code> is what "
      "describes it below; the CLI reads the same rows, so this cannot drift from "
      "what <code>--infer</code> actually accepts. What a route needs of the property "
      "is the first tag: a <code>--buchi</code> certificate needs an invariant "
      "<em>and</em> a ranking function, a <code>--safety</code> one needs only an "
      "invariant, and several routes do just one of the two.</p>")
    w('<div class="cards">')
    for r in route_docs():
        kinds = " ".join(f'<span class=tag>--{k}</span>' for k in r["kinds"])
        llm = '<span class="tag llm">LLM</span>' if r["llm"] else ""
        w('<div class="card">')
        w(f'<h4><code>{esc(r["name"])}</code>{kinds}{llm}</h4>')
        w(f"<p>{prose(r['summary'])}</p>")
        bits = []
        if r["seeds"]:
            bits.append("seeds from " + ", ".join(f"<code>{esc(s)}</code>"
                                                  for s in r["seeds"]))
        if r["opts"]:
            bits.append("tunable with " + ", ".join(f"<code>{esc(f)}</code>" for f in r["opts"]))
        if r.get("kinds_refusal"):
            bits.append("refuses the other kind, and says why")
        if bits:
            w(f"<p>{'; '.join(bits).capitalize()}.</p>")
        w("</div>")
    w("</div>")
    w('<div class="warn"><p><b>Two routes were measured with outside help.</b> '
      f'<code>ai</code> and <code>ai-cegis</code> called <code>{esc(meta["model"])}</code> '
      "over the Anthropic API, so their timings include network latency and are not "
      "reproducible run to run &mdash; a different pass may find a different "
      "certificate, or none. <code>fbk-proveit</code> ran against the "
      f'<code>{esc(Path(meta["proveit_dir"]).name)}</code> checkout and the ic3ia binary '
      f'at <code>{esc(meta["ic3ia"])}</code>.'
      + (f' <code>vampire</code> ran the prover at <code>{esc(meta["vampire"])}</code>.'
         if meta.get("vampire") else "")
      + "</p></div>")

    # ── machine ─────────────────────────────────────────────────────────
    w("<h2 id=machine>The machine, and what the cold start costs</h2>")
    w('<table class="kv">')
    core = (f'{m["cores"]} cores'
            + (f' ({m["perf_cores"]}P + {m["eff_cores"]}E)' if m.get("perf_cores") else ""))
    for k, v in [
        ("CPU", f'{esc(m["cpu"])} &middot; {core}'),
        ("Memory", f'{m["memory_gb"]} GB'),
        ("OS", f'macOS {esc(m["os_product"])} ({esc(m["os_build"])}), '
               f'Darwin {esc(m["os"].split()[-1])}'),
        ("Lean", f'<code>{esc(m["toolchain"])}</code> &middot; {esc(m["lake"])}'),
        ("Python", esc(m["python"])),
        ("Timeouts", f'{meta["gen_timeout"]} s per <code>verith</code>, '
                     f'{meta["build_timeout"]} s per <code>lake build</code>'),
        ("Samples", samples_cell(meta)),
    ]:
        w(f"<tr><td>{k}</td><td>{v}</td></tr>")
    cold = meta.get("coldstart", {})
    if cold:
        w(f'<tr><td>Shared <code>.lake</code></td><td>{m["lake_packages_gb"]} GB, '
          f'{len(m["packages"])} packages &mdash; '
          + ", ".join(f'<code>{esc(n)}</code> {esc(v)}'
                      for n, v in list(m["packages"].items())[:4])
          + ", &hellip;</td></tr>")
        for k, v in [
            ("Mathlib, cold", cold.get("mathlib_note", "")),
            ("First project in a fresh build dir",
             (f'<b>{cold["first_build_s"]:.0f} s</b> &mdash; '
              + cold.get("first_build_note", "")) if cold.get("first_build_s") else ""),
            ("Each project after it",
             f'{cold["warm_build_s"]:.1f} s' if cold.get("warm_build_s") else ""),
            ("No-op <code>lake build</code>",
             f'{cold["noop_build_s"]:.1f} s' if cold.get("noop_build_s") else ""),
        ]:
            if v:
                w(f"<tr><td>{k}</td><td>{v}</td></tr>")
    w("</table>")
    if cold.get("explain"):
        w(f"<p>{cold['explain']}</p>")
    w('<div class="warn"><p><b>The matrix is strictly serial, and must be.</b> '
      "Every generated project symlinks its <code>.lake</code> to one shared build "
      "dir, and module names are identical across projects, so two concurrent runs "
      "overwrite each other&rsquo;s oleans. It does not fail cleanly: it reads as "
      "<code>unknown constant &#39;hrank&#39;</code>.</p></div>")

    # ── verdicts ────────────────────────────────────────────────────────
    w("<h2 id=verdicts>Verdicts</h2>")
    w("<p>A label is <b>green when the answer is right</b>: a <code>VERIFIED</code> "
      "(Lean checked it), and a <code>REFUTED</code> on a property known to be false. "
      "It is red when the answer contradicts what is known about the property. Where "
      "that is known, the property carries a <em>known to hold</em> or <em>known "
      "false</em> tag: the fbk probes' declared outcome, the fixtures' own docstrings, "
      "and Houdini's invariants, which hold by construction.</p>")
    w('<table class="kv">')
    for name, (cls, blurb) in VERDICTS.items():
        n = sum(1 for r in runs.values() if r["verdict"] == name)
        w(f'<tr><td><span class="v {cls}">{name}</span></td>'
          f'<td>{blurb} <b>{n}</b> of {done}.</td></tr>')
    w("</table>")

    # ── the route columns, and what merge.py flagged ────────────────────
    route_order = [r["name"] for r in route_docs()]
    present = [r for r in route_order if any(x["route"] == r for x in runs.values())]
    warnings_section(list(warns), w)

    # ── the matrix ──────────────────────────────────────────────────────
    # Where the properties come from and an index into them, then the
    # scoreboard -- what each route carried, by suite -- and then the same
    # measurements one at a time: benchmark file, the properties asked of it
    # (from whichever suite asked them, since the fbk probes and the limit
    # matrix share modules), then one row per `--infer` route, every route
    # on every property.
    w("<h2 id=matrix>The matrix</h2>")
    suites = [s for s in SUITE_BLURB if any(v["suite"] == s for v in rows.values())]
    w("<p>Where the properties come from:</p><dl class=suites>")
    for s_ in suites:
        w(f"<dt>{esc(s_)}</dt><dd>{SUITE_BLURB[s_]}</dd>")
    w("</dl>")

    by_file: dict = {}
    for key, row in rows.items():
        by_file.setdefault(source_of(row), []).append((key, row))
    dirs: dict = {}
    for path in by_file:
        dirs.setdefault(str(Path(path).parent), []).append(path)
    dir_order = sorted(dirs, key=lambda d: (DIR_ORDER.index(d) if d in DIR_ORDER else 99, d))

    w('<div class="toc">')
    for d in dir_order:
        w(f'<div class="tocdir"><a href="#d-{slug(d)}"><code>{esc(d)}/</code></a> '
          f'&mdash; {len(dirs[d])} benchmarks<div class="tocb">'
          + " ".join(f'<a href="#b-{slug(f)}">{esc(Path(f).stem)}</a>'
                     for f in sorted(dirs[d]))
          + "</div></div>")
    w("</div>")

    w("<h2 id=summary>Summary</h2>")
    w("<p>How many of each suite&rsquo;s properties each route certified end to end "
      "&mdash; Lean discharged every obligation &mdash; out of the properties whose "
      "kind that route accepts. Every cell behind these counts is below it, one "
      "property at a time.</p>")
    w('<div class="scroll"><table class="sum"><thead><tr><th>suite</th>')
    for r in present:
        w(f"<th>{esc(r)}</th>")
    w("</tr></thead><tbody>")
    for suite in suites + ["all"]:
        w(f"<tr{' class=all' if suite == 'all' else ''}><td>{esc(suite)}</td>")
        for rt in present:
            ks = [k for k, v in rows.items() if suite in (v["suite"], "all")]
            got = [runs[f"{k}::{rt}"] for k in ks if f"{k}::{rt}" in runs
                   and runs[f"{k}::{rt}"]["verdict"] != "UNSUPPORTED"]
            ok = sum(1 for g in got if g["verdict"] == "VERIFIED")
            w(f'<td class="{"n0" if not got else ""}">'
              + (f"{ok} / {len(got)}" if got else "&mdash;") + "</td>")
        w("</tr>")
    w("</tbody></table></div>")

    suite_rank = {s_: i for i, s_ in enumerate(SUITE_BLURB)}
    for d in dir_order:
        w(f'<h3 class="dir" id="d-{slug(d)}"><code>{esc(d)}/</code></h3>')
        for path in sorted(dirs[d]):
            items = sorted(by_file[path], key=lambda kv: (suite_rank.get(kv[1]["suite"], 9), kv[0]))
            w(f'<div class="bench" id="b-{slug(path)}">')
            w(f'<div class="bh"><span class="bname">{esc(Path(path).stem)}</span>'
              + files.link((PY / path).resolve(), esc(path), "file path")
              + f'<span class="bcount">{len(items)} '
                f'{"property" if len(items) == 1 else "properties"}</span></div>')
            desc = describe(path)
            if desc:
                w(f'<p class="desc">{desc}</p>')
            for key, row in items:
                w('<div class="prop">')
                w('<div class="head">')
                w(f'<b>{esc(row["prop_label"] or row["kind"])}</b>')
                w(f'<span class=tag>--{esc(row["kind"])}</span>')
                w(f'<span class="tag suite">{esc(row["suite"])}</span>')
                if row.get("truth"):
                    w(f'<span class="tag truth {esc(row["truth"])}" title="what is known '
                      f'about this property, independently of any route">'
                      f'{"known to hold" if row["truth"] == "holds" else "known false"}</span>')
                w("</div>")
                w(f'<code class="smt">{esc(row["prop"])}</code>')
                if row.get("pre"):
                    w(f'<p class="note">under the precondition <code>{esc(row["pre"])}</code> '
                      "on the inputs, passed as <code>--pre</code></p>")
                note = row["note"].split("; the cases sharing it")[0]
                if note:
                    w(f'<p class="note">{esc(note)}</p>')
                if row.get("shared"):
                    sh = row["shared"]
                    w(f'<p class="note">Asked once for all {len(sh) + 1} cases that '
                      "differ only in the predicates they supply: "
                      + ", ".join(f"<code>{esc(n)}</code>" for n in sh[:6])
                      + (f" and {len(sh) - 6} more" if len(sh) > 6 else "") + ".</p>")
                w('<div class="grid">')
                w("<div class=hd><span>method</span><span>verdict</span>"
                  "<span>gen&nbsp;s</span><span>build&nbsp;s</span><span></span></div>")
                for rt in route_order:
                    run = runs.get(f"{key}::{rt}")
                    if not run:
                        w(f'<div class="m missing"><span class="name">{esc(rt)}</span>'
                          '<span class="v na">not measured</span>'
                          "<span></span><span></span><span></span></div>")
                        continue
                    method_row(w, rt, run, row.get("truth"))
                w("</div></div>")
            w("</div>")

    w('<h2 id=reproduce>Reproducing this</h2>')
    w("<p>Both halves are one command each, from <code>python/</code>. The measuring "
      "pass is resumable &mdash; a pair already in <code>results.json</code> is skipped "
      "&mdash; so it can be stopped and restarted, and a single route re-measured "
      "without disturbing its neighbours.</p>")
    w("<pre>cd python\n\n"
      "# measure: one verith + one lake build per cell, strictly serially\n"
      "uv run python tests/bench_matrix/run_matrix.py\n"
      "uv run python tests/bench_matrix/run_matrix.py --suites limits --routes nuterm\n"
      "uv run python tests/bench_matrix/run_matrix.py --only Countdown --redo\n\n"
      "# measure the cold start (needs a quiet machine; writes its own build dir)\n"
      "uv run python tests/bench_matrix/coldstart.py\n\n"
      "# render this page\n"
      "uv run python tests/bench_matrix/render.py -o matrix.html</pre>")
    w('<div class="warn"><p><b>Wall clock is only as quiet as the machine.</b> '
      "Another Lean build anywhere is enough to inflate these numbers several-fold, "
      "and unevenly: one earlier pass reported a case at 1996 s that re-timed at 75 s. "
      "Spotlight is the worst offender &mdash; a pass writes tens of thousands of "
      "<code>.olean</code> files &mdash; which is why the work directory is named "
      "<code>.noindex</code>. Verdicts were never affected by any of it; only timings "
      "lie. Re-time anything surprising before believing it.</p></div>")
    w("<h3>Building what one cell emits</h3>")
    w("<p>A command from a cell writes a Lean project and stops. To build it the way "
      "the matrix does &mdash; against the already-built Mathlib, so nothing resolves "
      "or recompiles it &mdash; point the project&rsquo;s <code>.lake</code> at the "
      "shared one and give it the manifest that goes with it:</p>")
    w("<pre>P=/tmp/verith-out.noindex/&lt;cell&gt;/Rea      # the -o of the pasted command\n"
      "mkdir -p /tmp/shared.noindex\n"
      "ln -sfn \"$PWD/tests/lean/.lake/packages\" /tmp/shared.noindex/packages\n"
      "ln -sfn /tmp/shared.noindex \"$P/.lake\"\n"
      "cp tests/limits/lake-manifest.json \"$P/\"\n"
      "(cd \"$P\" &amp;&amp; lake build)</pre>")
    w("<p>Two things that are not optional. The <code>.noindex</code> suffix &mdash; a "
      "build writes tens of thousands of <code>.olean</code> files and Spotlight "
      "indexing them is what turns a 9&nbsp;s case into a 928&nbsp;s one. And "
      "<b>one build directory takes one writer</b>: module names are identical across "
      "generated projects, so a second build into the same shared directory serves the "
      "first one&rsquo;s oleans, and the symptom is <code>unknown constant "
      "&#39;hrank&#39;</code> rather than a clean failure. "
      "The <code>fbk-proveit</code> column needs one line more, because its lakefile "
      "adds a path require: <code>lake update LTL_Certifying</code> before the build.</p>")
    w("<p>The prose above is generated too: the method cards come from "
      "<code>zrth/lean/infer_route.py</code>, the machine block and every timing from "
      "<code>results.json</code>. See "
      "<code>tests/bench_matrix/README.md</code> for what to do when a number surprises "
      "you.</p>")
    w("</div>")
    w('<dialog id="srcdlg"><div class="dlg-bar"><b class="dlg-title"></b>'
      '<a class="dlg-raw" href="#">open the file itself</a>'
      '<button class="dlg-close" type="button">close</button></div>'
      '<pre class="src"></pre></dialog>')
    page = files.linkify("\n".join(o))
    return (page
            + f'\n<style id="srccss">{SRC_CSS}</style>'
            + f'\n<script type="application/json" id="srcs">{files.payload()}</script>'
            + f"\n<script>{VIEWER_JS}</script>")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render the matrix page from one or more measurement passes.",
        epilog="Several -r files are averaged per cell; see merge.py for what "
               "is averaged and what is flagged instead.")
    ap.add_argument("-o", "--out", default=str(SP / "matrix.html"))
    ap.add_argument("-r", "--results", nargs="+", default=[str(WORK / "results.json")],
                    metavar="PATH", help="results file(s) to average")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any high-severity warning was raised")
    args = ap.parse_args()

    data, warns = merge(load(args.results))
    Path(args.out).write_text(render(data, warns))

    # The page carries these too, but whoever ran the script should not have
    # to open it to find out that the passes disagreed.
    for x in warns:
        print(x.line(), file=sys.stderr)
    n = data["meta"]["n_passes"]
    print(f"wrote {args.out}  ({Path(args.out).stat().st_size / 1024:.0f} KB, "
          f"{len(data['runs'])} cells from {n} pass{'es' if n != 1 else ''})")
    if args.strict and any(x.severity == "high" for x in warns):
        raise SystemExit(f"error: {sum(x.severity == 'high' for x in warns)} "
                         f"high-severity warning(s); page written anyway")


if __name__ == "__main__":
    main()
