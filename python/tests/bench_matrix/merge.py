"""Fold several measurement passes into one averaged dataset, and say what
about it should not be believed.

One pass is one sample. A timing from one sample is a timing of one machine on
one afternoon, and this project learned the hard way what that is worth: a pass
measured beside two runaway processes reads as a slower machine everywhere,
uniformly enough that nothing in the numbers looks wrong. Averaging several
passes is the cheap defence, and flagging the cells that disagree between
passes is the rest of it.

What is averaged
================
`gen.secs` and `build.secs`, over the runs that **agree on the verdict**. A
cell that timed out contributes no seconds to a mean: its 300.0 is the budget,
not a measurement of the work, and averaging a censored observation with a
completed one produces a number that describes neither. Everything else on a
cell -- the command, the certificate the route found, the error text -- is taken
from the first run holding the modal verdict, so the page shows a coherent
single run beside the averaged timing.

What is flagged
===============
:func:`warnings_for` is the whole list, each with a `kind`, a `severity` and
the cells involved. The two that matter most are `verdict-disagreement` (the
same question answered differently on different days, which no average can
paper over) and `overlapping-passes` (two passes whose runs interleave in
wall-clock time, so they were racing each other and *both* samples are of a
busier machine than either reports).

Interface
=========
:func:`merge` returns `(data, warnings)` where `data` has the shape
`render.py` already reads, with `gen`/`build` carrying `secs` plus `n`,
`spread` and `samples`. A single input file goes through unchanged apart from
those fields, so the one-pass and many-pass paths are the same path.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, median

# A cell whose completed runs disagree by more than this, relative to their
# mean, is reported. Short cells are noisy for reasons that are not
# interesting (process start-up dominates), so a cell must also be slower
# than SPREAD_FLOOR before its spread counts.
SPREAD_FRAC = 0.30
SPREAD_FLOOR = 2.0        # seconds
# A run this close to its budget is one bad afternoon away from a TIMEOUT,
# which would make it vanish from the mean rather than raise it.
NEAR_BUDGET = 0.80


@dataclass
class Warning_:
    kind: str
    severity: str            # "high" | "low"
    message: str
    cells: list = field(default_factory=list)

    def line(self) -> str:
        n = f"  ({len(self.cells)} cell{'s' if len(self.cells) != 1 else ''})" if self.cells else ""
        return f"[{self.severity:<4}] {self.kind}: {self.message}{n}"


def load(paths) -> list:
    """The passes, each tagged with where it came from."""
    out = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise SystemExit(f"error: {p} does not exist")
        d = json.loads(p.read_text())
        for need in ("meta", "rows", "runs"):
            if need not in d:
                raise SystemExit(f"error: {p} has no {need!r}: not a results file")
        d["meta"].setdefault("label", p.stem)
        d["_path"] = str(p)
        out.append(d)
    return out


# ── averaging ──────────────────────────────────────────────────────────────

def _secs(runs, half: str):
    """The mean of `half`'s seconds over `runs`, with its spread."""
    vals = [r[half]["secs"] for r in runs if half in r and r[half].get("secs") is not None]
    if not vals:
        return dict(secs=0.0, n=0, spread=0.0, samples=[])
    return dict(secs=fmean(vals), n=len(vals),
                spread=(max(vals) - min(vals)) if len(vals) > 1 else 0.0,
                samples=[round(v, 2) for v in vals])


def merge(datas: list):
    """`datas` folded into one dataset, plus what to warn about.

    A single input is the same code path as many, so the page cannot render
    differently depending on how many passes it was given."""
    rows: dict = {}
    for d in datas:                      # later passes win on row metadata,
        rows.update(d["rows"])           # which is derived, not measured
    meta = dict(datas[-1]["meta"])
    meta["samples"] = [dict(label=d["meta"].get("label", "?"), path=d["_path"],
                            runs=len(d["runs"]),
                            span=_span(d["runs"])) for d in datas]
    meta["n_passes"] = len(datas)

    by_pair: dict = {}
    for d in datas:
        for pair, run in d["runs"].items():
            by_pair.setdefault(pair, []).append(run)

    runs: dict = {}
    for pair, got in by_pair.items():
        counts = Counter(r["verdict"] for r in got)
        # Ties go to the verdict of the most recent pass holding one of them,
        # so a repeat measurement breaks a tie rather than the dict order.
        top = max(counts.values())
        modal = [v for v, c in counts.items() if c == top]
        verdict = next((r["verdict"] for r in reversed(got) if r["verdict"] in modal),
                       modal[0])
        agree = [r for r in got if r["verdict"] == verdict]
        rep = dict(agree[0])
        rep["gen"] = dict(agree[0]["gen"], **_secs(agree, "gen"))
        rep["build"] = dict(agree[0]["build"], **_secs(agree, "build"))
        rep["verdict"] = verdict
        rep["n"] = len(agree)
        if len(counts) > 1:
            rep["disagreed"] = dict(counts)
        runs[pair] = rep

    data = dict(meta=meta, rows=rows, runs=runs)
    return data, warnings_for(datas, data)


def _describe(field_: str, value):
    """`value` short enough to name in a warning.

    The machine record is a dozen fields; what distinguishes two of them in
    practice is the chip and the core count, so that is what is quoted."""
    if field_ == "machine" and isinstance(value, dict):
        return f'{value.get("cpu", "?")} ({value.get("cores", "?")} cores)'
    return str(value)


def _span(runs: dict):
    """The wall-clock span a pass's runs cover, if they were stamped."""
    when = sorted(r["when"] for r in runs.values() if r.get("when"))
    return [when[0], when[-1]] if when else None


# ── what not to believe ────────────────────────────────────────────────────

def warnings_for(datas: list, merged: dict) -> list:
    out = []
    meta, runs = merged["meta"], merged["runs"]

    # -- the passes are not comparable to each other ------------------------
    for field_, what in (("machine", "the machine"),
                         ("model", "the model behind the AI routes"),
                         ("gen_timeout", "the generation budget"),
                         ("build_timeout", "the build budget")):
        groups: dict = {}
        for d in datas:
            key = json.dumps(d["meta"].get(field_), sort_keys=True)
            groups.setdefault(key, []).append(d["meta"].get("label", "?"))
        if len(groups) > 1:
            shown = "; ".join(
                f"{', '.join(labels)} -> {_describe(field_, json.loads(key))}"
                for key, labels in groups.items())
            out.append(Warning_(
                f"{field_.replace('_', '-')}-mismatch", "high",
                f"the passes disagree about {what}, so a mean over them "
                f"describes no single configuration: {shown}."))

    # -- the passes overlapped in time, so they contended -------------------
    spans = [(d["meta"].get("label", "?"), _span(d["runs"])) for d in datas]
    dated = [(lbl, s) for lbl, s in spans if s]
    for i, (la, sa) in enumerate(dated):
        for lb, sb in dated[i + 1:]:
            if sa[0] <= sb[1] and sb[0] <= sa[1]:
                out.append(Warning_(
                    "overlapping-passes", "high",
                    f"passes {la!r} and {lb!r} were running at the same time "
                    f"({sa[0]}..{sa[1]} vs {sb[0]}..{sb[1]}). The harness is "
                    f"serial for a reason -- two passes at once measure a "
                    f"machine under load, and averaging them does not undo it."))
    if len(dated) < len(spans):
        out.append(Warning_(
            "unstamped-runs", "low",
            f"{len(spans) - len(dated)} of {len(spans)} passes have runs "
            f"without a `when` stamp, so they cannot be placed in time and "
            f"the overlap check above cannot see them."))

    # -- individual cells ---------------------------------------------------
    disagreed, spread, near, thin = [], [], [], []
    for pair, r in sorted(runs.items()):
        if r.get("disagreed"):
            disagreed.append(f"{pair} ({', '.join(f'{v}x{c}' for v, c in r['disagreed'].items())})")
        for half, budget in (("gen", meta.get("gen_timeout")),
                             ("build", meta.get("build_timeout"))):
            h = r.get(half, {})
            if h.get("n", 0) > 1 and h["secs"] >= SPREAD_FLOOR \
                    and h["spread"] / h["secs"] > SPREAD_FRAC:
                spread.append(f"{pair} {half} {h['samples']}")
            if budget and h.get("secs", 0) >= NEAR_BUDGET * budget \
                    and r["verdict"] != "TIMEOUT":
                near.append(f"{pair} {half} {h['secs']:.0f}s of {budget}s")
        # A disagreeing cell always has fewer agreeing runs than passes, but
        # that is the disagreement being reported above, not thin coverage.
        if r.get("n", 1) < meta["n_passes"] and not r.get("disagreed"):
            thin.append(f"{pair} ({r.get('n', 1)}/{meta['n_passes']})")

    if disagreed:
        out.append(Warning_(
            "verdict-disagreement", "high",
            "the same pair came out differently in different passes, so the "
            "verdict shown is the modal one and the timing averages only the "
            "runs that agreed with it. A route that is not deterministic "
            "cannot be summarised by one row.", disagreed))
    if spread:
        out.append(Warning_(
            "high-spread", "low",
            f"the agreeing runs differ by more than {SPREAD_FRAC:.0%} of their "
            f"mean, so the mean is not a tight estimate.", spread))
    if near:
        out.append(Warning_(
            "near-budget", "low",
            f"within {1 - NEAR_BUDGET:.0%} of the timeout, so another pass may "
            f"record TIMEOUT instead and drop the cell out of the mean.", near))
    if thin:
        out.append(Warning_(
            "uneven-coverage", "low",
            "measured in fewer passes than were given -- either the pass was "
            "scoped with --suites/--routes/--only, or it did not finish.", thin))
    if meta["n_passes"] == 1:
        out.append(Warning_(
            "single-sample", "low",
            "one pass, so every timing is a single observation and nothing "
            "here is averaged. Run again with --results pointing at a new "
            "file and pass both to this script."))
    return out


def main() -> None:
    """Check what a set of passes looks like before rendering them.

    Same folding, same warnings, no page -- for deciding whether a repeat
    pass is comparable to the one it is meant to join."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", nargs="+", help="results files to fold")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on any high-severity warning")
    args = ap.parse_args()

    data, warns = merge(load(args.results))
    meta, runs = data["meta"], data["runs"]
    print(f"{len(runs)} cells from {meta['n_passes']} pass"
          f"{'es' if meta['n_passes'] != 1 else ''}")
    for one in meta["samples"]:
        span = f"{one['span'][0]} .. {one['span'][1]}" if one["span"] else "undated"
        print(f"  {one['label']:<24} {one['runs']:>5} runs   {span}")

    ns = Counter(r.get("n", 1) for r in runs.values())
    print("\ncells by how many passes agreed on them:")
    for n in sorted(ns):
        print(f"  {n} pass{'es' if n != 1 else ''}: {ns[n]}")
    tot = [r["gen"]["secs"] + r["build"]["secs"] for r in runs.values()]
    if tot:
        print(f"\nper-cell gen+build: median {median(tot):.1f}s, "
              f"mean {fmean(tot):.1f}s, max {max(tot):.1f}s")

    print()
    for x in warns:
        print(x.line())
    if args.strict and any(x.severity == "high" for x in warns):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
