"""The generated project's `artifacts/` directory: what a run leaves behind.

A run that does not prove the property still produces something.  An
invariant that was inductive but too weak, a ranking function that dropped
on the right rounds but used an op the certificate cannot express, the cvc5
model that refuted attempt four -- today all of it is lost with the process,
and the next `uv run verith` starts from nothing.  Kept in `artifacts/`, it
is what the next run starts from: strengthen this invariant rather than
search for one, or take it as given and infer only the ranking function.

That makes the project a *workspace* rather than an output, and the two
things that follow from it are the whole of this module's design.

**An artifact is a file, and the metadata is beside it.**  Artifacts are
arbitrary -- `.smt2` predicates, `.md` notes, whatever a route finds worth
writing, including files a subprocess dropped there.  So the index
(`artifacts/index.json`) types what it knows and the store reconciles it
with what is actually on disk: a file no entry describes is still listed,
as `role="other"`, because a route that understands it should still find it.

**Metadata says what was wrong with it, machine-readably and in prose.**
`status` is what a consumer filters on; `why` is what it puts in a prompt or
shows a human.  The same split the route table uses: the field the code
tests, and the sentence a person reads.

Three fields exist only to stop a stale artifact being trusted.  An
invariant found for `G (s0 < 10)` is not an invariant for `G (s0 < 5)`
(`prp`), an invariant is not a ranking function's obligation (`kind`),
and neither survives the module being edited (`module_digest`).  A workspace
reused across a changed module or a changed property is the failure mode
that would otherwise produce a confident certificate about the wrong thing,
so `usable()` filters on all three and says what it dropped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ARTIFACTS_DIR = "artifacts"
_INDEX = "index.json"
_README = "README.md"

# What an artifact is.  `other` is not a gap: an arbitrary file a route
# understands and the table does not is exactly what it is for.
#
# The last three are not candidate certificates but the *questions* a run
# asked: the module as some solver was given it, the property it was asked
# about, and one obligation apiece.  Nothing resumes from them -- they are
# here because a run that reports `REFUTED` should leave behind the file that
# says so, not only the word.
ROLES = ("inv", "ranking", "note", "other", "system", "property", "obligation")

# Why it is in `artifacts/` rather than in the certificate.  What a consumer
# filters on; `why` carries the prose.
STATUSES = (
    "proved",           # it certifies the property -- kept for the record
    "too_weak",         # inductive, but does not imply the property
    "not_inductive",    # refuted at init or at the step
    "unsupported",      # true, but uses something the certificate cannot express
    "refuted",          # a counterexample is known, in `why`
    "unknown",          # the procedure did not finish, or nothing said
    "encoded",          # not a candidate at all: a question this run asked
)

# The suffix a language is *written* under, declared forwards. `.smt` reads
# as SMT-LIB too but is not what `put` writes, so the two directions are not
# each other's inverse and the forward map is the one to state.
# `smt` is a runnable script; `smt-src` is a predicate on its own, which is
# what a flag was given and not something a solver can be handed.
_SUFFIX = {"smt": ".smt2", "smt-src": ".smt", "md": ".md", "lean": ".lean",
           "json": ".json", "text": ".txt"}
_PREAMBLE = """\
# `artifacts/`

What this run left behind: the questions it asked, and the answers it found
that did not make it into the certificate.

Every `.smt2` file here is self-contained -- declarations, assertions,
`(check-sat)` -- so it can be run without `verith` in the loop:

```bash
cvc5 <file>        # or: z3 <file>
```

A `.smt` file is not a script: it is one predicate, exactly the text a flag
was given or a route found. `system.smt2` has the same predicates encoded,
where they can be run.

An **obligation** is written as its *negation*, which is how it is asked:
`unsat` means the obligation holds, `sat` means it is refuted and the model is
the counterexample. That is the same question `--pre-check cvc5` asks and the
same answer it reports.

`index.json` beside this file says the same thing in a form a program can
filter. What follows is written by the components of the run, in the order
they ran.
"""
_LANGUAGES = {suffix: lang for lang, suffix in _SUFFIX.items()}


def language_of(path: "Path | str") -> str:
    return _LANGUAGES.get(Path(path).suffix.lower(), "other")


def module_digest(module) -> str:
    """A short digest of the module an artifact is about.

    `str(module)` is the module's own rendering -- what `create_project`
    already writes to `dbg/system.txt` -- so this changes exactly when the
    module does, and an artifact from before an edit can be told apart from
    one after it.
    """
    return hashlib.sha256(str(module).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Artifact:
    """One file in `artifacts/`, and what is known about it."""

    name: str                       # file name within `artifacts/`
    role: str = "other"             # one of ROLES
    language: str = "other"         # one of _LANGUAGES' values
    status: str = "unknown"         # one of STATUSES
    what: str = ""                  # prose: what the file is
    why: str = ""                   # prose: what was wrong with it
    producer: str = ""              # the `--infer` route that wrote it
    seq: int = 0                    # monotonic; ordering across runs
    kind: str = ""                  # the proof rule it was found under
    prp: str = ""                   # the property it was found against
    module_digest: str = ""         # the module it is about
    derived_from: tuple[str, ...] = ()
    created: str = ""

    @property
    def indexed(self) -> bool:
        """Whether the index describes this file, or it was merely found."""
        return bool(self.created)


@dataclass
class ArtifactStore:
    """`artifacts/`, as the routes see it.

    Read by every route; written by every route.  Unlike `owns` on a route's
    row this is not per-route access: the point is that one run's leftovers
    are another run's starting point, and the run that wrote an artifact is
    usually not the one that uses it.
    """

    dir: Path
    module_digest: str = ""
    kind: str = ""
    prp: str = ""
    producer: str = ""              # the route writing, for `put`
    enabled: bool = True            # `--artifacts ignore` reads nothing
    log: "object" = None

    # --- reading --------------------------------------------------------

    def list(self) -> tuple[Artifact, ...]:
        """Every file present, newest first, described where the index does.

        A file with no entry is listed as `other`/`unknown` rather than
        hidden: whether a route understands an arbitrary file is the route's
        business, and the index is not the authority on what exists.
        """
        if not self.dir.is_dir():
            return ()
        indexed = {e["name"]: e for e in self._index()}
        found: list[Artifact] = []
        for path in sorted(self.dir.iterdir()):
            if not path.is_file() or path.name == _INDEX:
                continue
            entry = indexed.get(path.name)
            if entry is None:
                found.append(
                    Artifact(name=path.name, language=language_of(path))
                )
            else:
                found.append(Artifact(**{**entry, "derived_from": tuple(entry.get("derived_from", ()))}))
        return tuple(sorted(found, key=lambda a: (a.seq, a.name), reverse=True))

    def usable(
        self,
        role: str,
        *,
        languages: "tuple[str, ...] | None" = None,
        statuses: "tuple[str, ...] | None" = None,
    ) -> tuple[Artifact, ...]:
        """The artifacts of `role` this run may actually start from.

        Filters on the three staleness fields as well as on role: an
        artifact about a different module, a different property or a
        different proof rule is dropped, and what was dropped is logged
        rather than silently absent -- two identical command lines that
        resume differently is the one thing a workspace must not do quietly.
        """
        if not self.enabled:
            return ()
        keep: list[Artifact] = []
        for a in self.list():
            if a.role != role:
                continue
            if languages is not None and a.language not in languages:
                continue
            if statuses is not None and a.status not in statuses:
                continue
            stale = self._stale(a)
            if stale:
                self._say(f"   skipping {a.name}: {stale}")
                continue
            keep.append(a)
        return tuple(keep)

    def read(self, artifact: "Artifact | str") -> str:
        name = artifact if isinstance(artifact, str) else artifact.name
        return (self.dir / name).read_text()

    def _stale(self, a: Artifact) -> str:
        """Why this artifact is not about this run, or "" if it is."""
        if not a.indexed:
            return ""                       # nothing claimed, nothing to contradict
        if a.module_digest and self.module_digest and a.module_digest != self.module_digest:
            return "written for a different module"
        if a.prp and self.prp and a.prp != self.prp:
            return f"found against a different property ({a.prp})"
        if a.kind and self.kind and a.kind != self.kind:
            return f"found under a different proof rule ({a.kind})"
        return ""

    # --- writing --------------------------------------------------------

    def put(
        self,
        role: str,
        text: str,
        *,
        status: str = "unknown",
        what: str = "",
        why: str = "",
        language: str = "smt",
        derived_from: "tuple[Artifact | str, ...]" = (),
        stem: str = "",
        unique: bool = False,
    ) -> Artifact:
        """Record what this run found, whether or not it proved anything.

        `status` and `why` are not optional in spirit: an artifact whose
        status is `unknown` and whose `why` is empty is a file the next run
        can do nothing with but read.

        `what` is the other half, and it is for a person: a sentence saying
        what the file *is*, appended to `README.md` under the file's name.
        Nothing here knows what a run will produce -- which routes will run,
        what a later component will want to leave behind -- so the index is
        not one writer's table of contents. Each component says its own line
        as it writes, which makes the README exactly what was written, in the
        order it was written, including by code that did not exist when this
        module was.
        """
        if role not in ROLES:
            raise ValueError(f"unknown artifact role {role!r}; one of {ROLES}")
        if status not in STATUSES:
            raise ValueError(f"unknown artifact status {status!r}; one of {STATUSES}")
        self.dir.mkdir(parents=True, exist_ok=True)
        entries = self._index()
        seq = max((e.get("seq", 0) for e in entries), default=0) + 1
        suffix = _SUFFIX.get(language, ".txt")
        name = (f"{stem}{suffix}" if unique
                else f"{stem or role}-{seq:04d}-{self.producer or 'verith'}{suffix}")
        if unique:
            # A run records its predicates once before a route and once after,
            # so that an inferred one is written down too. When the route
            # changed nothing, saying so twice is noise and rewriting the file
            # is a lie about when it was written.
            here = self.dir / name
            if here.is_file() and here.read_text() == text:
                known = next((e for e in entries if e.get("name") == name), None)
                if known is not None:
                    return Artifact(**{**known,
                                       "derived_from": tuple(known.get("derived_from", ()))})
        # A `unique` artifact is the same file every run, so its entry replaces
        # the previous one rather than accumulating beside it.
        entries = [e for e in entries if e.get("name") != name]
        (self.dir / name).write_text(text)
        artifact = Artifact(
            name=name,
            role=role,
            language=language,
            status=status,
            what=what,
            why=why,
            producer=self.producer,
            seq=seq,
            kind=self.kind,
            prp=self.prp,
            module_digest=self.module_digest,
            derived_from=tuple(
                d if isinstance(d, str) else d.name for d in derived_from
            ),
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        entries.append(asdict(artifact))
        (self.dir / _INDEX).write_text(json.dumps(entries, indent=2) + "\n")
        self._describe(entries)
        self._say(f"   wrote artifact {name} ({status})")
        return artifact

    def _describe(self, entries: list[dict]) -> None:
        """Rewrite `README.md` from the index.

        Rendered rather than appended to. A component says its line once, by
        passing `what` to :meth:`put`, and this is where every line said so far
        becomes a file -- so the README is always exactly the index, and a
        `unique` artifact rewritten by a later run replaces its own section
        instead of gaining a second one.
        """
        out = [_PREAMBLE]
        for e in sorted(entries, key=lambda e: e.get("seq", 0)):
            said = (e.get("what") or "").strip() or \
                f"A `{e.get('role', 'other')}` artifact this run wrote."
            if (e.get("why") or "").strip():
                said += ("\n\nWhy it is here rather than in the certificate: "
                         + e["why"].strip())
            status = e.get("status", "unknown")
            # `proved` and `encoded` are not news: the first is a certificate
            # that worked and the second is a question, which the preamble
            # already explains. A status line is for what went wrong.
            out.append(f"\n## `{e['name']}`\n\n{said}\n"
                       + (f"\n*status: {status}*\n"
                          if status not in ("proved", "encoded") else ""))
        (self.dir / _README).write_text("".join(out))

    def note(self, text: str, **kw) -> Artifact:
        """A `.md` artifact: what a route wants the next run's prompt to know."""
        return self.put("note", text, language="md", **kw)

    def encoded(self, role: str, name: str, text: str, *, what: str,
                language: str = "smt") -> Artifact:
        """A file this run *encoded*, under a name it chose.

        `put` names an artifact after its role and a sequence number, which is
        right for a candidate an indefinite number of runs may produce and
        wrong for the one system or the one `step_inv` obligation: those are
        overwritten each run and are looked for by name.
        """
        return self.put(role, text, status="encoded", what=what,
                        language=language, stem=name, unique=True)

    # --- lifecycle ------------------------------------------------------

    def reset(self) -> int:
        """Drop everything (`--artifacts reset`); returns how many files went.

        A bad artifact is otherwise inherited by every subsequent run in the
        same `-o`, and nothing in a certificate would show it.
        """
        if not self.dir.is_dir():
            return 0
        gone = 0
        for path in self.dir.iterdir():
            if path.is_file():
                path.unlink()
                gone += 1
        return gone

    def _index(self) -> list[dict]:
        path = self.dir / _INDEX
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            # A corrupt index is not a reason to lose the files: they are
            # still listed, untyped, which is what an unindexed file is.
            self._say(f"   {path} is not readable JSON; artifacts listed untyped")
            return []
        return data if isinstance(data, list) else []

    def _say(self, message: str) -> None:
        if callable(self.log):
            self.log(message)
