"""What a route leaves in `artifacts/`: a predicate it found, or a note.

Both take the store a route was handed, which is `None` under `--artifacts
ignore`, and then write nothing.
"""

from __future__ import annotations


def found(store, role: str, src: str, *, what: str) -> None:
    """`src`, the `role` this run found and proved, as SMT-LIB.

    `proved` is the status that makes it resumable: `infer_route.FIXABLE`
    is what a later `--infer ai-cegis` takes as given.
    """
    if store is None:
        return
    store.put(role, src, status="proved", language="smt", what=what)


def note(store, text: str, *, status: str, what: str, why: str) -> None:
    """A note for the next run's prompt: what this one ruled out, or why it
    stopped."""
    if store is None:
        return
    store.note(text, status=status, what=what, why=why)
