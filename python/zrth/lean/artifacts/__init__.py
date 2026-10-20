"""The generated project's `artifacts/`: what a run leaves for the next.

`store` is the directory and its index (`ArtifactStore`, `Artifact`, and
the `ROLES` and `STATUSES` an artifact is filed under); `record` is the two
writes a route makes to it, which do nothing when the store is `None` (a
`--cert-file` run, which has no project). `--artifacts ignore` still
writes: it is about where a run starts, so it reads nothing back.
"""

from .record import found, note
from .store import (
    ARTIFACTS_DIR,
    ROLES,
    STATUSES,
    Artifact,
    ArtifactStore,
    language_of,
    module_digest,
)

__all__ = [
    "ARTIFACTS_DIR", "ROLES", "STATUSES", "Artifact", "ArtifactStore",
    "found", "language_of", "module_digest", "note",
]
