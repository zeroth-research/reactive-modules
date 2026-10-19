"""The names the generated Lean project is made of, in one place.

Codegen writes the project's files and modules under these names, and the
rest of the pipeline refers to them: the lake build target and its `sorry`
filter, the fbk bridge's imports, the files the fbk-proveit route returns
and owns. Each is defined here once, so a reference cannot drift from what
was written. A leaf: it imports nothing from the package, so `project`,
`translate/` and the route rows can all use it. The templates under
`templates/project/` spell the same names, and `tests/test_lean_names.py`
holds the two together.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    """Where one route puts an encoding and what Lean calls it.

    `base` is the name of the row with no suffix -- the functional encoding
    -- and every other row hangs off it. In the project the files live in
    their own `System/` directory and are named by suffix alone; a
    `--cert-file` writes them next to whatever else is in the output
    directory, so there the names carry `base` too (`flat`).
    """

    base: str
    prefix: str = ""            # Lean module prefix, e.g. "System."
    flat: bool = False          # file names carry `base`
    directory: Path = Path()    # only `path` uses it

    def stem(self, suffix: str) -> str:
        return f"{self.base}{suffix}" if self.flat else (suffix or self.base)

    def path(self, suffix: str) -> Path:
        return self.directory / f"{self.stem(suffix)}.lean"

    def lean_module(self, suffix: str) -> str:
        return f"{self.prefix}{self.stem(suffix)}"


# The project's own layout, also the one `generate_root` imports through.
SYSTEM_LAYOUT = Layout(base="System", prefix="System.")


# The certificate's library: the `Certificate/` directory and the lean_lib
# that globs it, so every file in it is built.
CERT_LIB = "Certificate"
DATA = "Data"                   # the predicates (`Data.lean.j2`)
PROOF = "Certificate"           # the proof (`Certificate.lean.j2`)
EQUIVALENCE = "Equivalence"     # fbk-proveit: the NA model is the module

# Where fbk-proveit installs `lean-ltl-certifying`'s side of the proof.
PROVEIT_DIR = "ProveIt"


def cert_file(stem: str) -> str:
    """The certificate library's file `stem`, relative to the project."""
    return f"{CERT_LIB}/{stem}.lean"


def cert_module(stem: str) -> str:
    """The Lean module of the certificate library's file `stem`."""
    return f"{CERT_LIB}.{stem}"
