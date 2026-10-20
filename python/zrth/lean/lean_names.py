"""The names the generated Lean project is made of, in one place.

Codegen writes the project's files and modules under these names, and the
rest of the pipeline refers to them: the lake build target and its `sorry`
filter, the fbk bridge's imports, the files the fbk-proveit route returns
and owns. Each is defined here once, so a reference cannot drift from what
was written. A leaf: it imports nothing from the package, so `project`,
`fbk/` and the route rows can all use it. The templates under
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


# The lemmas `simp_mat` reduces matrix arithmetic with, after the
# certificate's own definitions. The bridge's cascade uses the same set:
# without it a `Linear` arrives as `matVecAffine 2 [[1, 0], [0, 1]] [0, 1]
# (fun i j => match i, j with ...)` and no arithmetic prover can touch it.
MAT_SIMP: tuple[str, ...] = (
    "MatAdd_apply", "MatMul_apply", "MatZero_apply", "Pi.add_apply",
    "mul_Mat_apply", "add_Mat_apply", "Bool.or_eq_true", "decide_eq_true_eq",
    "Fin.sum_univ_succ", "Fin.sum_univ_zero", "Fin.isValue", "Fin.sum_univ_one",
    "Fin.sum_univ_two", "Fin.sum_univ_three", "matVecAffine", "dotL",
    "List.ofFn_succ", "List.ofFn_zero",
    # The three reductions unfold to a `List.ofFn` fold that the two lemmas
    # above then compute; without the defs here the fold never opens.
    "matMin", "matMax", "argmax_1d", "argmax",
)


# The linters every generated Lean file turns off, after its imports. Each
# proof is one tactic cascade for every goal shape, so on any one goal some
# of its steps do nothing and some of its simp lemmas go unused; and the
# encodings' binders are positional, used or not. Those five linters said
# so 1063 times over the test suite's certificates, which buried the one
# warning a build is read for: `declaration uses 'sorry'`.
LINTERS_OFF = "\n".join(
    f"set_option linter.{name} false"
    for name in (
        "unusedVariables", "unusedSimpArgs", "unusedTactic",
        "unreachableTactic", "unnecessarySeqFocus",
    )
)


def rm_def(extl_type: str, ctrl_type: str, noncomputable: bool = False) -> str:
    """`RM`, the module as a `ReactiveModule` over the functional
    encoding's `init`/`update` and the certificate's preconditions. The
    certificate states its theorems about it and the fbk bridge relates it
    to the NA model, each in its own namespace, so both print this one."""
    return "\n".join([
        f"{'noncomputable ' if noncomputable else ''}def RM : ReactiveModule "
        f"({extl_type}) ({ctrl_type}) := {{",
        "    init := fun e => init e.2",
        "    update := fun x e => update x e.1 e.2",
        "    init_pre := init_pre",
        "    update_pre := update_pre",
        "}",
    ])


def cert_file(stem: str) -> str:
    """The certificate library's file `stem`, relative to the project."""
    return f"{CERT_LIB}/{stem}.lean"


def cert_module(stem: str) -> str:
    """The Lean module of the certificate library's file `stem`."""
    return f"{CERT_LIB}.{stem}"
