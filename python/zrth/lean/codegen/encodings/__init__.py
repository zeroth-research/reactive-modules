"""The module's Lean encodings: ``ModuleToLean4``, one submodule per encoding.

`functional`, `circuit`, `scalar` and `relational` (the matrix and the
scalar relational ones) are the encodings; `skeleton` is the shape every
relational encoding has, and `common` what the scalar ones share.
"""

from functools import cached_property

from zrth.lean.codegen.context import LeanContext
from zrth import Module

from zrth.lean.codegen.encodings.functional import atom_to_lean_functional, constants_block
from zrth.lean.codegen.encodings.circuit import (
    atom_to_lean_circuit as _atom_to_lean_circuit,
    to_lean_equiv_theorems as _to_lean_equiv_theorems,
)
from zrth.lean.codegen.encodings.scalar import (
    atom_to_lean_scalar as _atom_to_lean_scalar,
    to_lean_scalar_equiv as _to_lean_scalar_equiv,
)
from zrth.lean.codegen.encodings.relational import (
    atom_to_lean_rel as _atom_to_lean_rel,
    atom_to_lean_mat_rel as _atom_to_lean_mat_rel,
)


class ModuleToLean4:
    """Convert a Python Module into Lean4 wiring diagram code.

    Read-only consumer of a pre-populated LeanContext — does not mutate it.
    Accepts a Module for convenience; in that case builds a fresh context.

    Every method stands alone: what an equivalence theorem needs of its
    encoding (the circuit's layer names, the scalar encoding's argmax
    variants) is computed once, by whichever method asks first.
    """

    def __init__(self, src: "LeanContext | Module"):
        self.ctx = src if isinstance(src, LeanContext) else LeanContext(src)

    @cached_property
    def _circuit(self) -> "tuple[str, list[str], list[str]]":
        """The circuit encoding, and its init and update layer names."""
        return _atom_to_lean_circuit(self.ctx)

    @cached_property
    def _scalar(self) -> "tuple[str, list[tuple[str, int]]]":
        """The scalar encoding, and the argmax variants it defines."""
        return _atom_to_lean_scalar(self.ctx)

    @property
    def module(self):
        return self.ctx.module

    # ------------------------------------------------------------------
    # Functional encoding
    # ------------------------------------------------------------------

    def to_lean_functional(self) -> str:
        """Generate the functional init/update definitions (plus constants)."""
        return "{}\n\n{}".format(
            constants_block(self.ctx), atom_to_lean_functional(self.ctx)
        )

    # ------------------------------------------------------------------
    # Circuit encoding
    # ------------------------------------------------------------------

    def to_lean_equiv_theorems(self) -> str:
        """Generate theorems proving circuit ≡ functional."""
        _, init_names, update_names = self._circuit
        return _to_lean_equiv_theorems(self.ctx, init_names, update_names)

    def to_lean_circ(self) -> str:
        """The module as a combinational circuit, and circuit ≡ functional."""
        return "{}\n\n{}".format(
            self._circuit[0],
            self.to_lean_equiv_theorems(),
        )

    # ------------------------------------------------------------------
    # Scalar encoding
    # ------------------------------------------------------------------

    def to_lean_scalar_equiv(self) -> str:
        """Generate theorems proving init/update = pack ∘ Scalar.init/update ∘ unpack."""
        return _to_lean_scalar_equiv(self.ctx, self._scalar[1])

    def to_lean_scalar(self) -> str:
        """The scalar encoding inside ``namespace Scalar``, and its
        composition theorems."""
        return "{}\n\n{}".format(
            self._scalar[0],
            self.to_lean_scalar_equiv(),
        )

    # ------------------------------------------------------------------
    # Relational encoding
    # ------------------------------------------------------------------

    def to_lean_rel(self) -> str:
        """Generate the relational encoding inside ``namespace ScalarRel``."""
        return _atom_to_lean_rel(self.ctx)

    def to_lean_mat_rel(self) -> str:
        """Generate the matrix-domain relational encoding inside ``namespace Rel``."""
        return _atom_to_lean_mat_rel(self.ctx)


    # ------------------------------------------------------------------
    # Combined output
    # ------------------------------------------------------------------

    def to_lean(
        self, circuit: bool = True, scalar: bool = True, rel: bool = True
    ) -> str:
        """Return all enabled encodings joined by blank lines.

        ``circuit=True`` appends the circuit (Box algebra) encoding and its
        equivalence theorems.  ``scalar=True`` appends the scalar namespace
        (``Scalar.init``/``update``, ``unpack_*``, ``pack``) and the
        composition theorems.  ``rel=True`` appends the relational namespace
        (``ScalarRel.effect_i``, ``ScalarRel.R_i``, ``ScalarRel.TransRel``, ``ScalarRel.InitCond``).
        All default to True; pass ``False`` to omit.
        """
        parts = [self.to_lean_functional()]
        if circuit:
            parts.append(self.to_lean_circ())
        if scalar:
            parts.append(self.to_lean_scalar())
        if rel:
            parts.append(self.to_lean_rel())
        return "\n\n".join(parts)
