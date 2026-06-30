"""Lean4 translation package.

Public API: ``ModuleToLean4`` — identical to the old ``translate.py``.
"""

from zrth.lean.common import LeanContext, dtype_shape
from zrth import Module

from zrth.lean.translate._shared import _scalar_bindings_with_recon, _prepend_recon
from zrth.lean.translate.functional import atom_to_lean_functional, _constants_block
from zrth.lean.translate.circ import (
    atom_to_lean_circuit as _atom_to_lean_circuit,
    to_lean_equiv_theorems as _to_lean_equiv_theorems,
)
from zrth.lean.translate.scalar import (
    atom_to_lean_scalar as _atom_to_lean_scalar,
    to_lean_scalar_equiv as _to_lean_scalar_equiv,
)
from zrth.lean.translate.rel import atom_to_lean_rel as _atom_to_lean_rel
from zrth.lean.translate.fbk import atom_to_lean_bool_rel as _atom_to_lean_bool_rel
from zrth.lean.translate.mat_rel import atom_to_lean_mat_rel as _atom_to_lean_mat_rel


class ModuleToLean4:
    """Convert a Python Module into Lean4 wiring diagram code.

    Read-only consumer of a pre-populated LeanContext — does not mutate it.
    Accepts a Module for convenience; in that case builds a fresh context.
    """

    def __init__(self, src: "LeanContext | Module"):
        self.ctx = src if isinstance(src, LeanContext) else LeanContext(src)
        # Layer names become available after atom_to_lean_circuit runs; the
        # equivalence-theorem emitter reads them back.
        self._init_layer_names: list[str] = []
        self._update_layer_names: list[str] = []
        self._argmax_scalar_variants: list[tuple[str, int]] = []

    @property
    def module(self):
        return self.ctx.module

    def _can_scalarize(self) -> bool:
        """True when every wire has a shape that can be flattened to scalars (≤ 2-D)."""
        ctx = self.ctx
        all_wires = ctx.ctrl_latched + ctx.ctrl_next + ctx.extl_latched + ctx.extl_next
        return all(len(dtype_shape(w.dtype)) <= 2 for w in all_wires)

    # ------------------------------------------------------------------
    # Functional encoding
    # ------------------------------------------------------------------

    def atom_to_lean_functional(self) -> str:
        return atom_to_lean_functional(self.ctx)

    def to_lean_functional(self) -> str:
        """Generate the functional init/update definitions (plus constants)."""
        return "{}\n\n{}".format(
            _constants_block(self.ctx), self.atom_to_lean_functional()
        )

    # ------------------------------------------------------------------
    # Circuit encoding
    # ------------------------------------------------------------------

    def atom_to_lean_circuit(self) -> str:
        """Generate the full Lean4 source for this module as a combinational circuit."""
        src, init_names, update_names = _atom_to_lean_circuit(self.ctx)
        self._init_layer_names = init_names
        self._update_layer_names = update_names
        return src

    def to_lean_equiv_theorems(self) -> str:
        """Generate theorems proving circuit ≡ functional."""
        return _to_lean_equiv_theorems(
            self.ctx,
            self._init_layer_names,
            self._update_layer_names,
        )

    def to_lean_circ(self) -> str:
        return "{}\n\n{}".format(
            self.atom_to_lean_circuit(),
            self.to_lean_equiv_theorems(),
        )

    # ------------------------------------------------------------------
    # Scalar encoding
    # ------------------------------------------------------------------

    def atom_to_lean_scalar(self) -> str:
        """Generate the scalar encoding inside ``namespace Scalar``."""
        src, argmax_variants = _atom_to_lean_scalar(self.ctx)
        self._argmax_scalar_variants = argmax_variants
        return src

    def to_lean_scalar_equiv(self) -> str:
        """Generate theorems proving init/update = pack ∘ Scalar.init/update ∘ unpack."""
        return _to_lean_scalar_equiv(self.ctx, self._argmax_scalar_variants)

    def to_lean_scalar(self) -> str:
        if not self._can_scalarize():
            return (
                "-- scalar encoding not available: module has non-scalar (matrix) wires"
            )
        return "{}\n\n{}".format(
            self.atom_to_lean_scalar(),
            self.to_lean_scalar_equiv(),
        )

    # ------------------------------------------------------------------
    # Relational encoding
    # ------------------------------------------------------------------

    def atom_to_lean_rel(self) -> str:
        """Generate the relational encoding inside ``namespace ScalarRel``."""
        return _atom_to_lean_rel(self.ctx)

    def atom_to_lean_bool_rel(self) -> str:
        """Generate a Bool-valued relational encoding inside ``namespace FBK``."""
        return _atom_to_lean_bool_rel(self.ctx)

    def atom_to_lean_mat_rel(self) -> str:
        """Generate the matrix-domain relational encoding inside ``namespace Rel``."""
        return _atom_to_lean_mat_rel(self.ctx)

    def to_lean_mat_rel(self) -> str:
        return self.atom_to_lean_mat_rel()

    def to_lean_rel(self) -> str:
        if not self._can_scalarize():
            return "-- relational encoding not available: module has non-scalar (matrix) wires"
        return self.atom_to_lean_rel()

    def to_lean_bool_rel(self) -> str:
        if not self._can_scalarize():
            return "-- bool relational encoding not available: module has non-scalar (matrix) wires"
        return self.atom_to_lean_bool_rel()

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
        All default to True; pass ``False`` to omit.  The scalar and
        relational encodings are silently skipped when the module has
        non-scalar (matrix) wires.
        """
        parts = [self.to_lean_functional()]
        if circuit:
            parts.append(self.to_lean_circ())
        if scalar:
            parts.append(self.to_lean_scalar())
        if rel:
            parts.append(self.to_lean_rel())
        return "\n\n".join(parts)
