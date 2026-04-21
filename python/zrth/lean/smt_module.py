"""SMT wrapper around a reactive Module: symbolic input/output in cvc5.

Exposes `ModuleSMT` — given a `Module`, provides:
  * `fresh_inputs(prefix)` — mint cvc5 consts for the ctrl-latched and
    extl-latched/next input wires.
  * `init_state(extl_n)` / `update_state(ctrl, extl_l, extl_n)` — return
    the ctrl-next outputs as a list of cvc5 terms (one per component).
  * `init_pre(...)` / `update_pre(...)` — optional precondition formulas
    translated from Python IR term lists (defaults to `True`).

State is carried as a Python list of per-component cvc5 terms, matching
the left-nested tuple layout used elsewhere in the codegen.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvc5

from zrth import Wire, Module
from .smt_encode import translate_terms, wire_sort


@dataclass
class ModuleSMT:
    tm: cvc5.TermManager
    module: Module

    def __post_init__(self):
        atoms = list(self.module.atoms)
        if len(atoms) != 1:
            raise ValueError(
                f"ModuleSMT: expected single-atom module, got {len(atoms)}"
            )
        self.atom = atoms[0]
        # Snapshot init/update terms immediately — the Rust-backed
        # `Atom.init` / `Atom.update` iterators are one-shot in some
        # wrappers, so other consumers (e.g. `LeanContext`) could
        # exhaust them before we get here.
        self.init_terms: list = list(self.atom.init)
        self.update_terms: list = list(self.atom.update)
        self.ctrl_latched: list[Wire] = [p[0] for p in self.module.ctrl]
        self.ctrl_next: list[Wire] = [p[1] for p in self.module.ctrl]
        self.extl_latched: list[Wire] = [p[0] for p in self.module.extl]
        self.extl_next: list[Wire] = [p[1] for p in self.module.extl]

    # --- input construction ---------------------------------------------

    def _consts_for(self, wires: list[Wire], prefix: str) -> list[cvc5.Term]:
        return [
            self.tm.mkConst(wire_sort(self.tm, w), f"{prefix}{i}")
            for i, w in enumerate(wires)
        ]

    def fresh_ctrl(self, prefix: str = "s") -> list[cvc5.Term]:
        return self._consts_for(self.ctrl_latched, prefix)

    def fresh_extl_l(self, prefix: str = "el") -> list[cvc5.Term]:
        return self._consts_for(self.extl_latched, prefix)

    def fresh_extl_n(self, prefix: str = "en") -> list[cvc5.Term]:
        return self._consts_for(self.extl_next, prefix)

    # --- symbolic init/update -------------------------------------------

    def init_state(self, extl_n: list[cvc5.Term]) -> list[cvc5.Term]:
        """Return ctrl-next outputs after running `init(extl_n)`."""
        bindings = {w.id: v for w, v in zip(self.extl_next, extl_n)}
        wt = translate_terms(self.tm, self.init_terms, bindings)
        return [wt[w.id] for w in self.ctrl_next]

    def update_state(
        self,
        ctrl: list[cvc5.Term],
        extl_l: list[cvc5.Term],
        extl_n: list[cvc5.Term],
    ) -> list[cvc5.Term]:
        """Return ctrl-next outputs after running `update(ctrl, extl_l, extl_n)`."""
        bindings: dict[int, cvc5.Term] = {}
        for w, v in zip(self.ctrl_latched, ctrl):
            bindings[w.id] = v
        for w, v in zip(self.extl_latched, extl_l):
            bindings[w.id] = v
        for w, v in zip(self.extl_next, extl_n):
            bindings[w.id] = v
        wt = translate_terms(self.tm, self.update_terms, bindings)
        return [wt[w.id] for w in self.ctrl_next]

    # --- preconditions --------------------------------------------------

    def _pre_formula(
        self,
        terms: list | None,
        bindings: dict[int, cvc5.Term],
    ) -> cvc5.Term:
        """Translate a list of IR terms for a precondition (last wire is
        the `Mat Bool 1 1` result). Returns `true` if `terms` is None/empty."""
        if not terms:
            return self.tm.mkBoolean(True)
        wt = translate_terms(self.tm, list(terms), bindings)
        from .smt_encode import mat_select, wire_shape

        out_wire = terms[-1].write[0]
        out = wt[out_wire.id]
        return mat_select(self.tm, out, wire_shape(out_wire), 0, 0)

    def init_pre(
        self,
        terms: list | None,
        extl_l: list[cvc5.Term],
        extl_n: list[cvc5.Term],
    ) -> cvc5.Term:
        bindings = {w.id: v for w, v in zip(self.extl_latched, extl_l)}
        bindings.update({w.id: v for w, v in zip(self.extl_next, extl_n)})
        return self._pre_formula(terms, bindings)

    def update_pre(
        self,
        terms: list | None,
        extl_l: list[cvc5.Term],
        extl_n: list[cvc5.Term],
    ) -> cvc5.Term:
        bindings = {w.id: v for w, v in zip(self.extl_latched, extl_l)}
        bindings.update({w.id: v for w, v in zip(self.extl_next, extl_n)})
        return self._pre_formula(terms, bindings)
