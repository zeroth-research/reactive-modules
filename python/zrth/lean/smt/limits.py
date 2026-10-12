"""Time limits on cvc5: the run's budget, and a solver held to it.

`tlimit` alone stops neither `checkSynth` nor a quantified `checkSat`;
`tlimit-per` stops both. Every solver built here gets the pair.

cvc5 is imported only when a solver is built: `cli` imports this module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

DEFAULT_CALL_MS = 5000
DEFAULT_PHASE_MS = 20000


@dataclass
class SmtBudget:
    """A per-call and a per-phase wall-clock bound on solving.

    `phase_ms` is spent down by every query in the phase, so a phase that
    fans out over many small queries (one per branch condition, say) cannot
    add up to more than one obligation check's worth of time. Call
    `start_phase` between phases to refill it.
    """

    per_call_ms: int = DEFAULT_CALL_MS
    phase_ms: int = DEFAULT_PHASE_MS
    _phase_start: float = field(default_factory=time.perf_counter, repr=False)

    def start_phase(self) -> None:
        self._phase_start = time.perf_counter()

    @property
    def phase_left_ms(self) -> int:
        spent = (time.perf_counter() - self._phase_start) * 1000
        return max(0, int(self.phase_ms - spent))

    @property
    def exhausted(self) -> bool:
        return self.phase_left_ms <= 0

    def next_call_ms(self) -> int:
        """The limit for the next query: whichever bound bites first."""
        return min(self.per_call_ms, self.phase_left_ms)


def call_ms(budget: "SmtBudget | None") -> int:
    """The limit for a search's next query: at least a millisecond, and
    `DEFAULT_CALL_MS` without a budget."""
    if budget is None:
        return DEFAULT_CALL_MS
    return max(1, budget.next_call_ms())


def bounded_solver(tm, ms: int, *, logic: "str | None" = None, **options):
    """A fresh solver on `tm`, each call held to `ms` milliseconds.

    Set in this order: `logic` when given, both limits, then `options`
    (an underscore in a name is a dash). A caller that sets the logic
    itself does so after.
    """
    import cvc5

    solver = cvc5.Solver(tm)
    if logic is not None:
        solver.setLogic(logic)
    solver.setOption("tlimit-per", str(ms))
    solver.setOption("tlimit", str(ms))
    for name, value in options.items():
        solver.setOption(name.replace("_", "-"), value)
    return solver
