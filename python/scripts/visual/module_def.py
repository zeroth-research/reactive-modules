"""Dual-EMA crossover — editable reactive module definition.

Edit this file, save it, and the browser at http://localhost:7777 updates
automatically (when launched via scripts/visual/ui_watch.py).

The algorithm
─────────────
Given a scalar price stream, this module computes two exponential moving
averages with different smoothing constants:

    ema_fast[t] = α_f · price[t] + (1 − α_f) · ema_fast[t−1]   (α_f = 0.2)
    ema_slow[t] = α_s · price[t] + (1 − α_s) · ema_slow[t−1]   (α_s = 0.05)

The crossover signal and a boolean buy indicator are derived from these:

    signal[t] = ema_fast[t] − ema_slow[t]
    buy[t]    = signal[t] > 0

Try editing the smoothing constants below and saving to see the graph update.
"""

import torch
import zrth.visual
from zrth import Module, Wire, DType, IType, Term

# ── Smoothing constants (try changing these) ──────────────────────────────────
ALPHA_FAST = 0.20   # fast EMA  — reacts quickly to price changes
ALPHA_SLOW = 0.05   # slow EMA  — tracks the long-run trend

# ── Observable wire pairs  [latched, next] ───────────────────────────────────
#    latched = value from the previous tick (read by the update)
#    next    = value computed for this tick  (written by the update)
price    = [Wire(DType.Float([1])), Wire(DType.Float([1]))]   # external input
ema_fast = [Wire(DType.Float([1])), Wire(DType.Float([1]))]   # fast EMA state
ema_slow = [Wire(DType.Float([1])), Wire(DType.Float([1]))]   # slow EMA state
signal   = [Wire(DType.Float([1])), Wire(DType.Float([1]))]   # crossover value
buy      = [Wire(DType.Bool([1])),  Wire(DType.Bool([1]))]    # buy flag

# ── Intermediate (temporary) computation wires ────────────────────────────────
w_af   = Wire(DType.Float([1]))   # constant: ALPHA_FAST
w_as   = Wire(DType.Float([1]))   # constant: ALPHA_SLOW
w_1af  = Wire(DType.Float([1]))   # constant: 1 − ALPHA_FAST
w_1as  = Wire(DType.Float([1]))   # constant: 1 − ALPHA_SLOW
w_zero = Wire(DType.Float([1]))   # constant: 0.0
w_zero = Wire(DType.Float([1]))   # constant: 0.0



w_axf  = Wire(DType.Float([1]))   # ALPHA_FAST · price
w_axs  = Wire(DType.Float([1]))   # ALPHA_SLOW · price
w_omyf = Wire(DType.Float([1]))   # (1 − ALPHA_FAST) · ema_fast_prev
w_omys = Wire(DType.Float([1]))   # (1 − ALPHA_SLOW) · ema_slow_prev

# ── Initialisation: all states start at zero ─────────────────────────────────
init_terms = [
    Term(IType.Tensor(torch.tensor([0.0])),   [ema_fast[1]]),
    Term(IType.Tensor(torch.tensor([0.0])),   [ema_slow[1]]),
    Term(IType.Tensor(torch.tensor([0.0])),   [signal[1]]),
    Term(IType.Tensor(torch.tensor([False])), [buy[1]]),
]

# ── Update: EMA recurrence + crossover signal ─────────────────────────────────
update_terms = [
    # smoothing constants
    Term(IType.Tensor(torch.tensor([ALPHA_FAST])),        [w_af]),
    Term(IType.Tensor(torch.tensor([1.0 - ALPHA_FAST])), [w_1af]),
    Term(IType.Tensor(torch.tensor([ALPHA_SLOW])),        [w_as]),
    Term(IType.Tensor(torch.tensor([1.0 - ALPHA_SLOW])), [w_1as]),
    Term(IType.Tensor(torch.tensor([0.0])),               [w_zero]),

    # new-input contribution: alpha · price
    Term(IType.Mul(), [w_axf],        [w_af,        price[0]]),
    Term(IType.Mul(), [w_axs],        [w_as,        price[0]]),

    # decay term: (1 − alpha) · previous EMA
    Term(IType.Mul(), [w_omyf],       [w_1af,       ema_fast[0]]),
    Term(IType.Mul(), [w_omys],       [w_1as,       ema_slow[0]]),

    # new EMA values
    Term(IType.Add(), [ema_fast[1]],  [w_axf,       w_omyf]),
    Term(IType.Add(), [ema_slow[1]],  [w_axs,       w_omys]),

    # crossover signal and buy indicator
    Term(IType.Sub(), [signal[1]],    [ema_fast[1], ema_slow[1]]),
    Term(IType.Gt(),  [buy[1]],       [signal[1],   w_zero]),
]

# ── Build and push ─────────────────────────────────────────────────────────────
module = Module(
    init=init_terms,
    update=update_terms,
    obs=[price, ema_fast, ema_slow, signal, buy],
)

zrth.visual.push(module)
