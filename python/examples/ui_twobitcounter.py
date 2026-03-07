"""Interactive TUI for the 2-bit counter — powered by InterpreterApp.

Run with:
    cd python && uv run python examples/ui_twobitcounter.py
"""
import torch
from zrth import Wire, Term, Module, DType as dt, IType as it
from tui import InterpreterApp


def make_twobitcounter():
    b0 = (Wire(dt.Bool([1])), Wire(dt.Bool([1])))
    b1 = (Wire(dt.Bool([1])), Wire(dt.Bool([1])))
    enable = (Wire(dt.Bool([1])), Wire(dt.Bool([1])))

    not_b0 = Wire(dt.Bool([1]))
    not_b1 = Wire(dt.Bool([1]))
    b0_and_enable = Wire(dt.Bool([1]))

    init = [
        Term(it.Tensor(torch.tensor([False])), [b0[1]]),
        Term(it.Tensor(torch.tensor([False])), [b1[1]]),
    ]
    update = [
        Term(it.Not(),  [not_b0],        [b0[0]]),
        Term(it.Ite(),  [b0[1]],         [enable[1], not_b0, b0[0]]),
        Term(it.And(),  [b0_and_enable], [b0[0], enable[1]]),
        Term(it.Not(),  [not_b1],        [b1[0]]),
        Term(it.Ite(),  [b1[1]],         [b0_and_enable, not_b1, b1[0]]),
    ]

    m = Module.sequential(init, update, obs=[b0, b1, enable])
    return m, b0, b1, enable


if __name__ == "__main__":
    m, b0, b1, enable = make_twobitcounter()

    InterpreterApp(
        module=m,
        observe={"b1": b1[0], "b0": b0[0]},
        inputs={"enable": (enable[1], torch.tensor([False]))},
        title="2-Bit Counter",
    ).run()
