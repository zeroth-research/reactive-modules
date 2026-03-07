"""Step through the 2-bit counter using the Python interpreter.

Run with:
    cd python && uv run python examples/run_twobitcounter.py
"""
import torch
from zrth import Wire, Term, Module, DType as dt, IType as it
from interpreter import Interpreter


def make_twobitcounter():
    b0 = (Wire(dt.Bool([1])), Wire(dt.Bool([1])))  # (latched, next)
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
        Term(it.Ite(),  [b0[1]],         [enable[1], not_b0, b0[0]]),   # b0 XOR enable
        Term(it.And(),  [b0_and_enable], [b0[0], enable[1]]),
        Term(it.Not(),  [not_b1],        [b1[0]]),
        Term(it.Ite(),  [b1[1]],         [b0_and_enable, not_b1, b1[0]]),  # b1 XOR (b0 AND enable)
    ]

    m = Module.sequential(init, update, obs=[b0, b1, enable])
    return m, b0, b1, enable


def show(interp, b0, b1, label=""):
    b0_val = int(interp.get(b0[0].id()).item())
    b1_val = int(interp.get(b1[0].id()).item())
    tag = f"  <- {label}" if label else ""
    print(f"  b1={b1_val}  b0={b0_val}  ({b1_val * 2 + b0_val}){tag}")


m, b0, b1, enable = make_twobitcounter()
interp = Interpreter(m)

EN  = {enable[1].id(): torch.tensor([True])}
HLD = {enable[1].id(): torch.tensor([False])}

print("=== init ===")
interp.initialize()
show(interp, b0, b1, "reset")

print("\n=== count up (enable=1 each step) ===")
for _ in range(4):
    interp.step(EN)
    show(interp, b0, b1)

print("\n=== hold (enable=0) ===")
interp.step(EN)
show(interp, b0, b1, "count to 1")
for _ in range(3):
    interp.step(HLD)
    show(interp, b0, b1, "hold")

print("\n=== mixed ===")
interp.initialize()
show(interp, b0, b1, "reset")
for label, inputs in [
    ("enable", EN), ("hold",   HLD), ("enable", EN),
    ("hold",   HLD), ("enable", EN), ("enable", EN),
]:
    interp.step(inputs)
    show(interp, b0, b1, label)
