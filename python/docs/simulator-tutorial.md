# Building a Simulator for Reactive Modules

This tutorial walks through writing a simulator that executes a reactive
module step by step. The complete, runnable code lives in
[`docs/examples/test_simulator.py`](examples/test_simulator.py) and is
tested by CI.

## Building a module

We'll use a simple counter as our running example. It has one state
variable `x` that starts at 0 and increments by 1 each step.

Every state variable is a **wire pair** `(latched, next)`:

- `latched` — the value from the *previous* step (what you read)
- `next` — the value being *computed* in the current step (what you write)

```python
import torch
from zrth import Wire, Term, Module, DType as dt, IType as it

# A wire pair for our state variable x
x = (Wire(dt.Int()), Wire(dt.Int()))
#     ^ latched          ^ next
```

The **init block** sets the initial value of `x` by writing to `x[1]`
(the next wire):

```python
init = [Term(it.Tensor(torch.tensor([0], dtype=torch.int64)), [x[1]])]
```

The **update block** computes the next value from the current one. Terms
are listed in dependency order — here we first create the constant `1`,
then add it to the latched value `x[0]`:

```python
one = Wire(dt.Int())
update = [
    Term(it.Tensor(torch.tensor([1], dtype=torch.int64)), [one]),
    Term(it.Add(), [x[1]], [x[0], one]),   # next_x = x + 1
]
```

A `Term(instruction, write_wires, read_wires)` is a single computation:
it reads input wires, applies the instruction, and writes the result.

Finally, wrap everything into a `Module`:

```python
m = Module.sequential(init, update, [x])
```

The third argument lists the observable wire pairs. When all state is
observable (no external inputs), `m.closed()` returns `True` — the
module can run on its own.


## The simulation loop

Simulating a module is three steps, repeated:

1. **Execute** a block (init or update) — evaluate each term in order
2. **Latch** — copy next-wire values into latched-wire slots
3. Repeat step 1–2 with the update block

The state is just a dictionary mapping wires to tensors:

```python
state = {}
```

### Executing a block

A module contains **atoms** (groups of terms). We iterate every term
across all atoms:

```python
for term in (t for atom in module.atoms for t in atom.init):
    inputs = [state[w] for w in term.read]
    outputs = evaluate(term.itype, inputs)
    state.update(zip(term.write, outputs))
```

Each term reads its input wires from the state dict, evaluates the
instruction, and writes the results back. The `evaluate` function is
described below.

### Latching

After executing a block, copy every next-wire value into its
corresponding latched-wire slot. The module exposes these pairs through
`module.ctrl`:

```python
state = {ltc: state[nxt] for (ltc, nxt) in module.ctrl}
```

For the update step (where the state dict already has latched values
from the previous step), latch in-place:

```python
for ltc, nxt in module.ctrl:
    if nxt in state:
        state[ltc] = state[nxt].clone()
```

### Putting it together

```python
# Initialize
state = {}
for term in (t for atom in m.atoms for t in atom.init):
    inputs = [state[w] for w in term.read]
    outputs = evaluate(term.itype, inputs)
    state.update(zip(term.write, outputs))

state = {ltc: state[nxt] for (ltc, nxt) in m.ctrl}

# Step
for term in (t for atom in m.atoms for t in atom.update):
    inputs = [state[w] for w in term.read]
    outputs = evaluate(term.itype, inputs)
    state.update(zip(term.write, outputs))

for ltc, nxt in m.ctrl:
    if nxt in state:
        state[ltc] = state[nxt].clone()

assert int(state[x[0]].item()) == 1  # x was 0, now 0+1=1
```

That's it — about 10 lines of logic.


## Evaluating instructions

The `evaluate` function dispatches on the instruction type. Each `IType`
variant is a distinct Python class, so you can use `type(itype)` as a
dictionary key:

```python
def evaluate(itype, inputs):
    dispatch = {
        type(it.Tensor(torch.zeros(1))): lambda: [itype._0.clone()],      # constant tensor
        type(it.Id()):       lambda: [inputs[0].clone()],      # identity / copy
        type(it.Add()):      lambda: [inputs[0] + inputs[1]],
        type(it.Sub()):      lambda: [inputs[0] - inputs[1]],
        type(it.Mul()):      lambda: [inputs[0] * inputs[1]],
        type(it.Not()):      lambda: [inputs[0].logical_not()],
        type(it.And()):      lambda: [inputs[0].logical_and(inputs[1])],
        type(it.Or()):       lambda: [inputs[0].logical_or(inputs[1])],
        type(it.Ite()):      lambda: [torch.where(inputs[0], inputs[1], inputs[2])],
        type(it.Eq()):       lambda: [inputs[0].eq(inputs[1])],
        type(it.Lt()):       lambda: [inputs[0].lt(inputs[1])],
        type(it.ReLU()):     lambda: [inputs[0].relu()],
    }

    fn = dispatch.get(type(itype))
    if fn is None:
        raise RuntimeError(f"unsupported instruction: {type(itype).__name__}")
    return fn()
```

Each entry returns a **list** of output tensors (matching the term's
write wires). Most instructions produce one output; `TensorSet` produces
one reshaped tensor.

To support more operations, add entries to the dispatch table. The full
set of `IType` variants is listed in the `zrth` API reference. Some
common ones:

| Instruction | Inputs | Output |
|---|---|---|
| `Tensor(t)` | — | constant tensor `t` |
| `Id()` | `[x]` | `x` (copy) |
| `Add()` | `[a, b]` | `a + b` |
| `Sub()` | `[a, b]` | `a - b` |
| `Mul()` | `[a, b]` | `a * b` |
| `Not()` | `[x]` | logical NOT |
| `And()` | `[a, b]` | logical AND |
| `Or()` | `[a, b]` | logical OR |
| `Ite()` | `[cond, t, f]` | `t` if cond else `f` |
| `Eq()` | `[a, b]` | `a == b` |
| `Lt()` | `[a, b]` | `a < b` |
| `ReLU()` | `[x]` | `max(0, x)` |
| `Linear()` | `[x, W, b]` | `x @ W + b` |
