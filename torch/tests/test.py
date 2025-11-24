import sys
from os.path import dirname, join as pathjoin

sys.path.append(pathjoin(dirname(__file__), "../python"))

from bindings.term import Var
import bindings

import torch as tch


Ctx = bindings.Context


def fun(x, y=1):
    # straightline programs can be written without using our API,
    # all should work automatically
    t = tch.Tensor([1, 1, 1])
    s = (2 * x * t + y).sum()

    # however, we cannot do branching (if-then-else) when tracing,
    # so for conditions and branches, we need to use our API
    b = zrm.eq(s, 0)
    return zrm.ifelse(b, x, x + 1)


ctx = Ctx()
terms, inputs, outputs = ctx.trace(fun)

print("Inputs: ")
for n, val in enumerate(inputs):
    print(f" - {n}: {val}")
print("Terms:")
for n, term in enumerate(terms):
    print(f" - {n}: ", end=" ")
    sys.stdout.flush()
    term.print()
print("Outputs: ")
for n, val in enumerate(outputs):
    print(f" - {n}: {val}")

print("--- Building atom ---\n")

inputs = [inp.wrapped_term() for inp in inputs if isinstance(inp, Var)]
outputs = [out.wrapped_term() for out in outputs if isinstance(out, Var)]
atom = bindings.libzrm_torch.WrappedAtom(
    ctx.context_, inputs, outputs, [], [t.term_ for t in terms]
)

atom.dbg()
