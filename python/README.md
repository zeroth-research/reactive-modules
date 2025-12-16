# Accessing and creating reactive modules from Python

## Setup

Install [uv](https://github.com/astral-sh/uv) and run `just rebuild-python`
(if this is the first time) or `just build-python` (any other time).

## Structure

This crate build a Python package called `zrth` that has the following structure:

```
zrth \            # top-level package
   _zrth \        # Rust-generated objects wrapping Rust objects, mostly defined in `src/lib.rs`
      smt         # defined in `src/smt/`
      torch       # defined in `src/torch/`
      toy         # defined in `src/toy/`
   smt            # Python code specific for `smt` crate
   torch          # Python code specific for `torch` crate
   toy            # Python code specific for `toy` crate

   __init__.py 
   context.py     # Files common for all crates
   ...
```

In short, `zrth` has two parts: a part generated from Rust which is accessible via `zrth._zrth`,
and a hand-written part which is the rest (`zrth`, `zrth.toy`, ...). Note that the Rust-generated
part is not physically present in the `zrth` package code. It will be filled-in during building
as a shared library.

## Defining a module

A module can be defined by inheriting from a class `zrth.{smt,torch,toy}.Module`
and defining the `update` (and optionally `init`) methods.

An `update` method has a fixed signature `update(self, control_variables, external_variables)`
and `init` has to be `init(self, external_variables)`. Objects `control_variables`
and `external_variables` are *tuples* of variables which are specified when creating an instance
of the module. The method `init` returns a tuple of values that define the initial value of control
variables and `update` computes the next value of the control variables (that implies that, e.g.,
if there are 4 control variables, the methods will each return 4 values).
Here is an example of a module:

```py
import zrth.toy as toy

class ToyModule(toy.Module):

    def init(self, extl) -> None:
        y0, z0 = extl
        return 0, y0, z0  # = x, y, z

    def update(self, ctrl, extl) -> None:
        x, y, z = ctrl
        xn = self.choose(
            ((x < y) or (x < z), x + 1),
            (~((x < y) or (x < z)), 0),
        )

        return xn, y, z

m = ToyModule(ctrl=("x: Int", "y: Int", "z: Int"), extl=("y0: Int", "z0: Int"))
m.to_html(open=True)
```

This is the general scheme, but the way to specify modules for torch and smt crates
may different slightly (because they are simply different crates with different operations).

For more examples, see the [`tests`](tests/) directory.


## Manual setup and build

```shell
uv sync
uv run maturin develop
uv run tests/test.py
```

