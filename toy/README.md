# Zeroth Reactive Modules -- Toy crate

This crate defines an instance of reactive modules used for prototyping and playing with reactive modules.

The [data types](src/dtype.rs) support `Int` (Rust's `i64`), `Real` (Rust's `f64`, subject to change),
and `Bool` (booleans). [Instructions](src/instruction.rs) support basic arithmetic on `Int`s and `Real`s and logical operations on `Bool`.
There are two instruction for conditionals which are `Ite` (if-then-else) with the classical meaning and `Choose`
which is a guarded non-deterministic choice.

## Input

There is an experimental [parser](src/parser/parsing.rs) for specifications very similar to how modules are
defined in the original reactive module's paper (there are some additional curly braces,
otherwise it is the same). An example of such a specification is:

```
module Counter {
  external y0, z0 : Int
  interface w, x, y, z : Int

  atom {
    controls x, y, z reads x, y, z # awaits y0, z0 

    init {
      [] true -> x' := 0; y' := y0'; z' := z0'
    }

    update {
      []  (x < y \or x < z) -> x' := x + 1 
      [] !(x < y \or x < z) -> x' := 0
    }
  }
}
```

Modules can be parsed and shown by the `parse` binary, e.g.:

```sh
cargo run --package toy --bin parse --stdout toy/tests/ex1.zrm 
```

A module can be also dumped into an interactive HTML page:

```sh
cargo run --package toy --bin parse --dump html --open toy/tests/ex1.zrm
```
