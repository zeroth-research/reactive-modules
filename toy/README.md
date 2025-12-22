# Zeroth Reactive Modules -- Toy crate

This crate defines an instance of reactive modules used for prototyping and playing with reactive modules.

The [data types](src/dtype.rs) support `Int` (Rust's `i64`), `Real` (Rust's `f64`, subject to change),
and `Bool` (booleans). There is also a partial support for matrices of `Int` or `Real`.
[Instructions](src/instruction.rs) support basic arithmetic on the data types (basic arithmetics, logical operations on booleans, etc.).
There are two instruction for conditionals which are `Ite` (if-then-else) with the classical meaning and `Choose`
(together with `IfThen` which is basically the Rust's `Option`) which is a guarded non-deterministic choice.

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
cargo run --package toy --bin toy --stdout toy/tests/ex1.zrm 
```

A module can be also dumped into an interactive HTML page:

```sh
cargo run --package toy --bin toy --dump html --open toy/tests/ex1.zrm
```

If the crate is built with the feature `convertions-smt`, you can convert a toy module
into an [`smt`](../smt) module. To see the result, you can use this command:

```sh
cargo run --package toy --bin toy --to smt toy/tests/ex1.zrm --stdout
```
