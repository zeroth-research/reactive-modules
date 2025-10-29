# nusmv (crate)

A small NuSMV-subset parser and converter used by the reactive-modules
workspace.

This crate provides a pragmatic parser for a restricted subset of the
NuSMV language (see `nusmv.pest`) and converts parsed modules into the
`Module<DType, IType>` representation used in the rest of the repo.

## Purpose

- Provide a testable, compact importer for simple NuSMV models.
- Support variable declarations (boolean/integer) and `ASSIGN` with `init`/
  `next` statements.

## Usage

Call `nusmv::parse_nusmv(input)` with the source text of a NuSMV model. The
function returns a `base::module::Module<DType, IType>` representing the
parsed module.

Example (see tests in `src/nusmv.rs`):

```rust
let module = nusmv::parse_nusmv(r#"
    MODULE main
    VAR
      x : boolean;
    ASSIGN
      init(x) := TRUE;
      next(x) := !x;
"#);
```

## Grammar

The grammar is defined in `nusmv.pest`. The supported constructs include:

- Module declarations (`MODULE ...`)
- `VAR` section with `ident : boolean|integer;`
- `ASSIGN` section with `init(ident) := expr;` and `next(ident) := expr;`
- Boolean expressions with `!`, `&`, `|`, `TRUE`, `FALSE`, identifiers and
  integer literals.

## Design notes

- Each declared variable maps to two sets of wires: latched (current) and
  next. Wires are arranged so that latched indices are in `[0..n)` and next
  indices in `[n..2n)` for `n` variables.
- The crate builds `IType` expression trees and wraps them into `Term`
  instances (from `base`) which pair instruction payloads with their
  write/read wires. `Atom`s are then constructed from init/update `Term`s.

## Error handling

- Parsing failures are surfaced via panics in this crate (used in tests). In
  particular numeric literal parsing panics with a clear message on invalid
  input.

## Limitations

- This is not a full NuSMV implementation — only a small subset.
- `Term` and `Atom` internals live in the `base` crate; this crate does not
  attempt to change that API.

## Tests

See `src/nusmv.rs` for small unit tests demonstrating parsing for boolean
and integer variables.

