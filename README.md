# Zeroth Reactive Modules

## Building

The easiest way is to use [`just`](https://just.systems/man/en/). Install this tool using Cargo
(or [some other way described in its README](https://github.com/casey/just?tab=readme-ov-file#installation)).

```sh
cargo install just
```

Then, simply run

```sh
just build
```

## Building the Python interface

You need the [`uv`](https://github.com/astral-sh/uv) Python package manager.
Once you have it installed, run:

```sh
just rebuild-python
```

Any later re-build of the `python` crate can be done using

```sh
just build-python
```

These commands build Python package that binds all available crates.
If you want to tweak the options, see the README in `python` crate
(or the `justfile`).

