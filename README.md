# Zeroth Reactive Modules


## Building

The easiest way to build the project is to use [`just`](https://just.systems/man/en/).
Install this tool using Cargo or Homebrew (or any other way
[described in its README](https://github.com/casey/just?tab=readme-ov-file#installation)).

```sh
brew install just # or `cargo install just`
```

Then run

```sh
just build
```

This will build the workspace in its default setup.
If you want to build all targets with all features (except Python API), run:

```sh
just build-all
```

To tweak the build process, examine the `justfile` to see what is being done.


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

## Project structure

Crates are structured as follows.

```sh
 - base      # core data-structures
 - python    # Python API to access the crates
 - smt       # An instance of reactive modules suitable for translating to SMT expressions
 - torch     # An instance of reactive modules where operations work with torch tensors
 - toy       # An instance of reactive modules for prototyping
 - visual    # Crate confining code for visualizing reactive modules
```

 For details on each crate see its own README.
