# Zeroth Reactive Modules


## Building

For building, you need the standard Rust development environment and Cargo.
If you want to build the Python API too, you will need also
Python shared libraries (see below).

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
If you want to build all targets with all features, run:

```sh
just build-all
```

To tweak the build process, examine the `justfile` to see what is being done.


## Building the Python interface

For building the Python interface, Python shared libraries are required
(in case you do not have them and cannot install them, you can install them through Pyenv,
see the section [Building with Poetry](python/README.md#Building with Poetry) in
[python crate's README](python/README.md)).

The provided `just` configuration uses the [`uv`](https://github.com/astral-sh/uv) Python package manager,
but other package managers can be used too (e.g., Poetry, see below).
The following command builds the Python interface, setting up the virtual environment
and installing all required Python packages (except Python shared libraries):

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
