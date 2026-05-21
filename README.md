# Zeroth Reactive Modules

Representation and manipulation of *reactive modules* in Rust and Python.

[Reactive modules]((https://link.springer.com/article/10.1023/A:1008739929481)) are a model for defining reactive systems,
i.e., "machines" that in loop compute outputs for incoming inputs.
This project contains a generic Rust representation of reactive modules with computation steps (atoms)
defined using wiring diagrams, several concrete instances of reactive modules (e.g., reactive modules
that use torch tensors as values for computations) and bindings in Python for the Rust code.

Examples and tutorials: TBD

## Building

For building, you need the standard Rust development environment and Cargo.
If you want to build the Python API too, you will need also
Python in version 3.10-3.13 installed (see below).

The easiest way to build the project is to use [`just`](https://just.systems/man/en/),
although you can build the project also manually (see later).
You can `just` using Cargo or Homebrew (or any other way
[described in its README](https://github.com/casey/just?tab=readme-ov-file#installation)).

```sh
brew install just # or `cargo install just`
```


### Building without Python interface 

To build the project without a Python interface, simply run

```sh
just build
```

or

```sh
cargo build
```

The project will be built with its default features.
For an overview of features and tweaking the build, see the
section [Advanced building](#advanced-building) below.

## Building with Python interface

The provided justfile uses the [`uv`](https://github.com/astral-sh/uv) Python package manager,
but other package managers can be used too (see the
section [Advanced building](#advanced-building) below).
The following command builds the project including the Python interface,
sets up the virtual environment and installs all required Python packages:

```sh
just rebuild-python
```

Any later build of the `python` crate can be done using

```sh
just build-python
```


## Project structure

Crates are structured as follows.

```sh
 - base      # core data-structures
 - python    # Python API to access the crates
 - smt       # an instance of reactive modules suitable for translating to SMT expressions
 - torch     # an instance of reactive modules where operations work with torch tensors
 - toy       # an instance of reactive modules for prototyping
 - visual    # crate confining code for visualizing reactive modules
```

 For details on each crate see its own README.

## Running tests

After building the project, you can run tests of the Rust code using `just test`.
Tests of the Python interface can be run by `just test-python`.
You can run all test with the command `just test-all`.

### Running individual tests

To run concrete Rust tests, use `cargo test <pattern>` which will run all tests matching `<pattern>`.
You can use `cargo test -- --exact module::test` to run a single test with fully specified path.

To run a concrete Python test, go to the `python` crate and run `uv run pytest file.py::test-name`,
e.g., `uv run pytest tests/test_basic_modules.py::test_counter_torch`.
Alternatively, you can use `just pytest [file::test]` from anywhere in the project
to run all or any concrete test.

### Slow tests (Lean proof verification)

Some Python tests verify Lean 4 certificates by running `lake build`, which
requires a working `lake` installation and a local Mathlib cache. These tests
are marked `@pytest.mark.slow` and are **skipped by default**.

```sh
just pytest                  # fast: skips all lake-build tests
just pytest -m slow          # slow: runs lake-build tests (requires lake + Mathlib)
just pytest -m "not slow"    # explicit fast-only run
```

Tests that carry the `slow` mark:

| File | Test | What it verifies |
|------|------|-----------------|
| `tests/lean/test_lean_hammer.py` | `test_zeroth_hammer_proofs` | `ZerothHammerTests.lean` compiles — individual tactic phase examples |
| `tests/lean/test_lean_hammer.py` | `test_manual_tests_build` | `ManualTests/Basic.lean` compiles — `zeroth_hammer` on standalone goal shapes |
| `tests/lean/test_lean_hammer.py` | `test_cert_countdown_build` | Generated `Certs/Countdown.lean` compiles without sorry |
| `tests/lean/test_lean_hammer.py` | `test_cert_twovars_build` | Generated `Certs/TwoVars.lean` compiles without sorry |
| `tests/lean/test_lean_hammer.py` | `test_cert_collatz_build` | Generated `Certs/Collatz.lean` compiles without sorry |
| `tests/test_lean_svcomp.py` | `test_countdown_proofs` | Countdown certificate compiles (separate lake project) |
| `tests/test_lean_svcomp.py` | `test_twovars_proofs` | Two-variable certificate compiles (separate lake project) |
| `tests/test_lean_svcomp.py` | `test_collatz_bounded_proofs` | Collatz-bounded certificate compiles (separate lake project) |

The first `lake build` invocation downloads Mathlib (~1 GB) and can take
10–30 minutes. Subsequent runs use the on-disk cache and typically finish in
under a minute per test.

## Advanced building

### Building all target with all features

You can build all targets, including the Python interfaces, with all features using the command

```sh
just build-all
```

Note that for successfully running `build-all`, the environment for the python crate needs to be set up first
(e.g., by running `just rebuild-python` before).

You can also select only some of the features when building the project. Available features are:

| Feature              | Default | Description                                        |
|----------------------|---------|----------------------------------------------------|
| `visual/html`          | `ON`  | Support for dumping modules to HTML                    |
| `toy/visual-html`      | `ON`  | Support for dumping toy modules to HTML                |
| `toy/conversions-smt`  | `OFF` | Enable converting toy modules to smt modules           |
| `python/visual-html`   | `ON`  | Support for dumping modules defined in Python to HTML  |
| `python/enable-torch`  | `OFF` | Enable Python bindings for torch modules               |
| `python/enable-smt`    | `OFF` | Enable Python bindings for smt modules                 |

For building the project with only some features, edit the justfile and rebuild the project
or run `just FEATURES=<list of features> build`.


### Building without `just` and `uv`

Here is a short guide how to build the whole project from scratch including Python interface,
but without using `just` and `uv` (when using those, simply see and edit [`justfile`](justfile)
and [`python/pyproject.toml`](python/pyproject.toml).
For more detailed guide, see [`python/README.md`](python/README.md).

#### 1. Setup Python version

First, we need the right Python version. Currently, we require Python 3.10-3.13.
If you cannot or do not want to do this system-wide, you can use `pyenv`:

```sh
# install the right Python version
pyenv install 3.13

# start a new shell with the installed Python
pyenv shell 3.13 
```

#### 2. Setup virtual environment and install requirements

Once you have the right version of Python installed, set up the virtual environment
and install `maturin`:

```sh
# setup the virtual environment
python3 -mvenv .venv
source .venv/bin/activate

# install maturin
pip install maturin==1.9
```

#### 3. Building the project

When `maturin` is installed, you can finally build the project:

```sh
# activate the virtual environment if in a new terminal:
source .venv/bin/activate

# enter the `python` crate (unless you were working from there already)
cd python 

# build the crate (pick features that you want)
maturin develop --features enable-torch,enable-smt

# run a test
python tests/test.py
```

