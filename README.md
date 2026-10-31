# Reactor

Reactor is a library for formally modelling and analysing reactive systems, including
cyber-physical and multi-agent ones. It describes a system as a composition of *modules*:
components that interact with their environment in a never-ending loop, reading inputs,
updating their state and producing outputs. A controller, the physical plant it drives, a neural-network policy, or
one agent among many are each a module, and larger systems are built by wiring the outputs of
some modules to the inputs of others. This makes Reactor a common ground for systems that are
usually modelled with separate tools:

### Software, hardware and communication

- **communication protocols**, as parties exchanging messages over channels;
- **distributed algorithms**, such as consensus, leader election and mutual exclusion, as processes communicating through shared or message-passing variables;
- **embedded and real-time software**, as control loops and interrupt handlers updated tick by tick over integers and bit-vectors;
- **hardware (RTL) designs**, as registers and combinational logic over bit-vectors, updated clock by clock.

### Control and autonomy

- **control systems**, as controllers in closed loop with plants evolving in discrete or continuous time;
- **autonomous and robotic systems**, as perception, planning and actuation modules, with neural components in the loop;
- **reinforcement learning**, as environments and policies interacting in closed loop, for training and for verification;
- **automotive and avionics**, as ECUs and flight controllers coupled with vehicle dynamics;
- **medical devices**, such as pacemakers and insulin pumps, in closed loop with a physiological model.

### Infrastructure

- **power grids and energy systems**, as generators, storage and loads balancing supply and demand over time;
- **traffic and transportation**, as vehicles, intersections and signals coordinating flows;
- **industrial automation**, as PLC logic driving machines and conveyors on a production line;
- **smart buildings and IoT**, as sensors, thermostats and actuators coordinating comfort and energy use.

### Economic, social and natural systems

- **supply chains**, as interacting agents (suppliers, warehouses, retailers) moving goods and orders between them;
- **financial and market systems**, as traders, exchanges and clearing houses reacting to orders and prices;
- **epidemics and social systems**, as agent-based models of populations and their interactions;
- **biological systems**, such as gene regulatory networks and cell populations, as interacting dynamic components.

**DISCLAIMER: the project is under heavy development and unstable.
It lacks documentation in many places and can change at any moment.**

## Modelling language

The modelling language is centred on, and inspired by,
[reactive modules](https://www.cis.upenn.edu/~alur/FMSD99.pdf) (Alur & Henzinger).
As there, a module is made of *atoms*, each controlling a set of variables through an initial
action, an update action, and a delay activity for continuous-time evolution. Reactor adapts the
formalism where it serves modelling and analysis: atoms are defined as wiring diagrams over typed
terms; terms are parametrised by a theory (e.g., linear integer or real arithmetic and bit-vectors),
which fixes what can be expressed and how it can be analysed; and values can also be tensors,
so learned components such as neural networks live in the same model as the rest of the system.

## Rust core and Python interface

The core is written in Rust, with a Python interface that lets you:

- build modules directly, or by subclassing in a lightweight DSL (`reactor.sugar`);
- extract modules from existing code: [Gymnasium](https://gymnasium.farama.org/) environments
  (`reactor.gym`), [PyTorch](https://pytorch.org/) networks (`reactor.torch`) and SMV models;
- simulate them, train learned components in the loop, and verify properties with SMT solvers
  such as Z3 (`reactor.z3`);
- visualise them (`reactor.visual`).

See the [tutorials](tutorials/README.md) for worked examples, from wrapping an environment and a
neural network to proving termination of small programs with learned ranking functions.

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
just py-rebuild
```

Any later build of the `python` crate can be done using

```sh
just py-build
```

## Project structure

Crates are structured as follows.

```sh
 - base      # core data-structures
 - python    # Python API to access the crates
 - tutorials # Tutorials and use cases
 ```

For details on each crate see its own README.

## Running tests

After building the project, you can run tests of the Rust code using `just rs-test`.
Tests of the Python interface can be run by `just py-test`.
You can run all tests with the command `just test`.

### Running individual tests

To run concrete Rust tests, use `cargo test <pattern>` which will run all tests matching `<pattern>`.
You can use `cargo test -- --exact module::test` to run a single test with fully specified path.

To run a concrete Python test, go to the `python` crate and run `uv run pytest file.py::test-name`,
e.g., `uv run pytest tests/test_base.py::test_term_new`.
Alternatively, you can use `just py-test [file::test]` from anywhere in the project
to run all or any concrete test.

## Advanced building

### Building all target with all features

You can build all targets, including the Python interfaces, with all features using the command

```sh
just build
```

Note that for successfully running `build`, the environment for the python crate needs to be set up first
(e.g., by running `just py-rebuild` before).

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
maturin develop

# run a test
python tests/test.py
```

