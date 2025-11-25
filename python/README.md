# Connecting toy crate to Python

## Setup

### Setup and build using uv

Install [uv](https://github.com/astral-sh/uv) and run

```shell
uv sync
uv run cargo build
```


### Setup with Poetry

Before building this crate, it is necessary to install the required Python
packages. You might need to use  `pyenv` and `virtualenv` to do the setup
successfully, depending on your system. Also make sure to have `poetry` installed.

```sh
# Install the right Python version *including shared libraries*.
# At the moment of writing, PyO3 supports Python 3.10--3.13.
env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.13

# Start a new shell with the installed Python
pyenv shell 3.13

# create a virtual environment with all dependencies
poetry install
```

#### Building with Maturin

Start the virtual environment:

```sh
$(poetry env activate)
```

Building the crate is then:

```sh
maturin develop
```

The command above will build the crate and install the Python package into the
virtual environment.

### Building manually


```sh
cargo build
```

TBD: the need to rename `libzrm_toy.dylib` into `zrm_toy.so`, otherwise Python will not load it.

## Usage

### With uv

```shell
uv run tests/test.py
```

