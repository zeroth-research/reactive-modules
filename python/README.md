# Accessing and creating reactive modules from Python

## Setup

Install [uv](https://github.com/astral-sh/uv) and run `just rebuild-python`
(if this is the first time) or `just build-python` (any other time).

## Testing

To run Python tests after the Python interface has been built,
run `just test-python`.
To run a concrete Python test, go to the `python` crate and run `uv run pytest file.py::test-name`,
e.g., `uv run pytest tests/test_base.py::test_term_new`.
Alternatively, you can use `just pytest [file::test]` from anywhere in the project
to run all or any concrete test.

If you need to see the output of the test, use the `-s` option with pytest commands,
e.g., `just pytest tests/test_analyzer.py -s`.
In case a test fails and you need to attach the debugger, use `--pdb` flag with pytest.

## Building without `just` and `uv`

Here is a guide how to build the whole project from scratch including Python interface,
but without using `just` and `uv`.

### 1. Setup Python version

First, we need the right Python version. Currently, we require Python 3.10-3.13.
If you cannot or do not want to do this system-wide, you can use `pyenv`:

```sh
# install the right Python version
pyenv install 3.13

# start a new shell with the installed Python
pyenv shell 3.13 
# if the previous command failed, you can use `pyenv local 3.13` to use the
# installed Python in the current shell, # or run `pyenv init` and follow the instructions
```

### 2. Setup virtual environment and install requirements

Once you have the right version of Python installed, set up the virtual environment
and install `maturin`:

```sh
# setup the virtual environment
python3 -mvenv .venv
source .venv/bin/activate

# install maturin
pip install maturin==1.9
# or `pyenv exec pip install maturin==1.9` if using pyenv
```

Note that there are other requirements mentioned in `pyproject.toml`
and you can install them too, but it is not necessary -- `maturin`
will do this for us.

Alternatively, for setting up the virtual environment and installing requirements,
you can use Poetry:

```sh
# install poetry if necessary
pip install poetry

# create the virtual environemnt
$(poetry env activate)

# install requirements
poetry install
```

If you are using `pyenv` without setting the installed Python as the global interpreter,
you may need to prefix all commands with `pyenv exec`:

```sh
pyenv exec pip install poetry
$(pyenv exec poetry env activate)
pyenv exec poetry install
```

### 3. Build the project

When `maturin` is installed, you can finally build the project:

```sh
# activate the virtual environment if in a new terminal:
source .venv/bin/activate
# or `$(poetry env activate)` if using poetry
# or `$(pyenv exec poetry env activate)` for poetry + pyenv

# build the crate (pick features that you want)
maturin develop --features enable-torch,enable-smt

# run a test
python tests/test.py
```

