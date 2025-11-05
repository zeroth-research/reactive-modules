# Connection to PyTorch

## Setup

We use [uv](https://github.com/astral-sh/uv) package manager to handle installing dependencies
and building this crate. Other building options are mentioned later
in this README.

When you have `uv` installed, the setup is as easy as

```shell
uv sync
uv run maturin develop
```

(The setup is to be made from the `torch` directory, not from the workspace
directory.)

## Building the package

TBD


## Usage

After setting up, Python packages and libraries have been
installed into the virtual environment, so you can start using them without any
other setup.

```sh
uv run tests/test.py
```

Using `uv run` ensures that the scripts are executed in the set up virtual environment.
If you run into an issue like this:

```py
ImportError: dlopen(reactive - modules / torch / zrm_torch.so, 0x0002): Library
not loaded:

@rpath / libtorch_python.dylib
```

It means that Python cannot find the shared libraries of PyTorch
(the `rpath` variable does not propagate into non-direct dependencies).
That can be fixed by running this command before using the package:

```sh
export DYLD_LIBRARY_PATH=<path-to-libtorch_python>:$DYLD_LIBRARY_PATH
```

(On Linux, use `LD_LIBRARY_PATH`.)


## Other setup options

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
export LIBTORCH_USE_PYTORCH=1
maturin develop
```

The command above will build the crate and install the Python package into the
virtual environment.

### Building manually

You can build the crate manually using the pip-installed PyTorch.

```sh
# setup venv and install torch
pyenv shell 3.13
source .venv/bin/activate

LIBTORCH_USE_PYTORCH=1 cargo build
```

After building the crate manually, you may need to rename the build `libzrm_torch.dylib` into `zrm_torch.so`,
otherwise Python may not load it. Use the crate then as follows:


```sh
# (skip if already done)
pyenv shell 3.13
source .venv/bin/activate

# tell python where to look for the zrm_torch.so library
export PYTHONPATH=<path-to-zrm_torch.so>

python scripts/check.py
```

Also, you might need to export the `DYLD_LIBRARY_PATH` (`LD_LIBRARY_PATH`) in case that Python
cannot find PyTorch libraries (see above).
