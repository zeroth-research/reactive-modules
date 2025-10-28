# Connection to PyTorch

## Setup

Before building this crate, it is necessary to install the required Python
packages. You might need to use  `pyenv` and `virtualenv` to do the setup
successfully, depending on your system.

```sh
# Install the right Python version *including shared libraries*.
# At the moment of writing, PyO3 supports Python 3.10--3.13.
env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.13

# Start a new shell with the installed Python
pyenv shell 3.13

# Create a Python virtual environment (you may use `pyenv`'s virtualenv
# plugin instead ).
python -m venv .venv
source .venv/bin/activate
# install the required Python packages
pip install poetry
poetry install
```

### Building with Maturin

First, make sure you have Maturin:
```sh
# (inside your virtualenv if you set it up)
pip install maturin
```

Building the crate is then:
```sh
export LIBTORCH_USE_PYTORCH=1
maturin develop
```

The command above will build the crate and install the Python package into the
virtual environment.

### Building manually
Then you can build the crate, using the pip-installed PyTorch.
```sh
LIBTORCH_USE_PYTORCH=1 cargo build
```

TBD: the need to rename `libzrm_torch.dylib` into `zrm_torch.so`, otherwise Python will not load it.


## Usage

### With Maturin

After running `maturin develop`, all the Python packages and libraries have been
installed into the virtual environment, so you can start using them without any
other setup.  If you run into an issue like this:

```py
ImportError: dlopen(reactive-modules/torch/zrm_torch.so, 0x0002): Library not loaded: @rpath/libtorch_python.dylib
```

It means that Python cannot find the shared libraries of PyTorch. That can be
fixed by running this command before using the package:

```sh
export DYLD_LIBRARY_PATH=<path-to-libtorch_python>:$DYLD_LIBRARY_PATH
```

(On Linux, use `LD_LIBRARY_PATH`.)


### Without Maturin

```sh
pyenv shell 3.13
source .venv/bin/activate
export PYTHONPATH=<path-to-zrm_torch.so>

python scripts/check.py
```

Also, you might need to export the `DYLD_LIBRARY_PATH` in case that Python
cannot find PyTorch libraries (see above).
