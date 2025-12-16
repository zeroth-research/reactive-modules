# Reactive modules that work with Torch tensors

## Building together with Python bindings (the `python` crate)

When this crate is being build to be used with the `python` crate,
we must make sure both crate use the same version of `libtorch`.
The easiest way is to directly use the `libtorch` installed
by the `python` crate. If you run `just rebuild-python`,
the command takes care of this happening.

If you need to do this manually for some reason, this should work:

```sh
# setup the virtual environment and install requirements into it
cd ../python
uv sync   # could use pip too

# activate the environment
source .venv/bin/activate
cd ../torch

# build the crate
export LIBTORCH_USE_PYTORCH=1
cargo build
```
```

Exporting `LIBTORCH_USE_PYTORCH` may be redundant if you do not have multiple
`libtorch` in the system.
```

## Building standalone

Do the setup as required by the `tch` crate here:
<https://github.com/LaurentMazare/tch-rs?tab=readme-ov-file#getting-started>
Then, run

```sh
cargo build
```

