# Dialect of reactive modules that uses Torch tensors

## Finding Torch

## Building together with Python bindings (the `python` crate)

When this crate is being build to be used with the `python` crate,
we must make sure both crate use the same version of `libtorch`.
The easiest way is to directly use the `libtorch` installed
by the `python` crate. This we can do with the following steps:

```sh
cd ../python
uv sync
source .venv/bin/activate
cd ../torch
```

After this setup, run:

```sh
export LIBTORCH_USE_PYTORCH=1
cargo build
```
```

Exporting `LIBTORCH_USE_PYTORCH` may be redundant if you do not have multiple
`libtorch` in the system.
```

## General

Do the setup as required by the `tch` crate here:
<https://github.com/LaurentMazare/tch-rs?tab=readme-ov-file#getting-started>

