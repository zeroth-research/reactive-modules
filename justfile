# Use bash for better scripting features
# set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Default recipe (runs when `just` is invoked with no args)
default:
    @just --list

# -------------------------------------------------
# Variables
# -------------------------------------------------
# Allow overriding from the CLI: `just PROFILE=release build`

PROFILE := ""
FEATURES := ""
# Run cargo through `uv` so the build/test env uses the project's Python
# and PyTorch — `theory` transitively links libtorch (via pyo3-tch/torch-sys),
# and `.cargo/config.toml` sets LIBTORCH_USE_PYTORCH=1, so cargo must see the
# venv's torch. If neccessary, override with `just CARGO=cargo ...` to use
# a bare toolchain.
CARGO := "uv run cargo"

# Convert FEATURES into a flag only if set

profile_flag := if PROFILE == "" { "" } else { "--profile {{PROFILE}}" }
features_flag := if FEATURES == "" { "" } else { "--features {{FEATURES}}" }

# Build the project in the default mode (use PROFILE and FEATURES variables to adjust)
build:
    {{ CARGO }} build {{ profile_flag }} {{ features_flag }}

# Build the project with all its features
build-all:
    {{ CARGO }} build --all-targets --all-features {{ profile_flag }}
    @just build-python

# Build code and prepare for running tutorials
build-tutorials:
    @just build-all
    uv sync --group tutorials

# Run tests
test:
    {{ CARGO }} test {{ features_flag }}

# Run python tests
test-python:
    @just run-python pytest

# Run the whole test suite (Rust + Python)
test-all:
    @just test-rust
    @just test-python

# Run the Rust test suite (all features)
# #
# Every Rust test binary transitively links libtorch (via `theory` -> `pyo3-tch` ->
# `torch-sys`), so they all need libtorch; and on macOS the Python interpreter's symbols at runtime.
# `uv run` does not set up the dynamic-linker path, so we do it here, per-OS;
test-rust:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB="$(uv run python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
    if [ "$(uname)" = "Darwin" ]; then
        export DYLD_FALLBACK_LIBRARY_PATH="$TORCH_LIB"
        export DYLD_INSERT_LIBRARIES="$(uv run python -c 'import sysconfig, os; base = sysconfig.get_config_var("PYTHONFRAMEWORKPREFIX") or sysconfig.get_config_var("LIBDIR"); print(os.path.join(base, sysconfig.get_config_var("LDLIBRARY")))')"
    else
        export LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    # Split by feature set: the `torch` feature is exclusive to `theory`, the rest of
    # the workspace is exercised with `theory/pyo3`.
    {{ CARGO }} test --features theory/pyo3 {{ profile_flag }}
    {{ CARGO }} test -p theory --features torch {{ profile_flag }}

# Run all or a concrete python test
pytest *args:
    cd python && uv run pytest {{ args }}

# Clean the current build
clean:
    {{ CARGO }} clean

# Run clippy on the workspace on all targets and with all features (warnings are errors)
clippy:
    {{ CARGO }} clippy --all-targets --all-features -- -D warnings

# Format the Rust sources in place. `cargo fmt` needs no torch, so skip the uv wrapper.
fmt:
    cargo fmt --all

# Check formatting without modifying files (used by CI)
fmt-check:
    cargo fmt --all -- --check

# Full rebuild from scratch
rebuild:
    @just clean
    @just build

# Build Python (from scratch) with all possible bindings
rebuild-python:
    # clean the build
    @just clean
    #DISABLED: build the `torch` crate and make sure it uses the libtorch that will
    #DISABLED:  be used also by the `python` crate
    #DISABLED: cd python && uv sync --no-build-package zrth --no-install-project
    #DISABLED: cd python && source .venv/bin/activate  && LIBTORCH_USE_PYTORCH=1 {{ CARGO }} build --package torch
    #DISABLED:  now build the python crate
    @just build-all
    #DISABLED: cd python && uv run maturin develop  --features enable-torch
    @echo "Now you can go into the \`python\` directory and use \`uv run <script.py>\`"\
          "(or \`uv run python\` to get Python interpreter with \`zrth\` available)"

# Build Python with all possible bindings (assuming `rebuild-python` was run previously)
build-python:
    # re-build the workspace
    @just build
    # now build the python crate
    cd python && uv run maturin develop

# Run a command inside the `python` crate (with rebuilding the Python crate). The command given is executed from *within* the `python` crate, i.e., with paths relative to the root of the crate.
run-python *args:
    @just build-python
    cd python && uv run {{ args }}

tutorials:
    @just build-tutorials
    uv run jupyter notebook tutorials/
