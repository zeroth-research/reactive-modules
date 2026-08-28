# Use bash for better scripting features
# set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
#
# Recipes are named by target: `rs-` for Rust, `py-` for Python, `nb-` for
# notebooks; an unprefixed name means "all" (e.g. `test` = `rs-test` + `py-test`
# + `nb-test`), and a `-fix` suffix marks the recipes that modify files in the
# repository — everything else is read-only towards the sources.
#
# Prerequisites are declared as native dependencies (not `@just` calls in the
# bodies), so shared ones run at most once per invocation: `just test` builds
# the python crate once, although both `py-test` and `nb-test` require it.

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
NBSTRIPOUT := "uvx nbstripout@0.9.1 --max-size 16k --drop-empty-cells"

# Convert FEATURES into a flag only if set

profile_flag := if PROFILE == "" { "" } else { "--profile " + PROFILE }
features_flag := if FEATURES == "" { "" } else { "--features " + FEATURES }

# -------------------------------------------------
# All (unprefixed = Rust + Python + notebooks)
# -------------------------------------------------

# Build everything: the Rust workspace (all targets and features) and the Python crate
build: rs-build-all py-build

# Run the whole test suite (Rust + Python + notebooks)
test: rs-test py-test nb-test

# Clean the current build (the shared cargo target directory)
clean:
    {{ CARGO }} clean

# Full rebuild from scratch
rebuild: clean build

# -------------------------------------------------
# Rust (rs-)
# -------------------------------------------------

# Build the Rust workspace in the default mode (use PROFILE and FEATURES variables to adjust)
rs-build:
    {{ CARGO }} build {{ profile_flag }} {{ features_flag }}

# Build the Rust workspace with all its targets and features
rs-build-all:
    {{ CARGO }} build --all-targets --all-features {{ profile_flag }}

# Every Rust test binary transitively links libtorch (via `theory` -> `pyo3-tch` ->
# `torch-sys`), so they all need libtorch; and on macOS the Python interpreter's symbols at runtime.
# `uv run` does not set up the dynamic-linker path, so we do it here, per-OS;

# Run the Rust test suite (all features)
rs-test:
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

# Run clippy on the workspace on all targets and with all features (warnings are errors)
rs-clippy:
    {{ CARGO }} clippy --all-targets --all-features -- -D warnings

# `cargo fmt` needs no torch, so both fmt recipes skip the uv wrapper.

# Check formatting without modifying files (used by CI)
rs-fmt:
    cargo fmt --all -- --check

# Format the sources in place
rs-fmt-fix:
    cargo fmt --all

# -------------------------------------------------
# Python (py-)
# -------------------------------------------------

# Build the Python crate (maturin builds the workspace crates it depends on itself)
py-build:
    cd python && uv run maturin develop

# Build Python (from scratch) with all possible bindings
py-rebuild: clean rs-build-all py-build
    #DISABLED: build the `torch` crate and make sure it uses the libtorch that will
    #DISABLED:  be used also by the `python` crate
    #DISABLED: cd python && uv sync --no-build-package zrth --no-install-project
    #DISABLED: cd python && source .venv/bin/activate  && LIBTORCH_USE_PYTORCH=1 {{ CARGO }} build --package torch
    #DISABLED: cd python && uv run maturin develop  --features enable-torch
    @echo "Now you can go into the \`python\` directory and use \`uv run <script.py>\`"\
          "(or \`uv run python\` to get Python interpreter with \`zrth\` available)"

# Run all or a concrete python test (rebuilding the Python crate first)
py-test *args: (py-run "pytest" args)

# Run a command inside the `python` crate (with rebuilding the Python crate). The command given is executed from *within* the `python` crate, i.e., with paths relative to the root of the crate.
py-run *args: py-build
    cd python && uv run {{ args }}

# -------------------------------------------------
# Notebooks (nb-)
# -------------------------------------------------

# Install the tutorials dependency group
nb-sync:
    uv sync --group tutorials

# Build code and prepare for running tutorials
nb-build: build nb-sync

# Open the tutorial notebooks in Jupyter
nb-run: nb-build
    uv run jupyter notebook tutorials/

# mountaincar and pendulum are parked: they need non-linear ops (sin/cos/
# tanh/pow) and Stack, which no current theory provides — the analyzer
# raises on them by design.
# (ZRTH_NO_BROWSER keeps zrth.visual.show from opening browser pages during the run)

# Execute the tutorial notebooks against the current python crate without modifying them; fails on the first cell that errors
nb-test *notebooks="tutorials/counter.ipynb tutorials/gym_and_sugar.ipynb tutorials/decrement_1d/decrement_1d.ipynb tutorials/cairo/cairo.ipynb tutorials/singapore-2/Singapore-2.ipynb": py-build nb-sync
    ZRTH_NO_BROWSER=1 uv run jupyter nbconvert --to notebook --execute --stdout \
        --ExecutePreprocessor.timeout=600 {{ notebooks }} > /dev/null

# Fail when a committed notebook carries outputs the stripper would remove (used by CI)
nb-strip:
    {{ NBSTRIPOUT }} --verify $(git ls-files '*.ipynb')

# Strip bulky generated notebook outputs in place; text outputs under 16k (verification results, module prints) survive
nb-strip-fix:
    {{ NBSTRIPOUT }} $(git ls-files '*.ipynb')
