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
CARGO := "cargo"

# Convert FEATURES into a flag only if set

profile_flag := if PROFILE == "" { "" } else { "--profile {{PROFILE}}" }
features_flag := if FEATURES == "" { "" } else { "--features {{FEATURES}}" }

# -------------------------------------------------
# Core Cargo Commands
# -------------------------------------------------

# Build the project in the default mode (use PROFILE and FEATURES variables to adjust)
build:
    {{ CARGO }} build {{ profile_flag }} {{ features_flag }}

# Build the project with all its features
build-all:
    {{ CARGO }} build --all-targets --all-features {{ profile_flag }}
    @just build-python

# Run tests
test:
    {{ CARGO }} test {{ features_flag }}

test-python:
    @just run-python pytest

test-all:
    {{ CARGO }} test --all-features {{ profile_flag }}
    @just test-python

# Run all or a concrete python test
pytest *args:
    cd python && uv run pytest {{ args }}

# Clean the current build
clean:
    {{ CARGO }} clean

# Run clippy on the workspace on all targets and with all features
clippy:
    {{ CARGO }} clippy --all-targets --all-features

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
