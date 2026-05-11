# Tutorials

Jupyter notebooks that walk through `zrth` step by step:

1. **counter.ipynb**: wrapping environments and neural networks, training a ranking function, formal verification with Z3
2. **pendulum.ipynb**: module composition with shared wires, training a controller, closed-loop verification
3. **mountaincar.ipynb**: wrapping an unmodified gymnasium environment (`MountainCarContinuous-v0`) and matching its behavior exactly

Tutorials 2 and 3 build on concepts from Tutorial 1, **do them in order**.

## Setup

Run

```
just build-tutorials
```

Or simply

```
just tutorials
```

which will do the necessary setup and run jupyter notebook automatically.

### Manual

From the project root:

```
uv sync --group tutorials
```

This installs `zrth` (built via maturin) plus the tutorial dependencies (`ipykernel`, `notebook`, `matplotlib`, `pygame`, `z3-solver`). Requires Python 3.12–3.13 and a Rust toolchain.

## Running

### VS Code

Install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter), open a notebook, and select the `.venv` kernel.

### uv

From the root directory, run:

```
uv run jupyter notebook tutorials/
```

### Jupyter directly

In the root directory, activate the venv first and then run Jupyter:

```
source .venv/bin/activate
jupyter notebook tutorials/
```
