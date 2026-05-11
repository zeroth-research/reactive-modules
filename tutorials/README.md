# Tutorials

Jupyter notebooks walking through `zrth` step by step:

1. **counter.ipynb** — wrapping environments and neural networks, training a ranking function, formal verification with Z3
2. **pendulum.ipynb** — module composition with shared wires, training a controller, closed-loop verification
3. **mountaincar.ipynb** — wrapping an unmodified Gymnasium environment (`MountainCarContinuous-v0`) and matching its behavior exactly

Each tutorial builds on the previous one — work through them in order.
This README assumes the project was built using `uv`. For building instructions, see the top-level README.

## Quick start

### Open tutorials in web browser

Set up and launch Jupyter, openning tutorials in the browser:

```
just tutorials
```

### Open tutorials in VS Code

Setup the tutorials:

```
just build-tutorials
```

Then install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) VS Code extension
(if not done yet), open a notebook, and select the `.venv` kernel
from the root directory.

## Manual setup and running

Install dependencies from the project root:

```
uv sync --group tutorials
```

This installs `zrth` and all tutorial dependencies (`ipykernel`, `notebook`, `matplotlib`, `pygame`, `z3-solver`).

Then launch Jupyter via uv:

```
uv run jupyter notebook tutorials/
```

Or activate the venv first and run directly:

```
source .venv/bin/activate
jupyter notebook tutorials/
```
