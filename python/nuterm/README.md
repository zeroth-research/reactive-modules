# nuterm

Neural termination proofs for reactive modules.

A small network is trained as a ranking function on sampled runs of a program.
Its weights are rounded to integers and composed into the module. A
Farkas-certificate CEGAR procedure then checks that the rank decreases on every
round. The result can be emitted as a Lean proof and kernel-checked.
The procedure is sound but incomplete.

## Contents

| File | Role |
|---|---|
| `_bench.py` | `Bench`: a program as a module with named state and inputs. |
| `_train.py` | `learn_ranking`: sample runs, train, quantize, certify. |
| `_termination.py` | Composes the rank into the module. Defines the `terminates` claim. |
| `_farkas.py` | The decision procedure: `read_system`, `certify`. |
| `_property.py` | Claims: `Liveness` and `Safety`. |
| `_nodes.py` | Solver-space view of a composed module. |
| `_invariants.py` | Houdini-style invariant inference to shrink the domain. |
| `_lean.py` | Emits a Lean proof from the certificates. |
| `_lean_check.py` | Compiles the proof with `lake`. |
| `lean/` | The Lean library the proofs build on. |

## Run on a file

```sh
python -m nuterm FILE.py                # train and verify
python -m nuterm FILE.py --lean         # also kernel-check the proof
python -m nuterm FILE.py --export out/  # write the proof as its own Lean project
```

`FILE.py` defines `module()`, which returns a `zrth.Module`. Or it defines
`BENCH`, a `Bench`. For a module, state is named `s0..` and inputs `e0..`.
See `tests/test_nuterm_cli.py` for an example. The exit code is 0 only if
every file is verified.

`--lean` and `--export` need `lake` on `PATH`. Proofs are written to
`nuterm/lean/proofs/<name>/`.

## Run on the benchmarks

The SV-COMP corpus is in `nuterm/tests/benchmarks/svcomp/`. Run from `python/`:

```sh
python -m nuterm.tests.benchmarks.svcomp                  # train and verify all
python -m nuterm.tests.benchmarks.svcomp ndecr            # names containing 'ndecr'
python -m nuterm.tests.benchmarks.svcomp --lean           # also kernel-check proofs
python -m nuterm.tests.benchmarks.svcomp --lean --jobs 4  # four in flight
python -m nuterm.tests.benchmarks.svcomp --export out/    # each proof as its own project
```

Check that the encodings match the C sources:

```sh
python -m nuterm.tests.benchmarks.svcomp._equiv
```

## Use from Python

Build a `Bench`. Copy `nuterm/tests/benchmarks/svcomp/dsl/_template.py` for the
conventions.

```python
from nuterm._train import learn_ranking

result = learn_ranking(BENCH)
print(result.verified, result.reason)
```

To certify a ranking function you already have:

```python
from nuterm._termination import compose, system_of, terminates
from nuterm._farkas import certify

system = system_of(BENCH)
composed, witness = compose(system, layers)
proof = certify(composed, terminates(), witness)
```

## Tests

Running tests in this directory (without using `just`):

```sh
python -m pytest nuterm/tests -q
```
