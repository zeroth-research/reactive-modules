"""How much the printer's DAG sharing saves, as a net gets deeper.

Each layer of the dense net feeds the next, so printing the term as a tree
is exponential in depth and printing it as a DAG is linear.

    uv run python tests/limits/probes/sharing_scale.py
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zrth.lean.cert import CertificateData
from zrth.lean.smt_query import ModuleQueries, predicate_facts
from zrth.lean.smt_to_lean import smt_to_lean_nat


def relu(e: str) -> str:
    return f"(ite (>= {e} 0) {e} 0)"


def dense(layers: int) -> str:
    a = b = "s0"
    for _ in range(layers):
        a, b = relu(f"(+ {a} {b})"), relu(f"(+ {a} (- {b}))")
    return f"(+ {a} {b})"


if __name__ == "__main__":
    module = importlib.import_module("mods.m_countdown").module()
    print(
        f"{'layers':>6s} {'units':>6s} {'chars tree':>11s} {'chars DAG':>10s} "
        f"{'max tree':>9s} {'max DAG':>8s} {'n_branch':>9s}"
    )
    for k in (2, 4, 6, 8, 10):
        q = ModuleQueries.build(module, CertificateData(ranking=dense(k)))
        wires = q.msmt.ctrl_next
        tree = smt_to_lean_nat(q.ranking, wires, share=False)
        dag = smt_to_lean_nat(q.ranking, wires, share=True)
        facts = predicate_facts(q)
        print(
            f"{k:6d} {2 * k:6d} {len(tree):11d} {len(dag):10d} "
            f"{tree.count('(max '):9d} {dag.count('(max '):8d} {facts.n_branch:9d}"
        )
