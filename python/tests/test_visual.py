from zrth import Int, LIA, Var, X
from zrth.sugar import Module
from zrth.visual.server import _serialize_module


class _Counter(Module):
    def init(self):
        return 0

    def update(self, x):
        return x + 1


def test_serialize_module_links_vars_to_term_wires():
    v = Var(Int([1, 1]))
    out = _serialize_module(_Counter(ctrl=(v,), theory=LIA))

    [atom] = out["atoms"]
    ctrl_ids = {c["id"] for c in atom["ctrl"]}
    assert ctrl_ids <= {w for t in atom["init"] for w in t["writes"]}
    assert ctrl_ids <= {w for t in atom["update"] for w in t["writes"]}
    assert {c["id"] for c in atom["read"]} <= {r for t in atom["update"] for r in t["reads"]}
    assert out["extl"] == []
    assert out["intf"] == [{"ltc": v.id, "nxt": X(v).id, "dtype": str(v.dtype)}]
