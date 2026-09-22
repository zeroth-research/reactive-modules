import json

import pytest

from zrth import Int, LIA, LRA, Real, Var
from zrth.sugar import Module, X, d, ite
from zrth.visual import to_html
from zrth.visual import cli
from zrth.visual.server import _render, _serialize_module


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



class _Clock(Module):
    """An open hybrid clock -- init, update and delay all populated."""

    def init(self, t):
        return 0

    def update(self, x, t):
        return ite(X(t) == 1, 0, x)

    def delay(self, x, t):
        return 1 * d(t)


def _clock():
    return _Clock(theory=LRA, ctrl=(Var(Real([1, 1])),), extl=(Var(Real([1, 1])),))


def test_serialize_module_carries_the_flow_block():
    module = _clock()
    [atom] = module.atoms

    out = _serialize_module(module)["atoms"][0]
    # `flow` is the atom's delay block, and it is kept distinct from the others
    assert [t["label"] for t in out["flow"]] == [str(t.itype) for t in atom.delay]
    assert out["flow"]
    ids = [t["id"] for phase in ("init", "update", "flow") for t in out[phase]]
    assert len(ids) == len(set(ids))
    assert all(t["id"].startswith("a0_flow_") for t in out["flow"])


def test_term_labels_name_the_operation():
    # Not the default object repr -- see test_base.test_itype_str_is_the_theory_rendering.
    out = _serialize_module(_clock())["atoms"][0]
    labels = [t["label"] for p in ("init", "update", "flow") for t in out[p]]
    assert labels and not any("builtins" in x or "object at" in x for x in labels)
    assert {"Id", "Ite", "Eq"} <= set(labels)


_MODULE_FILE = '''
from zrth import Int, LIA, Var
from zrth.sugar import Module


class Counter(Module):
    def init(self):
        return 0

    def update(self, x):
        return x + 1


def module_fun():
    return Counter(ctrl=(Var(Int([1, 1])),), theory=LIA)


def not_a_module():
    return 42


def needs_args(n):
    return module_fun()
'''


@pytest.fixture
def module_py(tmp_path):
    path = tmp_path / "mymodule.py"
    path.write_text(_MODULE_FILE)
    return path


def test_to_html_inlines_a_snapshot():
    v = Var(Int([1, 1]))
    html = to_html(_Counter(ctrl=(v,), theory=LIA), names={v: "n"})

    # Standalone: every placeholder filled, the snapshot inlined, no live feed.
    assert "{{" not in html
    assert "const SNAPSHOT=null;" not in html
    body = html.split("const SNAPSHOT=", 1)[1].split(";\n", 1)[0]
    assert json.loads(body)["wire_names"][str(v.id)] == "n"


def test_to_html_escapes_a_script_close():
    # A label carrying `</script>` must not end the inline block early.
    assert "</script>" not in _render(snapshot={"label": "</script><b>"})[:-20]


def test_show_template_still_asks_for_the_live_feed():
    live = _render(ws_port=4321)
    assert "const SNAPSHOT=null;" in live and "ws://127.0.0.1:4321" in live


def test_cli_writes_the_page(module_py, tmp_path):
    out = tmp_path / "module.html"
    assert cli.main([f"{module_py}:module_fun", "--output", str(out)]) == 0
    assert "const SNAPSHOT={" in out.read_text()


def test_cli_open_flag_opens_what_it_wrote(module_py, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", opened.append)
    out = tmp_path / "module.html"
    cli.main([f"{module_py}:module_fun", "--output", str(out)])
    assert opened == []
    cli.main([f"{module_py}:module_fun", "--output", str(out), "--open"])
    assert opened == [out.resolve().as_uri()]


@pytest.mark.parametrize("spec, msg", [
    ("{f}:not_a_module", "not a zrth Module"),
    ("{f}:needs_args", "required argument"),
    ("{f}:absent", "defines no 'absent'"),
    ("{f}", "expected a FILE:FUNCTION spec"),
    ("no_such_file.py:module_fun", "no such file"),
])
def test_cli_rejects_bad_specs(module_py, tmp_path, spec, msg):
    with pytest.raises(SystemExit) as e:
        cli.main([spec.format(f=module_py), "--output", str(tmp_path / "o.html")])
    assert msg in str(e.value)
