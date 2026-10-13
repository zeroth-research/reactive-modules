"""Tests for ``python -m nuterm``."""
import subprocess
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).parent.parent

HEAD = '''
from zrth import Module, Int, LIA, Var, X
from zrth.analyzer import convert_method


def init():
    return 100


def update(old_x):
    {body}


def module() -> Module:
    state = Var(Int([1, 1]))
    init_terms = convert_method(init, {{}}, [X(state)], theory=LIA)
    update_terms = convert_method(update, {{"old_x": state}}, [X(state)], theory=LIA)
    return Module.sequential([state], init_terms, update_terms)
'''
COUNTDOWN = "if old_x <= 0:\n        return 0\n    return old_x - 1"


def _nuterm(*args):
    return subprocess.run([sys.executable, "-m", "nuterm", *map(str, args)],
                          capture_output=True, text=True, cwd=PKG_ROOT)


def _write(tmp_path, name, body):
    f = tmp_path / name
    f.write_text(HEAD.format(body=body))
    return f


def test_a_module_file_is_verified(tmp_path):
    r = _nuterm(_write(tmp_path, "countdown.py", COUNTDOWN))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "VERIFIED" in r.stdout and "1/1 verified" in r.stdout


def test_a_nonterminating_module_is_unverified(tmp_path):
    r = _nuterm(_write(tmp_path, "stuck.py", "if old_x < 0:\n        return 0\n    return old_x"))
    assert r.returncode == 1
    assert "unverified" in r.stdout


def test_a_bench_file_is_verified(tmp_path):
    f = tmp_path / "wrapped.py"
    f.write_text(
        HEAD.format(body=COUNTDOWN)
        + "\nfrom nuterm._bench import bench_of_module\n"
        "BENCH = bench_of_module(module(), 'wrapped')\n")
    r = _nuterm(f)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "wrapped" in r.stdout


def test_a_file_without_a_module_is_refused(tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("x = 1\n")
    r = _nuterm(f)
    assert r.returncode != 0
    assert "neither `BENCH` nor `module()`" in r.stderr


def test_export_writes_a_project(tmp_path):
    out = tmp_path / "out"
    r = _nuterm(_write(tmp_path, "countdown.py", COUNTDOWN), "--export", out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "exported" in r.stdout and any(out.iterdir())


def test_a_corpus_file_with_relative_imports_is_verified():
    r = _nuterm(PKG_ROOT / "tests/benchmarks/svcomp/dsl/genady.py")
    assert r.returncode == 0, r.stdout + r.stderr
