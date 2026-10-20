"""The module's encodings agree with what the module does.

`smt.encode` turns a module into cvc5 terms, which every SMT route, the
pre-check and the fbk bridge reason about; `codegen/terms.py` turns it into Lean,
which every certificate is a theorem about. Nothing relates the two except
that they were written to mean the same thing. The equivalence theorems
relate the Lean encodings to each other, and can stay provable while all of
them are wrong: `argmax1d_scalar_n_eq` did while both its sides had argmax's
tie-breaking wrong (`test_lean_argmax.py`).

So each encoding is held to the module's own semantics instead: `zrth.eval`,
the interpreter `zrth.gym` runs modules with, on random runs. This file is
the SMT side, one round at a time: the encoded round with the round's state
and inputs as literals, which cvc5's rewriter evaluates without deciding
anything, against the round the interpreter takes from the same state. The
Lean side is `tests/lean/test_lean_agree.py`, which `#eval`s the functional
encoding on the same runs. Equal to one reference, the two encodings are
equal to each other, and nothing is parsed back.

The corpus is every module the suite already has -- `tests/limits/mods`,
the CLI's fixtures and the SV-COMP corpus -- and a one-op module for each
encoded op none of them reaches, which is most of BV. `agree.py` holds the
runs and the comparison, so the Lean side asks the same questions.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cvc5")

from tests.agree import CORPUS, Skip, disagreements_smt  # noqa: E402


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_the_smt_encoding_takes_the_rounds_the_module_takes(name):
    try:
        bad = disagreements_smt(CORPUS[name]())
    except Skip as e:
        pytest.skip(str(e))
    assert not bad, "\n".join(bad)


def test_a_wrong_emitter_is_caught(monkeypatch):
    """The check can fail: with `Sub`'s SMT emitter swapping its operands,
    the countdown's every round lands elsewhere. A check that never fails
    proves nothing about the encodings that pass it."""
    import dataclasses

    from zrth.lean.smt import encode as smt_encode

    real = smt_encode.smt_emitter

    def swapped(itype):
        emit = real(itype)
        if type(itype).__name__.endswith("_Sub"):
            return lambda op: emit(dataclasses.replace(op, args=op.args[::-1]))
        return emit

    monkeypatch.setattr(smt_encode, "smt_emitter", swapped)
    bad = disagreements_smt(CORPUS["limits/m_countdown"]())
    assert bad and "by the interpreter" in bad[0]
