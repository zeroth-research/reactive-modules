"""Tests for AI-based invariant/ranking inference."""

import os
import pytest

from zrth.lean.cert import CertificateData

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


def _make_magic_ai(source: str):
    """Lazy import to avoid failure when anthropic is not installed."""
    from zrth.lean.magic_ai import TA2MagicAI

    return TA2MagicAI(source)


COUNTER_SOURCE = """\
def init():
    return 0

def update(old_x):
    x = old_x + 1
    if x == 10:
        return 0
    return x
"""


def test_counter_reaches_zero():
    magic = _make_magic_ai(COUNTER_SOURCE)
    cd = CertificateData(prp="x == 0")
    result = magic.infer(cd)

    assert result.inv is not None
    assert result.ranking is not None
    print(f"invariant: {result.inv}")
    print(f"ranking:   {result.ranking}")
