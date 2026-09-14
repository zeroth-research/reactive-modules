"""DSL encodings of the SV-COMP ``termination-crafted-lit`` benchmarks.

Each module exposes a ``BENCH`` (:class:`._bench.Bench`). Use
``from benchmarks.svcomp import discover; discover()`` to collect them all.
"""

from ._bench import Bench, discover, pair, INT

__all__ = ["Bench", "discover", "pair", "INT"]
