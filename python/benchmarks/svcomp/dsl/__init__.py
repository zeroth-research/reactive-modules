"""DSL encodings of the sv_comp termination benchmarks (one module per program).

Each module exposes a ``BENCH`` (:class:`.._bench.Bench`). Mirrors the C sources
in ``../c``. Collected via ``benchmarks.svcomp.discover()``. ``_template.py`` is
the copy-me starting point for a new encoding (skipped by discovery).
"""
