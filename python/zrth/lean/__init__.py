"""Lean 4 certificate generation for reactive modules.

`TA2Magic` and `TA2MagicAI` resolve on first use rather than at import:
`magic.ai` imports `anthropic` at module scope (236 ms of the 850 ms a
`verith --help` takes), and the routes that need an LLM are imported by the
row that selects them -- which is the discipline `infer_route` documents and
an eager import here defeated for every invocation.
"""

from .cert import CertificateData
from .translate import ModuleToLean4

__all__ = ["ModuleToLean4", "CertificateData", "TA2Magic", "TA2MagicAI"]

_LAZY = {"TA2Magic": ".magic", "TA2MagicAI": ".magic.ai"}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)
