"""SMV input: a NuSMV model, read as NuSMV reads it or refused."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .api import InputAdapter, InputError, Problem, SourceText
from .names import Names, usable


def _load(source: SourceText, opts: Mapping[str, Any]) -> Problem:
    from lark.exceptions import LarkError

    from zrth.smv import SMVRefused, parse_smv

    try:
        module, variables = parse_smv(source.text, strict=True)
    except SMVRefused as e:
        raise InputError(f"{source.path}: {e}") from e
    except LarkError as e:
        raise InputError(f"{source.path} is not SMV verith reads: {e}") from e
    ctrl, extl = list(module.ctrl), list(module.extl)
    # A variable the module never reads (an unused IVAR) is not one of its
    # variables, and a name a claim could not use is left positional.
    named = {n: v for n, v in variables.items()
             if (v in ctrl or v in extl) and usable(n)}
    return Problem(module, source, Names.of(module, named))


ADAPTER = InputAdapter(
    name="smv",
    summary=(
        "a NuSMV model (one MODULE main), read as NuSMV reads it -- a model "
        "that would mean something else is refused -- with claims in its "
        "variable names"
    ),
    suffixes=(".smv",),
    load=_load,
)
