from zrth.lean.constants import _tensor_to_lean_inline
from zrth import Term, Wire


def _accessor(pos: int, total: int) -> str:
    """Accessor for position `pos` in a product (tuple) of `total` elements.

    total=1: '' (value itself)
    total=2: '.1', '.2'
    total=3: '.1', '.2.1', '.2.2'
    total=4: '.1', '.2.1', '.2.2.1', '.2.2.2'
    """
    if total == 1:
        return ""
    if pos == total - 1:
        return ".2" * pos
    return ".2" * pos + ".1"


def itype_name(itype) -> str:
    """Get the variant name of an IType, e.g. IType.Add() -> 'Add'."""
    name = type(itype).__name__
    # PyO3 exports variants as IType_Add, IType_Tensor, etc.
    if name.startswith("IType_"):
        return name[6:]
    return name


def _constant_expr(
    const_name: str, term: Term, w: Wire, constants: dict[int, str]
) -> str:
    if const_name == "Tensor":
        wire_id = w.id
        if wire_id in constants:
            return constants[wire_id]
        else:
            return _tensor_to_lean_inline(term.itype._0, w)
    elif const_name == "ConstBool":
        val = bool(term.itype._0)
        return "true" if val else "false"
    else:
        return str(int(term.itype._0))

    raise NotImplementedError
