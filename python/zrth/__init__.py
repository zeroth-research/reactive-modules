from .zrth import (
    Wire,
    DType,
    IType,
    Term,
    RustContext,
    Transition,
    WiredTransitions,
    Module,
)

from .interpreter import Interpreter

from .context import Context, get_ctx, set_ctx, reset_ctx
from .module import ReactiveModule

from typing import Generator


#####################################################################
# IType and DType
#####################################################################


# Save original Rust enum constructors before monkey-patching
_orig_Bool = DType.TensorBool
_orig_Int = DType.TensorInt
_orig_Float = DType.TensorFloat
_orig_Real = DType.TensorReal


# Add type aliases and convenient function
def mk_DTypeBool(shape: None | list[int] = None) -> DType:
    """
    Create a Bool DType. If shape is given, create a tensor of bools
    """
    if shape is None:
        shape = [1]
    return _orig_Bool(shape)


def mk_DTypeInt(shape: None | list[int] = None) -> DType:
    """
    Create a Int DType. If shape is given, create a tensor of ints
    """
    if shape is None:
        shape = [1]
    return _orig_Int(shape)


def mk_DTypeFloat(shape: None | list[int] = None) -> DType:
    """
    Create a Float DType. If shape is given, create a tensor of floats
    """
    if shape is None:
        shape = [1]
    return _orig_Float(shape)


def mk_DTypeReal(shape: None | list[int] = None) -> DType:
    """
    Create a Real DType. If shape is given, create a tensor of reals
    """
    if shape is None:
        shape = [1]
    return _orig_Real(shape)


#####################################################################
# Wire/Term helpers
#####################################################################

DType.Bool = mk_DTypeBool  # ty: ignore
DType.Int = mk_DTypeInt  # ty: ignore
DType.Float = mk_DTypeFloat  # ty: ignore
DType.Real = mk_DTypeReal  # ty: ignore


# Add type aliases to the DType object
def Bool(*shape):
    return DType.Bool([*shape])


def Int(*shape):
    return DType.Int([*shape])


def Real(*shape):
    return DType.Real([*shape])


def Float(*args):
    return DType.Float([*args])


def to_wire(w: Wire | Term) -> Wire:
    """
    Take a wire or a term and get a wire for it.
    For wires, this function is identity, for terms it returns the write wire
    (which we can do, because our terms has a single unique write wire)
    """
    if isinstance(w, Wire):
        return w

    if isinstance(w, Term):
        assert len(w.write) == 1
        return w.write[0]

    raise ValueError(f"Invalid argument, expected Wire or Term, got {type(w)}")


def mk_term(itype, write, read=None) -> Term:
    if read is None:
        read = []

    return Term(itype, [to_wire(w) for w in write], [to_wire(w) for w in read])


#####################################################################
# Transitions
#####################################################################


# override WiredTransitions.wire_transition to pass there the global context
orig_wire_transition = WiredTransitions.wire_transition

WiredTransitions.wire_transition = lambda self, t: orig_wire_transition(
    self, t, get_ctx().unwrap()
)


def process_subst_pair(
    lhs: Wire, rhs: Wire
) -> Generator[tuple[Wire, Wire], None, None]:
    raise NotImplementedError()


# if isinstance(lhs, Sym):
#    if lhs.has_nxt():
#        lhs = (lhs.wire(), lhs.nxt().wire())
#    else:
#        lhs = (lhs.wire(),)
# else:
#    assert isinstance(lhs, Wire), lhs
#    lhs = (lhs,)
# if isinstance(rhs, Sym):
#    if rhs.has_nxt():
#        rhs = (rhs.wire(), rhs.nxt().wire())
#    else:
#        rhs = (rhs.wire(),)
# else:
#    assert isinstance(rhs, Wire), rhs
#    rhs = (rhs,)
#
# if len(lhs) != len(rhs):
#    raise RuntimeError("Invalid mapping")
# for l, r in zip(lhs, rhs):
#    yield l, r


def remap_term(term, subst: dict) -> Term:
    """
    Create a new [Term] with re-mapping wires according to substitutions given in `subst`.
    If a wire `w` is not found in `subst`, a new wire `wn` with fresh ID (and the same DType)
    is created and the mapping is updated with `w -> wn`. The updated mapping is returned
    as the second return value from this function (the first one is the new term).

    The substitutions is a dictonary mapping either [Wire]s to [Wire]s or [Sym]s to [Sym]s.
    In the case of [Sym]s, they are mapped to their wires. If a [Sym] has associated next [Sym],
    wires of both the latched and next [Sym]s are used for the substitution.
    """
    # make sure the substitution is wire-to-wire
    subst = {l: r for k, v in subst.items() for l, r in process_subst_pair(k, v)}
    ctx = get_ctx()

    def new_wire(w: Wire) -> Wire:
        map_w: int | None = subst.get(w)
        if map_w is None:
            # this is a temporary (or unmapped wire)
            # create a new mapping
            new_w = Wire(w.dtype())
            subst[w] = new_w
            return new_w
        return map_w

    read = [new_wire(w) for w in term.read()]
    write = [new_wire(w) for w in term.write()]

    return mk_term(term.itype(), write, read), subst


Term.remap = lambda self, subst: remap_term(self, subst)

__all__ = [
    "Wire",
    "DType",
    "IType",
    "Term",
    "Module",
    "RustContext",
    "Transition",
    "WiredTransitions",
    "to_wire",
    "mk_term",
    "ReactiveModule",
    "Interpreter",
]
