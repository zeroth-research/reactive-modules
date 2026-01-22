from .zrth import *
from .context import Context, get_ctx, set_ctx, reset_ctx
from .module import ReactiveModule


#####################################################################
# Wire/Term helpers
#####################################################################

# Add type aliases to the DType object
DType.Bool = DType.TensorBool([1])
DType.Int = DType.TensorInt([1])
DType.Float = DType.TensorFloat([1])


def to_wire(w: Wire | Term) -> Wire:
    """
    Take a wire or a term and get a wire for it.
    For wires, this function is identity, for terms it returns the write wire
    (which we can do, because our terms has a single unique write wire)
    """
    if isinstance(w, Wire):
        return w

    if isinstance(w, Term):
        assert len(w.write()) == 1
        return w.write()[0]

    raise ValueError(f"Invalid argument, expected Wire or Term, got {type(w)}")


def mk_term(itype, write, read=None) -> Term:
    if read is None:
        read = []

    return Term(itype, [to_wire(w) for w in write], [to_wire(w) for w in read])
