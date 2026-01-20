from .zrth import *
from .context import Context


#####################################################################
# Wire/Term helpers
#####################################################################


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


#####################################################################
# Global context
#####################################################################

# Term are created in this global context.
# The context can be switched manually, but that is for advanced users
_global_context = Context()


def set_ctx(ctx: Context) -> Context:
    global _global_context
    old = _global_context
    _global_context = ctx
    return old


def get_ctx() -> Context:
    global _global_context
    return _global_context
