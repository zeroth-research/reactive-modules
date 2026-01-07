from zrth import _zrth

ll = _zrth.torch.ll

Wire = ll.Wire
Term = ll.Term
Module = ll.Module
DType = ll.DType
IType = ll.IType


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


def mk_term(itype, write, read=None):
    if read is None:
        read = []

    return Term(itype, [to_wire(w) for w in write], [to_wire(w) for w in read])


#
# class DType:
#     """
#     A simple wrapper around `ll.DType` that renames methods
#     to precisely match `torch::DType`
#     """
#
#     def Tensor(shape):
#         return ll.DType.tensor(shape)
#
#     def Bool():
#         return ll.DType.bool()
