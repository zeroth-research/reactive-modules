from zrth import _zrth

ll = _zrth.torch.ll

Wire = ll.Wire
Term = ll.Term
Module = ll.Module
DType = ll.DType
IType = ll.IType

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
