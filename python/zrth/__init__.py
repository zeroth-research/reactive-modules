from sympy.sets.fancysets import Reals

from .zrth import *

# makes sorts available on the base namespace
for _name in dir(Sort):
    if not _name.startswith('_'):
        globals()[_name] = getattr(Sort, _name)

# #####################################################################
# # IType — direct alias to the Rust IType
# #####################################################################
#
# IType = _IType
#
#
# #####################################################################
# # DType convenience constructors
# #####################################################################
#
#
# def Bool(*shape):
#     return DType.Bool([*shape] if shape else [1])
#
#
# def Int(*shape):
#     return DType.Int([*shape])
#
#
# def Real(*shape):
#     return DType.Real([*shape])
#
#
# def Float(*args):
#     return DType.Float([*args])
#
#
# #####################################################################
# # Derived-operation helpers
# #
# # These helpers lower a missing op to a sequence of primitive terms,
# # using the IType facade (so they pick up the current theory). Each
# # helper takes a caller-allocated output Wire plus input Wires and
# # returns a list[Term] whose last write is the output.
# #####################################################################
#
#
# def xnor(out: Wire, a: Wire, b: Wire) -> list[Term]:
#     """out = Xnor(a, b) ≡ Not(Xor(a, b))."""
#     tmp = Wire(out.dtype)
#     return [
#         Term(_IType.LIA.Xor(), [tmp], [a, b]),
#         Term(_IType.LIA.Not(), [out], [tmp]),
#     ]
#
#
# def implies(out: Wire, a: Wire, b: Wire) -> list[Term]:
#     """out = Implies(a, b) ≡ Or(Not(a), b)."""
#     nota = Wire(a.dtype)
#     return [
#         Term(_IType.LIA.Not(), [nota], [a]),
#         Term(_IType.LIA.Or(), [out], [nota, b]),
#     ]
#
#
# # --- BV-only: two's-complement-derived ops -----------------------------------
#
#
# def bv_neg(out: Wire, x: Wire) -> list[Term]:
#     """out = -x via two's complement ≡ Add(Not(x), 1).
#
#     Bit-width of the result is inferred from `out.dtype`. The intermediate
#     constant 1 carries the same width via BV's type inference.
#     """
#     notx = Wire(x.dtype)
#     one = Wire(out.dtype)
#     return [
#         Term(_IType.BV.Const(1), [one]),
#         Term(_IType.BV.Not, [notx], [x]),
#         Term(_IType.BV.Add, [out], [notx, one]),
#     ]
#
#
# def bv_sub(out: Wire, a: Wire, b: Wire) -> list[Term]:
#     """out = a - b ≡ Add(a, bv_neg(b))."""
#     negb = Wire(b.dtype)
#     terms = bv_neg(negb, b)
#     terms.append(Term(_IType.BV.Add, [out], [a, negb]))
#     return terms
#
#
# def bv_mod(out: Wire, a: Wire, b: Wire, *, signed: bool = False) -> list[Term]:
#     """out = a mod b ≡ Sub(a, Mul(Div(a, b), b)).
#
#     `signed=False` uses `UDiv`; `signed=True` uses `SDiv`.
#     """
#     div = Wire(out.dtype)
#     prod = Wire(out.dtype)
#     div_op = _IType.BV.SDiv if signed else _IType.BV.UDiv
#     terms = [
#         Term(div_op, [div], [a, b]),
#         Term(_IType.BV.Mul, [prod], [div, b]),
#     ]
#     terms.extend(bv_sub(out, a, prod))
#     return terms
#
#
# from .gym import Wrapper, Env
# from .smv import parse_smv
# from .smt import z3
# from .builder import LRATermBuilder, LIATermBuilder, BVTermBuilder, builder_for
#
# # Submodule access: from zrth.gym import Env, Wrapper
# #                   from zrth.torch import Module
# from . import gym as gym
# from . import torch as torch
#
# __all__ = [
#     "Wire",
#     "DType",
#     "IType",
#     "Term",
#     "Module",
#     "Wrapper",
#     "Env",
#     "builder_for",
#     "xnor",
#     "implies",
#     "bv_neg",
#     "bv_sub",
#     "bv_mod",
#     "LRATermBuilder",
#     "LIATermBuilder",
#     "BVTermBuilder",
# ]
