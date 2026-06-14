import torch
import zrth

from zrth import BV, Term, Wire

a = BV.Const(torch.tensor([1.0]))

match a:
    case BV.Const(_):
        print('hello')
    case _:
        print('meh')

match type(a):
    case BV.Const:
        print('hello')
    case _:
        print('meh')

t = zrth.Real([1, 1])

print(isinstance(t, zrth.Sort))
