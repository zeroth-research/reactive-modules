import unittest

from zrth.torch.ll import Wire, Term, DType, IType


class MyTestCase(unittest.TestCase):
    def test_something(self):
        x = Wire.new(0, DType.C)
        y = Wire.new(1, DType.D)
        f = Term.function(IType.A, [x, y], [x])
        print(f)


if __name__ == '__main__':
    unittest.main()
