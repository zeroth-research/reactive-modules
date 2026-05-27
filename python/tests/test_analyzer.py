import pytest
from zrth.analyzer import AbstractInterpreter, AbstractValue, format_results
from zrth import IType, set_theory


@pytest.fixture(autouse=True)
def _theory():
    set_theory(IType.LIA)


def add(x, y):
    return x + y


class A:
    def __init__(self, z: int):
        self.x: int = z
        self.y: float = 0.0

    def add(self) -> int:
        self.z = 3
        return add(self.z, self.z)


class B:
    def __init__(self, xx: int):
        self.a = A(2)
        self.b = None
        if self.a.x == xx:
            self.a.z = 3


def test_1():
    example = """
def compute(x:int, y, config: A):
    result = x + y
    config.total = result
    
    if result > 10:
        label = "big"
        z = transform(result)
    else:
        label = "small"
        z = result * 2
    
    final = process(z, label)
    return final
"""
    interp = AbstractInterpreter(example)
    states = interp.analyze(
        {
            "x": AbstractValue.const(3),
            "y": AbstractValue.const(5),
            "config": AbstractValue.typed(object),
        }
    )
    print(format_results(states))

    print("\n" + "=" * 60 + "\n")


def test_2():
    example2 = """
def classify(score):
    if score >= 90:
        grade = "A"
        passed = True
    else:
        if score >= 70:
            grade = "B"
            passed = True
        else:
            grade = "F"
            passed = False
    return grade, passed
"""
    interp2 = AbstractInterpreter(example2)
    states2 = interp2.analyze({"score": AbstractValue.typed(int)})
    print(format_results(states2))


def test_3():
    interp2 = AbstractInterpreter(B.__init__)
    states2 = interp2.analyze({"score": AbstractValue.typed(int)})
    print(format_results(states2))
