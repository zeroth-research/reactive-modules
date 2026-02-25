from zrth.gym.analyzer import AccessAnalyzer


def foo1(x: list[int], y: tuple[int, float], z: int) -> int:
    ty = y[0]
    return x[0] + x[1] + ty + z


def test_simple():
    summaries = AccessAnalyzer().analyze(foo1)
    for fn, sm in summaries.items():
        print(sm)


def foo2(x: list[int], y: tuple[int, float]) -> int:
    z = foo1(x, y, 4)
    x = []
    return z


def test_call():
    summaries = AccessAnalyzer().analyze(foo2)
    for fn, sm in summaries.items():
        print(sm)


def add(x, y):
    return x + y


class A:
    def __init__(self):
        self.x: int = 0
        self.y: float = 0.0

    def add(self) -> int:
        self.z = 3
        return add(self.x, self.y)


def test_class():
    summaries = AccessAnalyzer().analyze(A.add)
    for fn, sm in summaries.items():
        print(sm)


class B:
    def __init__(self, xx: int):
        self.a: A = A()
        if self.a.x == xx:
            tmp: float = self.a.y
            self.a.z = 3


def test_nested_class():
    summaries = AccessAnalyzer().analyze(B.__init__)
    for fn, sm in summaries.items():
        print(sm)
