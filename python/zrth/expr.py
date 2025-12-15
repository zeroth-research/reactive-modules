class Expr:
    """
    A minimalistic representation of an expression
    """

    __cnt = 0

    def __init__(self, op: str, args: list):
        self.op = op
        self.args = args

        Expr.__cnt += 1
        self.id = Expr.__cnt

    def __rmatmul__(self, rhs):
        return Expr("matmul", [self, rhs])

    def __add__(self, rhs):
        return Expr("add", [self, rhs])

    def __lt__(self, rhs):
        return Expr("lt", [self, rhs])

    def __gt__(self, rhs):
        return Expr("gt", [self, rhs])

    def __or__(self, rhs):
        return Expr("or", [self, rhs])

    def __invert__(self):
        return Expr("not", [self])

    def __str__(self) -> str:
        return f'<{self.id}> {self.op}({", ".join(map(str, self.args))})'

    def get_children(self) -> list:
        op = self.op
        if op in ("const", "var"):
            return []
        return self.args


class Var(Expr):
    def __init__(self, name):
        super().__init__("var", [])
        self.name = name

    def __str__(self):
        return f"Var({self.name})"
