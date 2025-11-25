class Node:
    def __init__(self, children: list):
        self.children = children

    def visit_dfs(self, fun):
        def _visit(lvl, nd):
            for child in nd.children:
                _visit(lvl + 1, child)

            fun(lvl, child)

        _visit(0, self)

    def visit_bfs(lvl, nd):
        def _visit_bfs(lvl, nd):
            for child in nd.children:
                fun(lvl + 1, child)
                _visit_bfs(lvl + 1, child)

        fun(lvl, child)
        _visit_bfs(0, self)


class Expr(Node):
    pass


class Const(Expr):
    def __init__(self, value, ty: str):
        super().__init__([])
        self.value = value
        self.type = ty

    def __str__(self) -> str:
        return f"Const({self.value}: {self.type})"


class Var(Expr):
    def __init__(self, name: str, primed: bool = False):
        super().__init__([])
        self.name = name
        self.primed = primed

    def __str__(self) -> str:
        primed = "'" if self.primed else ""
        return f"Var({self.name}{primed})"


class ArithExpr(Expr):
    pass


class ArithOp(ArithExpr):
    def __init__(self, op: str, args: list):
        super().__init__(args)
        self.op = op

    def __str__(self):
        return f'{self.op}({", ".join(map(str, self.children))})'


class BoolExpr(Expr):
    pass


class BinaryBoolOp(BoolExpr):
    def __init__(self, op: str, lhs: BoolExpr, rhs: BoolExpr):
        super().__init__([lhs, rhs])
        assert op in ("and", "or")
        self.op = op

    def __str__(self):
        return f'{self.op}({", ".join(map(str, self.children))})'


class UnaryBoolOp(BoolExpr):
    def __init__(self, op: str, arg: BoolExpr):
        super().__init__([arg])
        # right now we support only negation
        assert op == "neg"
        self.op = op

    def __str__(self):
        return f"{self.op}({self.children[0]})"


class CmpOp(BoolExpr):
    def __init__(self, op: str, lhs: ArithExpr, rhs: ArithExpr):
        super().__init__([lhs, rhs])
        self.op = op

    def __str__(self):
        return f'{self.op}({", ".join(map(str, self.children))})'


class Assign(Node):
    def __init__(self, to: Var, what: Expr):
        super().__init__([to, what])
        assert isinstance(to, Var), to
        assert to.primed, "LHS of assignment must be a primed variable"

    @property
    def lhs(self):
        return self.children[0]

    @property
    def rhs(self):
        return self.children[1]

    def __str__(self) -> str:
        return f"{self.lhs} := {self.rhs}"


class GuardedCommand(Node):
    def __init__(self, cond: BoolExpr, assignments: list):
        super().__init__(assignments)
        self.condition = cond

    def __str__(self) -> str:
        return f'[] {self.condition} -> {"; ".join(map(str, self.children))}'


class GuardedProgram(Node):
    """
    A list of :class:`GuardedCommand`s with some metadata
    """

    def __init__(self, name, commands: list):
        assert all(isinstance(c, GuardedCommand) for c in commands), commands
        assert name in ("init", "update"), name
        super().__init__(commands)
        self.name = name

    def __str__(self) -> str:
        return f'{self.name}:\n{"\n".join(map(str, self.children))}'


class Module(Node):
    def __init__(self, name: str, extn: list, intf: list, atoms: list):
        super().__init__(atoms)

        self.name = name
        self.extern_variables = extn
        self.interface_variables = intf

    @property
    def atoms(self):
        return self.children


class Atom(Node):
    def __init__(
        self,
        ctrl: list,
        reads: list,
        awaits: list,
        init: GuardedProgram,
        update: GuardedProgram,
    ):
        super().__init__([init, update])
        self.ctrl = ctrl
        self.reads = reads
        self.awaits = awaits
