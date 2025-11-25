from lark import Lark, Transformer
from rm_ast import *


class ASTTransformer(Transformer):

    def constint(self, items):
        return Const(int(items[0]), "Int")

    def const_true(self, items):
        return True

    def const_false(self, items):
        return False

    def boolconst(self, items):
        assert len(items) == 1
        assert isinstance(items[0], bool), items
        return Const(items[0], "Bool")

    def constant(self, items):
        if isinstance(items[0], Const):
            return items[0]
        raise NotImplementedError(f"Unhandled constant: {items}")

    def primed_var(self, items):
        name = items[0].children[0]
        return Var(name, primed=True)

    def var(self, items):
        name = items[0].children[0]
        return Var(name, primed=False)

    def arithexpr(self, items):
        if len(items) == 1:
            item = items[0]
            if isinstance(item, (Var, Const, ArithOp)):
                return item

        raise NotImplementedError(f"Unhandled arith. expression: {items}")

    def add(self, items):
        return ArithOp("add", items)

    def mul(self, items):
        return ArithOp("mul", items)

    def sub(self, items):
        return ArithOp("sub", items)

    def div(self, items):
        return ArithOp("div", items)

    def cmp(self, items):
        op = items[1].data

        if op == "cmp_neq":
            return BinaryBoolOp("not", CmpOp("EQ", items[0], items[2]))
        elif op == "cmp_lt":
            op = "LT"
        elif op == "cmp_gt":
            op = "GT"
        elif op == "cmp_le":
            op = "LE"
        elif op == "cmp_ge":
            op = "GE"
        elif op == "cmp_eq":
            op = "EQ"
        else:
            raise NotImplementedError(f"Unknown comparison: {op}")

        return CmpOp("EQ", items[0], items[2])

    def lor(self, items):
        assert len(items) == 2
        return BinaryBoolOp("or", items[0], items[1])

    def land(self, items):
        assert len(items) == 2
        return BinaryBoolOp("and", items[0], items[1])

    def neg(self, items):
        assert len(items) == 1
        return UnaryBoolOp("neg", items[0])

    def assign(self, items):
        return Assign(items[0], items[1])

    def guarded_command(self, items):
        assert len(items) == 2
        assert items[0].data == "boolexpr", items[0]
        assert len(items[0].children) == 1, items[0].children

        assert items[1].data == "assignments", items[1]
        assert len(items[1].children) >= 1, items[1].children

        cond = items[0].children[0]
        return GuardedCommand(cond, items[1].children)

    def atom_update(self, items):
        assert len(items) >= 1
        return GuardedProgram("update", items)

    def atom_init(self, items):
        assert len(items) >= 1
        return GuardedProgram("init", items)

    def atom_variables(self, items):
        assert len(items) > 0, items
        atom_vars = {}
        for item in items:
            name = item.data
            children = item.children
            assert len(children) == 1, children
            assert children[0].data == "var_list", children[0]
            vars = children[0].children
            print(vars)
            if name == "atom_controls":
                atom_vars["controls"] = vars
            elif name == "atom_reads":
                atom_vars["reads"] = vars
            elif name == "atom_awaits":
                atom_vars["awaits"] = vars
            else:
                raise NotImplementedError(f"Unknown type of atom variables: {name}")
        return self


def print_nd(lvl, nd):
    print(" " * lvl * 2, nd)


def parse(path: str) -> Module:
    """
    Parse :param:`path`
    """
    parser = Lark.open("grammar.lark", rel_to=__file__, start="start")
    input = open(path, "r").read()
    return ASTTransformer().transform(parser.parse(input))


if __name__ == "__main__":
    from sys import argv

    ast = parse(argv[1])
    print(ast.pretty())
    # ast.visit_dfs(print_nd)
