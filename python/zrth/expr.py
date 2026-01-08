class Expr:
    """
    A minimalistic representation of an expression.

    We use this class during translating Python code into reactive modules terms.
    It is nothing else than a symbolic representation of a computation of a value:
    an `Expr` is basically a node in an abstract-syntax tree. As such, it has
    some attributes and *children*.

    We deliberately use strings for operations so that this class
    is as flexible as possible, it is temporary until we have a global IType.

    :param op: name of the operation of the expression
    :param ty: a string representing the type of the expression
    :param args: a list of expression arguments (children)
    """

    __cnt = 0

    def __init__(self, op: str, args: list):
        self.op = op
        self.args = args

        Expr.__cnt += 1
        self.id = Expr.__cnt

    def __str__(self) -> str:
        return f"<{self.id}> {self.op}({', '.join(map(str, self.args))})"

    def get_children(self) -> list:
        return self.args


###
# Transform an expression using a visitor pattern
class Transform:
    def transform(self, formula):
        return self._visit(formula)

    def default(self, expr: Expr, args: list):
        return [expr]

    def _visit(self, expr: Expr, depth=0):
        # print(" " * depth, "Visiting", expr)

        before_all = getattr(self, "before_all", None)

        translated_args = []
        if isinstance(expr, Expr):
            for child in expr.get_children():
                translated_args.extend(self._visit(child, depth + 1))

        if before_all is not None:
            before_all(expr, translated_args)

        # this is an expression, recur into the children
        if isinstance(expr, Expr):
            # operations are named 'group.group.op' (with arbitrary many groups).
            # Find the most specific handler by checking methods
            #  visit_group_group_op
            #  visit_group_group
            #  visit_group
            names = expr.op.split(".")
            op = []
            # find the most generic method the transformer has
            while names:
                method = getattr(self, f"visit_{'_'.join(names)}", None)
                if method:
                    break
                op.append(names[-1])
                names = names[:-1]

            if method:
                if op:
                    # partially qualified name, add the "op" parameter
                    op = reversed(op)
                    return method(expr, translated_args, ".".join(op))
                # fully qualified name, do not add the "op" parameter
                return method(expr, translated_args)

        else:
            method = getattr(self, f"visit_type_{type(expr)}", None)
            if method is None:
                # try a generic type visitor
                method = getattr(self, "visit_type", None)
            if method:
                return method(expr, translated_args)

        if method is None:
            return self.default(expr, translated_args)
