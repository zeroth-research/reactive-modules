from zrth.expr import Expr, Var, Transform
from itertools import chain
from torch import Tensor

from zrth import _zrth

PyVal = _zrth.PyVal


def to_terms(ctx, ctrl, extl, init_ret, update_ret) -> (list, list, list, list):
    """
    MODIFIES `ctx`
    """
    walker = ExprToTerms(ctx)

    def var_to_pyval(var: Var) -> PyVal:
        return ctx.unwrap().var(var.name)

    init_terms = []
    assert len(init_ret) == len(ctrl)
    for var, expr in zip(ctrl, init_ret):
        tmp, outvar = walker.translate(expr)
        init_terms.extend(tmp)

        # map the output of the expression to the output wire
        assert len(outvar) == 1, outvar
        init_terms.append(WrappedTerm("Id", outvar, [var_to_pyval(ctx.next_var(var))]))

    update_terms = []
    assert len(update_ret) == len(ctrl)
    for var, expr in zip(ctrl, update_ret):
        tmp, outvar = walker.translate(expr)
        update_terms.extend(tmp)

        # map the output of the expression to the output wire
        assert len(outvar) == 1
        update_terms.append(
            WrappedTerm("Id", outvar, [var_to_pyval(ctx.next_var(var))])
        )

    cur_vars = [var_to_pyval(v) for v in chain(ctrl, extl)]
    nxt_vars = [var_to_pyval(ctx.next_var(v)) for v in chain(ctrl, extl)]
    return cur_vars, nxt_vars, init_terms, update_terms


class ExprToTerms(Transform):
    def __init__(self, ctx) -> None:
        """
        MODIFIES `ctx`
        """
        self.ctx = ctx
        # FIXME: this is confusing
        self._ctx = ctx.unwrap()
        self.terms = []

    def translate(self, formula):
        """
        Translate a formula into reactive module terms.
        Return the terms and the output variable (wire) for the formula.
        """
        self.terms = []
        r = self.transform(formula)
        return self.terms, r

    def default(self, expr, args):
        raise NotImplementedError(f"Not implemented translation of operation: `{expr}`")

    # def before_all(self, expr: Expr, args: list):
    #    print("VIS", expr, type(expr))

    # def forall(self, expr: Expr, args: list):
    #    assert all(isinstance(a, PyVal) for a in args), args

    def visit_type_bool(self, expr):
        return [PyVal.bool(expr)]

    def visit_type_int(self, expr):
        return [PyVal.int(expr)]

    def visit_type(self, expr, args):
        if isinstance(expr, Tensor):
            return [PyVal.tensor(expr)]
        return self.default(expr, args)

    def visit_var(self, expr, args):
        return [self._ctx.var(expr.name)]

    def visit_arith(self, expr, args, op):
        assert len(expr.args) == 2
        assert args[0].dtype() == args[1].dtype(), args
        out = self._ctx.tmp_var()
        self.terms.append(WrappedTerm(arith_to_str(op), reads=args, writes=[out]))
        return [out]

    def visit_cmp(self, expr, args, op):
        assert len(args) == 2
        out = self._ctx.tmp_var()
        self.terms.append(WrappedTerm(rel_to_str(op), reads=args, writes=[out]))
        return [out]

    def visit_logic(self, expr, args, op):
        if op == "not":
            assert len(args) == 1
            out = self._ctx.tmp_var()
            self.terms.append(WrappedTerm("Not", reads=args, writes=[out]))
        else:
            assert len(args) == 2
            assert op in ("and", "or"), op
            out = self._ctx.tmp_var()
            op = "Or" if op == "or" else "And"
            self.terms.append(WrappedTerm(op, reads=args, writes=[out]))
        return [out]

    def visit_ite(self, expr, args):
        assert len(args) == 3
        assert args[1].dtype() == args[2].dtype(), args
        out = self._ctx.tmp_var()
        self.terms.append(WrappedTerm("Cond", reads=args, writes=[out]))
        return [out]

    def visit_ifthen(self, expr, args):
        assert len(args) == 2
        out = self._ctx.tmp_var()
        self.terms.append(WrappedTerm("IfThen", reads=args, writes=[out]))
        return [out]

    def visit_choose(self, expr, args):
        assert len(args) > 0, args
        out = self._ctx.tmp_var()
        self.terms.append(WrappedTerm("Choose", reads=args, writes=[out]))
        return [out]

    def visit_choose_or(self, expr, args):
        assert len(args) > 0, args
        out = self._ctx.tmp_var()
        self.terms.append(WrappedTerm("ChooseOr", reads=args, writes=[out]))
        return [out]


def rel_to_str(op: str) -> str:
    if op == "lt":
        return "Lt"
    if op == "gt":
        return "Gt"
    if op == "le":
        return "Le"
    if op == "ge":
        return "Ge"
    if op == "eq":
        return "Eq"

    raise NotImplementedError(f"Unknown relation: {opty}")


def arith_to_str(op) -> str:
    if op == "add":
        return "Add"
    if op == "mul":
        return "Mul"
    if op == "matmul":
        return "MatMul"

    raise NotImplementedError(f"Unknown operation: {op}")
