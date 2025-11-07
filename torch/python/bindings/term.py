from torch import Tensor as TorchTensor
from . import libzrm_torch

Context = libzrm_torch.Context

PyVal = libzrm_torch.PyVal
WrappedTerm = libzrm_torch.WrappedTerm


def to_pyval(v) -> PyVal:
    if isinstance(v, PyVal):
        return v
    if isinstance(v, Var):
        return v.term_
    if isinstance(v, str):  # this is a wire symbol
        return Var(v).term_
    return PyVal(v)


class Term:
    def __init__(self, ctx, op, reads, writes=None):
        """
        NOTE: This constructor should be called only by Context.
        """
        self.ctx_ = ctx

        if op == "Var":
            # fields are set in the parent ctor
            return

        if writes is None:
            outvar = ctx.fresh_var()
        else:
            # more outputs is unsupported yet
            assert len(writes) == 1
            outvar = writes[0]

        reads = [to_pyval(r) for r in reads]
        writes = [to_pyval(outvar)]

        self.term_: WrappedTerm = WrappedTerm(op, reads, writes)
        self.outvar: Var = outvar

        # libzrm_torch.print_pyterm(self.term_)

    def print(self):
        self.term_.print()

    def wrapped_term(self) -> WrappedTerm:
        return self.term_

    def __str__(self) -> str:
        return str(self.term_)

    def __repr__(self) -> str:
        return f"Term({self.term_})"

    def _op(self, op, arg) -> "Term":
        if isinstance(arg, (int, float, TorchTensor, Var)):
            term = self.ctx_.term(op, [self.outvar, arg])
            return term.outvar
        raise NotImplementedError(
            f"Unsupported type for operation: {arg} ({type(arg)})"
        )

    def _r_op(self, op, arg) -> "Term":
        if isinstance(arg, (int, float, TorchTensor, Var)):
            term = self.ctx_.term(op, [arg, self.outvar])
            return term.outvar
        raise NotImplementedError(
            f"Unsupported type for operation: {arg} ({type(arg)})"
        )

    def __add__(self, rhs):
        return self._op("Add", rhs)

    def __mul__(self, rhs):
        return self._op("Mul", rhs)

    def __radd__(self, rhs):
        return self._r_op("Add", rhs)

    def __rmul__(self, rhs):
        return self._r_op("Mul", rhs)

    def __lt__(self, rhs):
        return self._op("Lt", rhs)

    def __rlt__(self, rhs):
        return self._r_op("Lt", rhs)

    # TODO: other relational operators

    # logical or
    def __or__(self, rhs: "Term") -> "Term":
        return self.ctx_.term("Or", [self.outvar, rhs.outvar]).outvar

    # logical and
    def __and__(self, rhs: "Term") -> "Term":
        return self.ctx_.term("And", [self.outvar, rhs.outvar]).outvar

    # logical negation: ~
    def __invert__(self) -> "Term":
        return self.neg()

    # do not convert terms to 0 or 1
    def __bool__(self):
        raise NotImplementedError("Cannot convert term to bool")

    def neg(self) -> "Term":
        return self.ctx_.term("Neg", [self.outvar]).outvar

    def sum(self) -> "Term":
        return self.ctx_.term("Sum", [self.outvar]).outvar


class Var(Term):
    """
    NOTE: This constructor should be called only by Context.
    """

    def __init__(self, ctx: Context, name: str, id_=None):
        super().__init__(ctx, "Var", [])
        self.name: str = name

        # below I set the attributes from the parent class directly,
        # I'll burn in hell for that.
        self.outvar: Var = self
        # The underlying WrappedTerm
        if id_ is None:
            id_ = ctx.get_var_id(name)
        self.term_: PyVal = PyVal.sym(id_)


class Assignment:
    def __init__(self, lhs: "NextVar", rhs: Term):
        self.var = lhs
        self.rhs = rhs

    def __str__(self):
        return f"{self.var} := {self.rhs}"


class NextVar(Var):
    """
    A class for primed variables. For these, we want to override some operators
    differently. Otherwise, they are the same as Var.

    :param: name  The name of **current state** variable.
    """

    def __init__(self, ctx: Context, name: str):
        assert isinstance(name, str), (name, type(name))
        super().__init__(ctx, f"{name}'")
        # we store also the current variable name, so that we can easily
        # get the latched variable from this one
        self.latched_name: str = name

    def get_latched(self) -> Var:
        return self.ctx_.var(self.latched_name)

    def __eq__(self, rhs: Term) -> Assignment:
        """
        Operator == works as assignment for primed variables.
        """
        return Assignment(self, rhs)
