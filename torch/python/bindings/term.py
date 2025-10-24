from torch import Tensor as TorchTensor
from . import libzrm_torch
Context = libzrm_torch.Context

PyVal = libzrm_torch.PyVal
PyTerm = libzrm_torch.PyTerm


def to_pyval(v):
    if isinstance(v, PyVal):
        return v
    if isinstance(v, Var):
        return v.term_
    if isinstance(v, str):  # this is a wire symbol
        return Var(v)
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

        self.term_ = PyTerm(op, reads, writes)
        self.outvar = outvar

        # libzrm_torch.print_pyterm(self.term_)

    def print(self):
        self.term_.print()

    def __str__(self) -> str:
        return str(self.term_)

    def __repr__(self) -> str:
        return f"Term({self.term_})"

    def _op(self, op, arg):
        if isinstance(arg, (int, float, TorchTensor, Var)):
            term = self.ctx_.term(op, [self.outvar, arg])
            return term.outvar
        raise NotImplementedError(
            f"Unsupported type for operation: {arg} ({type(arg)})"
        )

    def _r_op(self, op, arg):
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

    # do not convert terms to 0 or 1
    def __bool__(self):
        raise NotImplementedError("Cannot convert term to bool")

    def neg(self):
        return self.ctx_.term("Neg", [self.outvar]).outvar

    def sum(self):
        return self.ctx_.term("Sum", [self.outvar]).outvar


class Var(Term):
    """
    NOTE: This constructor should be called only by Context.
    """

    def __init__(self, ctx, name, id_=None):
        super().__init__(ctx, "Var", [])
        self.name = name

        # below I set the attributes from the partent class directly,
        # I'll burn in hell for that.
        self.outvar = self
        # The underlying PyTerm
        if id_ is None:
            id_ = ctx.get_var_id(name)
        self.term_ = PyVal.sym(id_)
