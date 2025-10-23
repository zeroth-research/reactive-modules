from torch import Tensor as TorchTensor
from . import libzrm_torch

from inspect import signature, _empty

PyVal = libzrm_torch.PyVal
PyTerm = libzrm_torch.PyTerm

# While tracing the execution of a python function,
# we gather the terms in this global variable.
# This is to support creating terms also using functions
# (like `guard(term1, term2)`) and not only via methods of `Term`.
# That also means that tracing is **not thread-safe**
_terms = None


def start_gathering_terms():
    global _terms
    _terms = []


def stop_gathering_terms():
    global _terms
    tmp = _terms
    _terms = None

    return tmp


def add_term(term):
    global _terms
    if _terms is not None:
        _terms.append(term)


def to_pyval(v):
    if isinstance(v, PyVal):
        return v
    if isinstance(v, Var):
        return v.term_
    if isinstance(v, str):  # this is a wire symbol
        return Var(v)
    return PyVal(v)


class Term:
    outvars_cnt = 1

    def __init__(self, op, reads, writes=None):

        if op == "Var":
            # fields are set in the parent ctor
            return

        if writes is None:
            outvar = Var(f"x__{Term.outvars_cnt}")
            Term.outvars_cnt += 1
        else:
            # more outputs is unsupported yet
            assert len(writes) == 1
            outvar = writes[0]

        reads = [to_pyval(r) for r in reads]
        writes = [to_pyval(outvar)]

        self.term_ = PyTerm(op, reads, writes)
        self.outvar = outvar

        add_term(self)

        # libzrm_torch.print_pyterm(self.term_)

    def print(self):
        self.term_.print()

    def __str__(self) -> str:
        return str(self.term_)

    def __repr__(self) -> str:
        return f"Term({self.term_})"

    def _op(self, op, arg):
        if isinstance(arg, (int, float, TorchTensor, Var)):
            term = Term(op, [self.outvar, arg])
            return term.outvar
        raise NotImplementedError(
            f"Unsupported type for operation: {arg} ({type(arg)})"
        )

    def _r_op(self, op, arg):
        if isinstance(arg, (int, float, TorchTensor, Var)):
            term = Term(op, [arg, self.outvar])
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
        return Term("Neg", [self.outvar]).outvar

    def sum(self):
        return Term("Sum", [self.outvar]).outvar


class Var(Term):

    # a map of names to numeric identifiers that are used in wires.
    # In the future, it will probably become a part of some other class,
    # some builder class that builds modules (because now all modules
    # shared these names, no matter if they are in the same file or not)
    name_to_id = {}

    def __init__(self, name):
        super().__init__("Var", [])
        self.name = name

        # below I set the attributes from the partent class directly,
        # I'll burn in hell for that.
        self.outvar = self

        name_to_id = Var.name_to_id
        id_ = name_to_id.get(name)
        if id_ is None:
            id_ = len(name_to_id)
            name_to_id[name] = id_

        self.term_ = PyVal.sym(id_)
        self.id_ = id_

        assert name_to_id.get(name) == id_

def _cmp(op, term1, term2):
    term = Term(op, [term1, term2])
    return term.outvar


def eq(t1, t2):
    return _cmp("Eq", t1, t2)


def le(t1, t2):
    return _cmp("Le", t1, t2)


def ge(t1, t2):
    return _cmp("Ge", t1, t2)


def lt(t1, t2):
    return _cmp("Lt", t1, t2)


def gt(t1, t2):
    return _cmp("Gt", t1, t2)


def ifelse(cond, iftrue, iffalse):
    neg_cond = cond.neg()
    t1 = Term("Guard", [cond, iftrue])
    Term("Guard", [neg_cond, iffalse], [t1.outvar])
    return t1.outvar


# sometimes we might want to explicitly create a variable
# to replace some concrete values
def var(name: str) -> Var:
    return Var(name)


def to_terms(fun, *args, **kwargs):
    """
    Create a list of terms from a function `fun`.  `args` and `kwargs` are used
    for the function arguments. For example, if normally you would call `fun(a,
    3, w=5)`, now you call `to_terms(fun, a, 3, w=5)`.  Unspecified arguments
    are replaced by variables that represent arbitrary input.

    The function cannot do any branching. Instead of branching, one can use
    `ifelse` function from our bindings.
    Also, the function must return either a single value, a list of values,
    or a tuple of values, where values must be representable by PyVal.
    """

    sig = signature(fun)
    all_args = []
    n = 0
    for name, param in sig.parameters.items():
        if n < len(args):
            arg = args[n]
        elif name in kwargs:
            arg = kwargs[name]
        elif param.default != _empty:
            arg = param.default
        else:
            # create a variable for each unset parameter
            arg = Var(name)
        all_args.append(arg)
        n += 1

    start_gathering_terms()
    r = fun(*all_args)

    # transform the return value into a list
    if r is None:
        r = []
    elif isinstance(r, tuple):
        r = [x for x in r]
    elif not isinstance(r, list):
        r = [r]

    return stop_gathering_terms(), all_args, r
