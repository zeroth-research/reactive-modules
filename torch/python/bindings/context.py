from . import libzrm_torch

from .term import Term, Var
import inspect


class Context:
    def __init__(self):
        self.context_ = libzrm_torch.Context()
        # here we store terms during tracing a function
        # with the `trace` method.
        self.terms_ = None

    def _fresh_var_id(self):
        return self.context_.fresh_var()

    def get_var_id(self, name: str):
        return self.context_.get_var(name)

    def fresh_var(self) -> Var:
        name, new_id = self.context_.fresh_var_with_name()
        return Var(self, name, new_id)

    def var(self, name: str) -> Var:
        return Var(self, name)

    def term(self, op: str, reads: list, writes: list = None) -> Term:
        t = Term(self, op, reads, writes)
        self.add_traced_term(t)
        return t

    def trace(self, fun, *args, **kwargs) -> list:
        """
        Create a list of terms from a function `fun`. Parameters  `args` and `kwargs` are used
        for the function arguments. For example, if normally you would call `fun(a,
        3, w=5)`, now you call `Context.trace(fun, a, 3, w=5)`.  Unspecified arguments
        are replaced by variables that represent arbitrary input.

        Function `fun` is executed and the execution is _traced_. That means the terms
        are a result of a _single_ execution of the function. If the execution of the function is data-dependent,
        i.e., it contains branching on the input arguments, or if it reads external inputs,
        the terms do not describe the function entirely.
        Therefore, in order to capture the function entirely by tracing, the function cannot do any branching
        nor external inputs reading.
        You can model branching by using the `ifelse` function.

        Also, the function `fun` must return either a single value, a list of values,
        or a tuple of values, where values must be representable by PyVal.
        """

        sig = inspect.signature(fun)
        all_args = []
        n = 0
        _empty = inspect._empty
        for name, param in sig.parameters.items():
            if n < len(args):
                arg = args[n]
            elif name in kwargs:
                arg = kwargs[name]
            elif param.default != _empty:
                arg = param.default
            else:
                # create a variable for each unset parameter
                arg = Var(self, name)
            all_args.append(arg)
            n += 1

        self._start_gathering_terms()

        # we want to access the context from the function in order to
        # create terms via API that we cannot map to Python operations.
        # For that, we need to add it as a new argument.
        def wrapped_fun():
            assert 'zrm' not in fun.__globals__
            fun.__globals__['zrm'] = self
            r  = fun(*all_args)
            del fun.__globals__['zrm']
            return r

        r = wrapped_fun()

        # transform the return value into a list
        if r is None:
            r = []
        elif isinstance(r, tuple):
            r = [x for x in r]
        elif not isinstance(r, list):
            r = [r]

        return self._stop_gathering_terms(), all_args, r

    def _start_gathering_terms(self):
        self._terms = []

    def _stop_gathering_terms(self):
        tmp = self._terms
        self._terms = None

        return tmp

    def add_traced_term(self, term):
        terms = self._terms
        if terms is not None:
            terms.append(term)

    def _cmp(self, op, term1, term2):
        term = self.term(op, [term1, term2])
        return term.outvar

    def eq(self, t1, t2):
        return self._cmp("Eq", t1, t2)

    def le(self, t1, t2):
        return self._cmp("Le", t1, t2)

    def ge(self, t1, t2):
        return _cmp(self, "Ge", t1, t2)

    def lt(self, t1, t2):
        return _cmp(self, "Lt", t1, t2)

    def gt(self, t1, t2):
        return _cmp(self, "Gt", t1, t2)

    def ifelse(self, cond, iftrue, iffalse):
        neg_cond = cond.neg()
        t1 = self.term("Guard", [cond, iftrue])
        self.term("Guard", [neg_cond, iffalse], [t1.outvar])
        return t1.outvar
