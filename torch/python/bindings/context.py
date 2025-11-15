from . import libzrm_torch
from typing import Callable, Any

from .term import Term, Var, NextVar, Assignment
import inspect

WrappedModule = libzrm_torch.WrappedModule
WrappedAtom = libzrm_torch.WrappedAtom


class GuardImpl:
    def __init__(self, ctx: "Context"):
        self.ctx: Context = ctx
        self.cond: Term | None = None

    def __getitem__(self, cond: Term) -> "GuardImpl":
        self.cond = cond
        return self

    def __rshift__(self, rhs: Assignment | list[Assignment]) -> None:
        assert self.cond is not None
        if isinstance(rhs, Assignment):
            rhs = [rhs]

        self.ctx.gc(self.cond, rhs)


def handle_return_value(r):
    # transform the return value into a list
    if r is None:
        r = []
    elif isinstance(r, tuple):
        r = [x for x in r]
    elif not isinstance(r, list):
        r = [r]

    return [x.var if isinstance(x, Assignment) else x for x in r]


class Context:
    def __init__(self):
        self.context_ = libzrm_torch.Context()
        # here we store terms during tracing a function
        # with the `trace` method.
        self._terms: list[Term] | None = None

    def unwrap(self):
        return self.context_

    def _fresh_var_id(self) -> int:
        return self.context_.fresh_var()

    def get_var_id(self, name: str) -> int:
        return self.context_.get_var(name)

    def fresh_var(self) -> Var:
        name, new_id = self.context_.fresh_var_with_name()
        return Var(self, name, new_id)

    def var(self, name: str) -> Var:
        return Var(self, name)

    def term(self, op: str, reads: list[Var], writes: list[Var] | None = None) -> Term:
        t = Term(self, op, reads, writes)
        self.add_traced_term(t)
        return t

    def next_var(self, var: Var | str) -> NextVar:
        """
        Get next variable for `var`.
        """
        if isinstance(var, Var):
            var = var.name
        return NextVar(self, var)

    def module(
        self,
        vars: list[str],
        init: Callable[[], None],
        update: Callable[[], None],
        name: str | None = None,
    ) -> WrappedModule:
        cur_vars = [self.var(name) for name in vars]
        nxt_vars = [self.var(f"{name}'") for name in vars]

        init, _ = self.trace_with_vars(init, cur_vars)
        update, _ = self.trace_with_vars(update, cur_vars)

        cur_vars = [v.unwrap() for v in cur_vars]
        nxt_vars = [v.unwrap() for v in nxt_vars]
        atom = WrappedAtom(
            self.context_,
            cur_vars,
            nxt_vars,
            [t.unwrap() for t in init],
            [t.unwrap() for t in update],
        )

        # TODO: here we unnecessarily copy the terms (they are once copied into
        # Atom and then again into Module)
        module = WrappedModule(self.context_, cur_vars, nxt_vars, atom)
        if name is not None:
            module.set_name(name)
        return module

    def module_from_fn(
        self,
        fun: Callable[[], None],
        name: str | None = None
    ) -> WrappedModule:

        update, cur_vars, _ = self.trace(fun)

        nxt_vars = [self.var(f"{v.name}'") for v in cur_vars]

        cur_vars = [v.unwrap() for v in cur_vars]
        nxt_vars =[v.unwrap() for v in nxt_vars]


        atom = WrappedAtom(
            self.context_,
            cur_vars,
            nxt_vars,
            [],
            [t.unwrap() for t in update],
        )

        # TODO: here we unnecessarily copy the terms (they are once copied into
        # Atom and then again into Module)
        module = WrappedModule(self.context_, cur_vars, nxt_vars, atom)
        if name is not None:
            module.set_name(name)
        return module

    def trace_with_vars(self, fun: Callable, vars: list[Var]):
        """
        Trace a function, assuming given variables `vars`. The function should
        take no arguments.
        """
        self._start_gathering_terms()

        # we want to access the context from the function in order to
        # create terms via API that we cannot map to Python operations.
        # For that, we need to add it as a new argument.
        def wrapped_fun():
            assert "zrm" not in fun.__globals__
            assert "next" not in fun.__globals__
            fun.__globals__["zrm"] = self
            fun.__globals__["next"] = self.next_var
            fun.__globals__["Guard"] = GuardImpl(self)
            for var in vars:
                assert (
                    var.name not in fun.__globals__
                ), f"Module variable collision with Python globoals: `{var.name}`"
                fun.__globals__[var.name] = var

            r = fun()

            del fun.__globals__["zrm"]
            del fun.__globals__["next"]
            del fun.__globals__["Guard"]
            for var in vars:
                del fun.__globals__[var.name]
            return r

        r = wrapped_fun()

        return self._stop_gathering_terms(), handle_return_value(r)

    def trace(self, fun: Callable, *args, **kwargs):
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
            assert "zrm" not in fun.__globals__
            assert "next" not in fun.__globals__
            fun.__globals__["zrm"] = self
            fun.__globals__["next"] = self.next_var
            r = fun(*all_args)
            del fun.__globals__["next"]
            del fun.__globals__["zrm"]
            return r

        r = wrapped_fun()

        return self._stop_gathering_terms(), all_args, handle_return_value(r)

    def _start_gathering_terms(self):
        assert self._terms is None, "Already gathering terms"
        self._terms = []

    def _stop_gathering_terms(self):
        tmp = self._terms
        self._terms = None

        return tmp

    def add_traced_term(self, term: Term) -> None:
        terms = self._terms
        if terms is not None:
            terms.append(term)

    def _cmp(self, op: str, term1: Var, term2: Var) -> Var:
        term = self.term(op, [term1, term2])
        return term.outvar

    def gc(self, cond: Var, assignments: list[Assignment]):
        for assign in assignments:
            assert isinstance(assign, Assignment)
            # just create the term, it will be traced
            _ = self.term(
                "Ite",
                [cond, assign.rhs, assign.var.get_latched().outvar],
                [assign.var.outvar],
            )

    def eq(self, t1: Var, t2: Var):
        return self._cmp("Eq", t1, t2)

    def le(self, t1: Var, t2: Var):
        return self._cmp("Le", t1, t2)

    def ge(self, t1: Var, t2: Var):
        return self._cmp("Ge", t1, t2)

    def lt(self, t1: Var, t2: Var):
        return self._cmp("Lt", t1, t2)

    def gt(self, t1: Var, t2: Var):
        return self._cmp("Gt", t1, t2)

    def ite(self, cond: Var, iftrue: Var, iffalse: Var) -> Var:
        return self.term("Ite", [cond, iftrue, iffalse]).outvar

    def ifthenelse(self, cond, iftrue, iffalse):
        return self.ite(cond, iftrue, iffalse)
