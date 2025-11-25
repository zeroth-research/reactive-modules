from zrth import _zrth

PyVal = _zrth.toy.PyVal
WrappedTerm = _zrth.toy.WrappedTerm


# def to_pyval(v) -> PyVal:
#     if isinstance(v, PyVal):
#         return v
#     if isinstance(v, Var):
#         return v._term
#     if isinstance(v, str):  # this is a wire symbol
#         return Var(v)._term
#     return PyVal(v)
#
#
# def op_ret_type(op: str, reads: list) -> str:
#     if op in (
#         "Logical::Or",
#         "Logical::And",
#         "Logical::Not",
#         "Cmp::Lt",
#         "Cmp::Le",
#         "Cmp::Ge",
#         "Cmp::Gt",
#         "Cmp::Eq",
#         "Cmp::Ne",
#     ):
#         return "Bool"
#     if op in (
#         "Arith::Add",
#         "Arith::Mul",
#     ):
#         assert len(reads) == 2
#         assert isinstance(reads[0], Var), reads
#         return reads[0].ty()
#     raise NotImplementedError(f"Do not know the type for operation {op}")
#
#
# class Term:
#     def __init__(self, ctx, op, reads, writes=None):
#         """
#         NOTE: This constructor should be called only by Context.
#         """
#         self.ctx_ = ctx
#
#         if op == "Var":
#             # fields are set in the parent ctor
#             return
#
#         if writes is None:
#             outvar = ctx.fresh_var(op_ret_type(op, reads))
#         else:
#             # more outputs is unsupported yet
#             assert len(writes) == 1
#             outvar = writes[0]
#
#         reads = [to_pyval(r) for r in reads]
#         writes = [to_pyval(outvar)]
#
#         self._term: WrappedTerm = WrappedTerm(op, reads, writes)
#         self.outvar: Var = outvar
#
#         # libzrm_torch.print_pyterm(self._term)
#
#     def print(self):
#         self._term.print()
#
#     def unwrap(self) -> WrappedTerm:
#         return self._term
#
#     def __str__(self) -> str:
#         return str(self._term)
#
#     def __repr__(self) -> str:
#         return f"Term({self._term})"
#
#     def _op(self, op, arg) -> "Term":
#         if isinstance(arg, (int, float, Var)):
#             term = self.ctx_.term(op, [self.outvar, arg])
#             return term.outvar
#         raise NotImplementedError(
#             f"Unsupported type for operation: {arg} ({type(arg)})"
#         )
#
#     def _r_op(self, op, arg) -> "Term":
#         if isinstance(arg, (int, float, Var)):
#             term = self.ctx_.term(op, [arg, self.outvar])
#             return term.outvar
#         raise NotImplementedError(
#             f"Unsupported type for operation: {arg} ({type(arg)})"
#         )
#
#     def __add__(self, rhs):
#         return self._op("Arith::Add", rhs)
#
#     def __mul__(self, rhs):
#         return self._op("Arith::Mul", rhs)
#
#     def __radd__(self, rhs):
#         return self._r_op("Arith::Add", rhs)
#
#     def __rmul__(self, rhs):
#         return self._r_op("Arith::Mul", rhs)
#
#     def __lt__(self, rhs):
#         return self._op("Cmp::Lt", rhs)
#
#     def __rlt__(self, rhs):
#         return self._r_op("Cmp::Lt", rhs)
#
#     # TODO: other relational operators
#
#     # logical or
#     def __or__(self, rhs: "Term") -> "Term":
#         return self.ctx_.term("Logical::Or", [self.outvar, rhs.outvar]).outvar
#
#     # logical and
#     def __and__(self, rhs: "Term") -> "Term":
#         return self.ctx_.term("Logical::And", [self.outvar, rhs.outvar]).outvar
#
#     # logical negation: ~
#     def __invert__(self) -> "Term":
#         return self.neg()
#
#     # do not convert terms to 0 or 1
#     def __bool__(self):
#         raise NotImplementedError("Cannot convert term to bool")
#
#     def neg(self) -> "Term":
#         return self.ctx_.term("Logical::Neg", [self.outvar]).outvar
#
#     def sum(self) -> "Term":
#         return self.ctx_.term("Arith::Sum", [self.outvar]).outvar
#
#
# class Var(Term):
#     """
#     NOTE: This constructor should be called only by Context.
#     """
#
#     def __init__(self, ctx: "Context", name: str, ty: str, id_=None):
#         super().__init__(ctx, "Var", [])
#         self.name: str = name
#
#         # below I set the attributes from the parent class directly,
#         # I'll burn in hell for that.
#         self.outvar: Var = self
#         # The underlying WrappedTerm
#         if id_ is None:
#             id_ = ctx.get_var_id(name)
#
#         self._type = ty
#         self._term: PyVal = PyVal.sym(id_, ty)
#
#     def ty(self) -> str:
#         return self._type
#
#
# class Assignment:
#     def __init__(self, lhs: "NextVar", rhs: Term):
#         self.var = lhs
#         self.rhs = rhs
#
#     def __str__(self):
#         return f"{self.var} := {self.rhs}"

#
# class NextVar:
#     """
#     A class for primed variables. For these, we want to override some operators
#     differently. Otherwise, they are the same as Var.
#
#     :param: name  The name of **current state** variable.
#     """
#
#     def __init__(self, ctx: "Context", var: Var):
#         assert isinstance(var, Var), var
#         super().__init__(ctx, f"{var.name}'", var.ty())
#         # we store also the current variable name, so that we can easily
#         # get the latched variable from this one
#         self._latched = var
#
#     def get_latched(self) -> Var:
#         return self._latched
#
#     def __eq__(self, rhs: Term) -> Assignment:
#         """
#         Operator == works as assignment for primed variables.
#         """
#         return Assignment(self, rhs)
