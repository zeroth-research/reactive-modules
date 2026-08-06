"""Stateless Term factories, one per theory.

Temporary scaffolding kept to move fast during the theory_pyo3 migration; the
end-state is to call the theory ops directly, so this should not grow.

Per the theory_pyo3 design, Sorts and ops expose no accessor methods — shape,
bitwidth and const payloads are obtained via `match`/unpacking (see the module
helpers below).
"""

import torch
from .zrth import Wire, Term, Sort, LRA, LIA, BV


def _wire(t) -> Wire:
    if isinstance(t, Term):
        return t.write[0]
    return t


def _dtype(t) -> Sort:
    return _wire(t).dtype


def _normalize_shape(shape: list) -> list:
    if len(shape) == 0:
        return [1, 1]
    if len(shape) == 1:
        return [1, shape[0]]
    return shape


# --- Sort / op access via `match` (no methods on Sort or the ops) ------------


def _shape(sort: Sort) -> list:
    match sort:
        case Sort.Bool(s) | Sort.Int(s) | Sort.Real(s):
            return list(s)
        case Sort.BitVec(_, s):
            return list(s)
    raise TypeError(f"sort has no shape: {sort}")


def _with_shape(sort: Sort, shape: list) -> Sort:
    match sort:
        case Sort.Bool(_):
            return Sort.Bool(shape)
        case Sort.Int(_):
            return Sort.Int(shape)
        case Sort.Real(_):
            return Sort.Real(shape)
        case Sort.BitVec(bw, _):
            return Sort.BitVec(bw, shape)
    raise TypeError(f"unknown sort: {sort}")


def _bv_width(sort: Sort) -> int:
    match sort:
        case Sort.BitVec(bw, _):
            return bw
    raise TypeError(f"not a BitVec sort: {sort}")


def _bool_sort(sort: Sort) -> Sort:
    return Sort.Bool(_shape(sort))


def _const_tensor(op):
    """The constant tensor carried by a const op, or None (ops expose no getters)."""
    match op:
        case LRA.Const(t) | LIA.Const(t) | BV.Const(t):
            return t
    return None


class TheoryError(Exception):
    """An operation cannot be expressed in the selected theory."""


class NonLinearError(TheoryError):
    """A non-linear op (e.g. variable * variable) in a linear theory (LIA/LRA),
    which only support multiplication by a constant."""

    def __init__(self, theory: str, detail: str = "multiplication of two non-constant operands"):
        self.theory = theory
        super().__init__(f"{theory} does not support {detail}")


class TermBuilder:
    """Abstract base. Subclasses set `_ns = LRA / LIA / BV`."""

    _ns = None

    # --- low-level helpers ---

    def _binary_op(self, op_fn, output_sort: Sort, a, b) -> Term:
        return Term(op_fn(), [Wire(output_sort)], [_wire(a), _wire(b)])

    def _unary_op(self, op_fn, a) -> Term:
        return Term(op_fn(), [Wire(_dtype(a))], [_wire(a)])

    # --- theory-independent ops ---

    def id_(self, src, output_wire=None) -> Term:
        assert self._ns
        w = output_wire or Wire(_dtype(src))
        return Term(self._ns.Id(), [w], [_wire(src)])

    def ite(self, cond, a, b, output_wire=None) -> Term:
        assert self._ns
        w = output_wire or Wire(_dtype(a))
        return Term(self._ns.Ite(), [w], [_wire(cond), _wire(a), _wire(b)])

    def not_(self, a) -> Term:
        assert self._ns
        return Term(self._ns.Not(), [Wire(_bool_sort(_dtype(a)))], [_wire(a)])

    def and_(self, a, b) -> Term:
        return self._binary_op(self._ns.And, _bool_sort(_dtype(a)), a, b)

    def or_(self, a, b) -> Term:
        return self._binary_op(self._ns.Or, _bool_sort(_dtype(a)), a, b)

    def xor_(self, a, b) -> Term:
        return self._binary_op(self._ns.Xor, _bool_sort(_dtype(a)), a, b)

    def const_bool(self, value: bool, output_wire=None) -> Term:
        # Route through the theory's const() so each theory represents bools its
        # own way (LIA/LRA Const on a Bool wire; BV Const as BitVec<1>).
        return self.const(
            torch.tensor([[bool(value)]], dtype=torch.bool), output_wire=output_wire
        )

    def transpose(self, a) -> Term:
        assert self._ns
        w = _wire(a)
        s = _shape(w.dtype)
        return Term(self._ns.Transpose(), [Wire(_with_shape(w.dtype, [s[1], s[0]]))], [w])

    def relu(self, a) -> Term:
        assert self._ns
        return self._unary_op(self._ns.ReLU, a)

    def argmax(self, a) -> Term:
        assert self._ns
        w = _wire(a)
        # Output keeps the input's sort family, reshaped to [1, 1].
        return Term(self._ns.Argmax(), [Wire(_with_shape(w.dtype, [1, 1]))], [w])

    def python_type_to_dtype(self, python_type: type, shape: list):
        raise NotImplementedError

    def const_for_value(self, value, output_wire=None) -> "Term":
        raise NotImplementedError

    def _try_mul_as_linear(self, a, b, const_cls, scalar_dtype):
        assert self._ns
        for const_t, var_t in [(a, b), (b, a)]:
            if not isinstance(const_t, Term):
                continue
            if not isinstance(const_t.itype, const_cls):
                continue
            data = _const_tensor(const_t.itype)
            if data is None or data.numel() != 1 or data.dtype == torch.bool:
                continue
            var_wire = _wire(var_t)
            s = _shape(var_wire.dtype)
            if len(s) < 2 or s[-1] != 1:
                continue
            scalar = data.item()
            A = torch.tensor([[scalar]], dtype=scalar_dtype)
            b_bias = torch.zeros(1, 1, dtype=scalar_dtype)
            return Term(self._ns.Linear(A, b_bias), [Wire(var_wire.dtype)], [var_wire])
        return None

    def uninterpreted(self, name: str, sort: Sort) -> Term:
        assert self._ns
        return Term.constant(self._ns.Uninterpreted(name), [Wire(sort)])

    def _numeric_wire(self, shape: list) -> Wire:
        """Wire of this theory's numeric sort with the given shape."""
        raise NotImplementedError

    def linear(self, x, weight, bias) -> list:
        """Emit [Transpose-in, Linear, Transpose-out]. Last write[0] is the output wire."""
        assert self._ns
        x_wire = _wire(x)
        shape = _shape(x_wire.dtype)  # [1, in_features] row-major
        in_features = shape[-1]
        out_features = weight.shape[0]

        input_col_wire = Wire(_with_shape(x_wire.dtype, [in_features, 1]))
        t_in = Term(self._ns.Transpose(), [input_col_wire], [x_wire])

        linear_out_wire = self._numeric_wire([out_features, 1])
        t_lin = Term(self._ns.Linear(weight, bias), [linear_out_wire], [input_col_wire])

        out_wire = self._numeric_wire([1, out_features])
        t_out = Term(self._ns.Transpose(), [out_wire], [linear_out_wire])

        return [t_in, t_lin, t_out]

    def linear_symbolic(self, x, out_features: int):
        # theory_pyo3's Linear carries its weight/bias as *constant* tensors in the
        # op, so a Linear with symbolic (wire) weights is no longer expressible.
        # Revisit when porting the analyzer/torch NN path (Step 4).
        raise NotImplementedError(
            "symbolic Linear (wire weights) is not expressible: theory Linear takes constant A, B"
        )

    # --- theory-specific ops (abstract) ---

    def add(self, a, b) -> Term:
        raise NotImplementedError

    def sub(self, a, b) -> Term:
        raise NotImplementedError

    def mul(self, a, b) -> Term:
        raise NotImplementedError

    def lt(self, a, b) -> Term:
        raise NotImplementedError

    def le(self, a, b) -> Term:
        raise NotImplementedError

    def gt(self, a, b) -> Term:
        raise NotImplementedError

    def ge(self, a, b) -> Term:
        raise NotImplementedError

    def eq(self, a, b) -> Term:
        raise NotImplementedError

    def ne(self, a, b) -> Term:
        raise NotImplementedError

    def const(self, tensor, output_wire=None) -> Term:
        raise NotImplementedError

    def div(self, a, b) -> Term:
        raise ValueError(f"{type(self).__name__} does not support div")

    def matmul(self, a, b) -> Term:
        raise NotImplementedError

    # --- deferred ops (no current theory provides them) ----------------------
    # These stay routed through the builder so the analyzer (and the tutorials)
    # remain theory-agnostic. When a suitable theory is added, only the relevant
    # TermBuilder subclass (and the theory crate) override these -- no analyzer
    # or notebook changes are needed.

    def _theory(self) -> str:
        return self._ns.__name__ if self._ns is not None else "the selected theory"

    def sin(self, a) -> Term:
        raise NonLinearError(self._theory(), "sin (non-linear; needs a non-linear theory)")

    def cos(self, a) -> Term:
        raise NonLinearError(self._theory(), "cos (non-linear; needs a non-linear theory)")

    def tanh(self, a) -> Term:
        raise NonLinearError(self._theory(), "tanh (non-linear; needs a non-linear theory)")

    def pow(self, base, exponent) -> Term:
        raise NonLinearError(self._theory(), "** / pow (non-linear; needs a non-linear theory)")

    def stack(self, elements) -> Term:
        raise TheoryError(
            f"{self._theory()} has no Stack op "
            "(list-valued state needs a theory that provides Stack)"
        )

    def tensor_get(self, base, index) -> Term:
        raise TheoryError(
            f"{self._theory()} has no TensorGet op "
            "(indexing needs a theory that provides TensorGet)"
        )


class LRATermBuilder(TermBuilder):
    _ns = LRA

    def _numeric_wire(self, shape: list) -> Wire:
        return Wire(Sort.Real(shape))

    def python_type_to_dtype(self, python_type: type, shape: list):
        if python_type is bool:
            return Sort.Bool(shape)
        return Sort.Real(shape)

    def add(self, a, b) -> Term:
        return self._binary_op(LRA.Add, _dtype(a), a, b)

    def sub(self, a, b) -> Term:
        return self._binary_op(LRA.Sub, _dtype(a), a, b)

    def lt(self, a, b) -> Term:
        return self._binary_op(LRA.Lt, _bool_sort(_dtype(a)), a, b)

    def le(self, a, b) -> Term:
        return self._binary_op(LRA.Le, _bool_sort(_dtype(a)), a, b)

    def gt(self, a, b) -> Term:
        return self._binary_op(LRA.Gt, _bool_sort(_dtype(a)), a, b)

    def ge(self, a, b) -> Term:
        return self._binary_op(LRA.Ge, _bool_sort(_dtype(a)), a, b)

    def eq(self, a, b) -> Term:
        return self._binary_op(LRA.Eq, _bool_sort(_dtype(a)), a, b)

    def ne(self, a, b) -> Term:
        return self._binary_op(LRA.Ne, _bool_sort(_dtype(a)), a, b)

    # LRA is linear real arithmetic: no transcendentals / generic division.
    # (sin/cos/tanh/pow/stack/tensor_get inherit the base's deferred-op errors.)
    def mul(self, a, b) -> Term:
        result = self._try_mul_as_linear(a, b, LRA.Const, torch.float32)
        if result is not None:
            return result
        raise NonLinearError("LRA")

    def matmul(self, a, b) -> Term:
        a_wire, b_wire = _wire(a), _wire(b)
        out_shape = [_shape(a_wire.dtype)[0], _shape(b_wire.dtype)[1]]
        if isinstance(a, Term) and isinstance(a.itype, LRA.Const):
            no_bias = torch.empty(0, 0)
            return Term(
                LRA.Linear(_const_tensor(a.itype), no_bias),
                [Wire(Sort.Real(out_shape))],
                [b_wire],
            )
        raise RuntimeError(
            "LRA matmul requires a constant left operand; use builder.linear() instead"
        )

    def const_for_value(self, value, output_wire=None) -> Term:
        if isinstance(value, bool):
            return self.const_bool(value, output_wire=output_wire)
        return self.const(
            torch.tensor([float(value)], dtype=torch.float32), output_wire=output_wire
        )

    def const(self, tensor, output_wire=None) -> Term:
        shape = _normalize_shape(list(tensor.size()))
        tensor = tensor.reshape(shape)  # theory const ops require a 2-D initializer
        if tensor.dtype == torch.bool:
            w = output_wire or Wire(Sort.Bool(shape))
            return Term.constant(LRA.Const(tensor), [w])
        w = output_wire or Wire(Sort.Real(shape))
        return Term.constant(LRA.Const(tensor), [w])


class LIATermBuilder(TermBuilder):
    _ns = LIA

    def _numeric_wire(self, shape: list) -> Wire:
        return Wire(Sort.Int(shape))

    def python_type_to_dtype(self, python_type: type, shape: list):
        if python_type is bool:
            return Sort.Bool(shape)
        return Sort.Int(shape)

    def add(self, a, b) -> Term:
        return self._binary_op(LIA.Add, _dtype(a), a, b)

    def sub(self, a, b) -> Term:
        return self._binary_op(LIA.Sub, _dtype(a), a, b)

    def lt(self, a, b) -> Term:
        return self._binary_op(LIA.Lt, _bool_sort(_dtype(a)), a, b)

    def le(self, a, b) -> Term:
        return self._binary_op(LIA.Le, _bool_sort(_dtype(a)), a, b)

    def gt(self, a, b) -> Term:
        return self._binary_op(LIA.Gt, _bool_sort(_dtype(a)), a, b)

    def ge(self, a, b) -> Term:
        return self._binary_op(LIA.Ge, _bool_sort(_dtype(a)), a, b)

    def eq(self, a, b) -> Term:
        return self._binary_op(LIA.Eq, _bool_sort(_dtype(a)), a, b)

    def ne(self, a, b) -> Term:
        return self._binary_op(LIA.Ne, _bool_sort(_dtype(a)), a, b)

    def mul(self, a, b) -> Term:
        result = self._try_mul_as_linear(a, b, LIA.Const, torch.int64)
        if result is not None:
            return result
        raise NonLinearError("LIA")

    def matmul(self, a, b) -> Term:
        a_wire, b_wire = _wire(a), _wire(b)
        out_shape = [_shape(a_wire.dtype)[0], _shape(b_wire.dtype)[1]]
        if isinstance(a, Term) and isinstance(a.itype, LIA.Const):
            no_bias = torch.empty(0, 0)
            return Term(
                LIA.Linear(_const_tensor(a.itype), no_bias),
                [Wire(Sort.Int(out_shape))],
                [b_wire],
            )
        raise RuntimeError(
            "LIA matmul requires a constant left operand; use builder.linear() instead"
        )

    def const_for_value(self, value, output_wire=None) -> Term:
        if isinstance(value, bool):
            return self.const_bool(value, output_wire=output_wire)
        tensor = torch.tensor([int(value)], dtype=torch.int64)
        return self.const(tensor, output_wire=output_wire)

    def const(self, tensor, output_wire=None) -> Term:
        shape = _normalize_shape(list(tensor.size()))
        tensor = tensor.reshape(shape)  # theory const ops require a 2-D initializer
        if tensor.dtype == torch.bool:
            w = output_wire or Wire(Sort.Bool(shape))
            return Term.constant(LIA.Const(tensor), [w])
        w = output_wire or Wire(Sort.Int(shape))
        return Term.constant(LIA.Const(tensor), [w])


class BVTermBuilder(TermBuilder):
    _ns = BV

    # FIXME: make 32 a parameter
    def _numeric_wire(self, shape: list) -> Wire:
        return Wire(Sort.BitVec(32, shape))

    def python_type_to_dtype(self, python_type: type, shape: list):
        if python_type is bool:
            return Sort.Bool(shape)
        return Sort.BitVec(32, shape)

    def add(self, a, b) -> Term:
        return self._binary_op(BV.Add, _dtype(a), a, b)

    def sub(self, a, b) -> Term:
        return self._binary_op(BV.Sub, _dtype(a), a, b)

    def mul(self, a, b) -> Term:
        return self._binary_op(BV.Mul, _dtype(a), a, b)

    # In the BV (SMT-LIB) model there is no Bool: bitwise ops preserve the
    # operand's BV width, and comparisons yield a 1-bit BV.
    def _bv1(self, a) -> Sort:
        return Sort.BitVec(1, _shape(_dtype(a)))

    def and_(self, a, b) -> Term:
        return self._binary_op(BV.And, _dtype(a), a, b)

    def or_(self, a, b) -> Term:
        return self._binary_op(BV.Or, _dtype(a), a, b)

    def xor_(self, a, b) -> Term:
        return self._binary_op(BV.Xor, _dtype(a), a, b)

    def not_(self, a) -> Term:
        return Term(BV.Not(), [Wire(_dtype(a))], [_wire(a)])

    # Comparisons default to unsigned (Sort carries no signedness yet; the smv
    # parser tracks signedness and picks U*/S* directly). TODO: signed path.
    def lt(self, a, b) -> Term:
        return self._binary_op(BV.ULt, self._bv1(a), a, b)

    def le(self, a, b) -> Term:
        return self._binary_op(BV.ULe, self._bv1(a), a, b)

    def gt(self, a, b) -> Term:
        return self._binary_op(BV.UGt, self._bv1(a), a, b)

    def ge(self, a, b) -> Term:
        return self._binary_op(BV.UGe, self._bv1(a), a, b)

    def eq(self, a, b) -> Term:
        return self._binary_op(BV.Eq, self._bv1(a), a, b)

    def ne(self, a, b) -> Term:
        return self._binary_op(BV.Ne, self._bv1(a), a, b)

    def neg(self, a) -> Term:
        return self._unary_op(BV.Neg, a)

    def abs_(self, a) -> Term:
        return self._unary_op(BV.Abs, a)

    def matmul(self, a, b) -> Term:
        a_w, b_w = _wire(a), _wire(b)
        out_shape = [_shape(a_w.dtype)[0], _shape(b_w.dtype)[1]]
        out = Wire(Sort.BitVec(_bv_width(a_w.dtype), out_shape))
        return Term(BV.MatMul(), [out], [a_w, b_w])

    def bit_select(self, a, high: int, low: int) -> Term:
        bw = high - low + 1
        out = Wire(Sort.BitVec(bw, [1, 1]))
        return Term(BV.BitSelect(high=high, low=low), [out], [_wire(a)])

    def extend(self, a, extra: int) -> Term:
        a_w = _wire(a)
        new_bw = _bv_width(a_w.dtype) + extra
        out = Wire(Sort.BitVec(new_bw, [1, 1]))
        return Term(BV.Extend(extra=extra), [out], [a_w])

    def const_for_value(self, value, output_wire=None) -> Term:
        if isinstance(value, bool):
            tensor = torch.tensor([value], dtype=torch.bool)
        else:
            tensor = torch.tensor([int(value)], dtype=torch.int64)
        return self.const(tensor, output_wire=output_wire)

    def const(self, tensor, output_wire=None) -> Term:
        shape = _normalize_shape(list(tensor.size()))
        tensor = tensor.reshape(shape)  # theory const ops require a 2-D initializer
        if tensor.dtype == torch.bool:
            w = output_wire or Wire(Sort.BitVec(1, shape))
            return Term.constant(BV.Const(tensor), [w])
        out = output_wire or Wire(Sort.BitVec(32, shape))
        return Term.constant(BV.Const(tensor), [out])


def builder_for(theory=None) -> "TermBuilder":
    """Create the appropriate TermBuilder for the given theory.

    theory: LRA, LIA, BV, or None (defaults to LRA).
    """
    if theory is None or theory is LRA:
        return LRATermBuilder()
    if theory is LIA:
        return LIATermBuilder()
    if theory is BV:
        return BVTermBuilder()
    raise ValueError(f"Unknown theory: {theory!r}. Use LRA, LIA, or BV.")
