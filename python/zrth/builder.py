"""Stateless Term factories, one per theory."""

import torch
from .zrth import Wire, Term, IType as _IType, DType
from . import Bool, Float, Int


def _wire(t) -> Wire:
    if isinstance(t, Term):
        return t.write[0]
    return t


def _dtype(t) -> DType:
    return _wire(t).dtype


def _normalize_shape(shape: list) -> list:
    if len(shape) == 0:
        return [1, 1]
    if len(shape) == 1:
        return [1, shape[0]]
    return shape


class TermBuilder:
    """Abstract base. Subclasses set _ns = IType.LIA / IType.LRA / IType.BV."""

    _ns = None

    # --- low-level helpers ---

    def _binary_op(self, op_fn, output_dtype: DType, a, b) -> Term:
        return Term(op_fn(), [Wire(output_dtype)], [_wire(a), _wire(b)])

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
        return Term(self._ns.Not(), [Wire(Bool(*_dtype(a).shape))], [_wire(a)])

    def and_(self, a, b) -> Term:
        return self._binary_op(self._ns.And, Bool(*_dtype(a).shape), a, b)

    def or_(self, a, b) -> Term:
        return self._binary_op(self._ns.Or, Bool(*_dtype(a).shape), a, b)

    def xor_(self, a, b) -> Term:
        return self._binary_op(self._ns.Xor, Bool(*_dtype(a).shape), a, b)

    def const_bool(self, value: bool, output_wire=None) -> Term:
        assert self._ns
        w = output_wire or Wire(Bool(1, 1))
        return Term(self._ns.ConstBool(value), [w])

    def transpose(self, a) -> Term:
        assert self._ns
        w = _wire(a)
        s = w.dtype.shape
        return Term(self._ns.Transpose(), [Wire(w.dtype.reshape([s[1], s[0]]))], [w])

    def relu(self, a) -> Term:
        assert self._ns
        return self._unary_op(self._ns.ReLU, a)

    def argmax(self, a) -> Term:
        assert self._ns
        w = _wire(a)
        # Output has same dtype family as input, reshaped to [1, 1]
        return Term(self._ns.Argmax(), [Wire(w.dtype.reshape([1, 1]))], [w])

    def python_type_to_dtype(self, python_type: type, shape: list):
        raise NotImplementedError

    def const_for_value(self, value, output_wire=None) -> "Term":
        raise NotImplementedError

    def _try_mul_as_linear(self, a, b, const_op_name: str, scalar_dtype):
        assert self._ns
        for const_t, var_t in [(a, b), (b, a)]:
            if not isinstance(const_t, Term):
                continue
            if const_t.itype.op_name != const_op_name:
                continue
            data = const_t.itype.const_data
            if data.numel() != 1:
                continue
            var_wire = _wire(var_t)
            s = var_wire.dtype.shape
            if len(s) < 2 or s[-1] != 1:
                continue
            scalar = data.item()
            A = torch.tensor([[scalar]], dtype=scalar_dtype)
            b_bias = torch.zeros(1, 1, dtype=scalar_dtype)
            return Term(self._ns.Linear(A, b_bias), [Wire(var_wire.dtype)], [var_wire])
        return None

    def uninterpreted(self, name: str, dtype) -> Term:
        assert self._ns
        return Term(self._ns.Uninterpreted(name), [Wire(dtype)])

    def _numeric_wire(self, shape: list) -> Wire:
        """Wire of this theory's numeric type with the given shape."""
        raise NotImplementedError

    def linear(self, x, weight, bias) -> list:
        """Emit [Transpose-in, Linear, Transpose-out]. Last write[0] is the output wire."""
        assert self._ns
        x_wire = _wire(x)
        shape = x_wire.dtype.shape  # [1, in_features] row-major
        in_features = shape[-1]
        out_features = weight.shape[0]

        input_col_wire = Wire(x_wire.dtype.reshape([in_features, 1]))
        t_in = Term(self._ns.Transpose(), [input_col_wire], [x_wire])

        linear_out_wire = self._numeric_wire([out_features, 1])
        t_lin = Term(self._ns.Linear(weight, bias), [linear_out_wire], [input_col_wire])

        out_wire = self._numeric_wire([out_features])
        t_out = Term(self._ns.Transpose(), [out_wire], [linear_out_wire])

        return [t_in, t_lin, t_out]

    def linear_symbolic(self, x, out_features: int):
        """Linear with unknown (symbolic) weights. Returns (term, weight_wire, bias_wire)."""
        assert self._ns
        x_wire = _wire(x)
        in_features = x_wire.dtype.shape[-1]
        weight_wire = self._numeric_wire([out_features, in_features])
        bias_wire = self._numeric_wire([out_features, 1])
        out_wire = self._numeric_wire([out_features])
        term = Term(self._ns.Linear(), [out_wire], [x_wire, weight_wire, bias_wire])
        return term, weight_wire, bias_wire

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

    def sin(self, a) -> Term:
        raise NotImplementedError

    def cos(self, a) -> Term:
        raise NotImplementedError

    def tanh(self, a) -> Term:
        raise NotImplementedError


class LRATermBuilder(TermBuilder):
    _ns = _IType.LRA

    def _numeric_wire(self, shape: list) -> Wire:
        return Wire(DType.Float(shape))

    def python_type_to_dtype(self, python_type: type, shape: list):
        if python_type is bool:
            return DType.Bool(shape)
        return DType.Float(shape)

    def add(self, a, b) -> Term:
        return self._binary_op(_IType.LRA.Add, _dtype(a), a, b)

    def sub(self, a, b) -> Term:
        return self._binary_op(_IType.LRA.Sub, _dtype(a), a, b)

    def lt(self, a, b) -> Term:
        return self._binary_op(_IType.LRA.Lt, Bool(*_dtype(a).shape), a, b)

    def le(self, a, b) -> Term:
        return self._binary_op(_IType.LRA.Le, Bool(*_dtype(a).shape), a, b)

    def gt(self, a, b) -> Term:
        return self._binary_op(_IType.LRA.Gt, Bool(*_dtype(a).shape), a, b)

    def ge(self, a, b) -> Term:
        return self._binary_op(_IType.LRA.Ge, Bool(*_dtype(a).shape), a, b)

    def eq(self, a, b) -> Term:
        return self._binary_op(_IType.LRA.Eq, Bool(*_dtype(a).shape), a, b)

    def ne(self, a, b) -> Term:
        return self._binary_op(_IType.LRA.Ne, Bool(*_dtype(a).shape), a, b)

    def sin(self, a) -> Term:
        return self._unary_op(_IType.LRA.Sin, a)

    def cos(self, a) -> Term:
        return self._unary_op(_IType.LRA.Cos, a)

    def tanh(self, a) -> Term:
        return self._unary_op(_IType.LRA.Tanh, a)

    def mul(self, a, b) -> Term:
        result = self._try_mul_as_linear(a, b, "ConstReal", torch.float32)
        if result is not None:
            return result
        raise ValueError("LRA does not support mul with non-constant operands")

    def div(self, a, b) -> Term:
        return self._binary_op(_IType.LRA.Div, _dtype(a), a, b)

    def matmul(self, a, b) -> Term:
        a_wire, b_wire = _wire(a), _wire(b)
        out_shape = [a_wire.dtype.shape[0], b_wire.dtype.shape[1]]
        if isinstance(a, Term) and a.itype.op_name == "ConstReal":
            no_bias = torch.empty(0, 0)
            return Term(_IType.LRA.Linear(a.itype.const_data, no_bias), [Wire(DType.Float(out_shape))], [b_wire])
        raise RuntimeError("LRA matmul requires a constant left operand; use builder.linear() instead")

    def const_for_value(self, value, output_wire=None) -> Term:
        if isinstance(value, bool):
            return self.const_bool(value, output_wire=output_wire)
        return self.const(
            torch.tensor([float(value)], dtype=torch.float32), output_wire=output_wire
        )

    def const(self, tensor, output_wire=None) -> Term:
        if tensor.dtype == torch.bool:
            shape = _normalize_shape(list(tensor.size()))
            w = output_wire or Wire(Bool(*shape))
            return Term(_IType.LRA.ConstBool(tensor), [w])
        shape = _normalize_shape(list(tensor.size()))
        w = output_wire or Wire(Float(*shape))
        return Term(_IType.LRA.ConstReal(tensor), [w])


class LIATermBuilder(TermBuilder):
    _ns = _IType.LIA

    def _numeric_wire(self, shape: list) -> Wire:
        return Wire(DType.Int(shape))

    def python_type_to_dtype(self, python_type: type, shape: list):
        if python_type is bool:
            return DType.Bool(shape)
        return DType.Int(shape)

    def add(self, a, b) -> Term:
        return self._binary_op(_IType.LIA.Add, _dtype(a), a, b)

    def sub(self, a, b) -> Term:
        return self._binary_op(_IType.LIA.Sub, _dtype(a), a, b)

    def lt(self, a, b) -> Term:
        return self._binary_op(_IType.LIA.Lt, Bool(*_dtype(a).shape), a, b)

    def le(self, a, b) -> Term:
        return self._binary_op(_IType.LIA.Le, Bool(*_dtype(a).shape), a, b)

    def gt(self, a, b) -> Term:
        return self._binary_op(_IType.LIA.Gt, Bool(*_dtype(a).shape), a, b)

    def ge(self, a, b) -> Term:
        return self._binary_op(_IType.LIA.Ge, Bool(*_dtype(a).shape), a, b)

    def eq(self, a, b) -> Term:
        return self._binary_op(_IType.LIA.Eq, Bool(*_dtype(a).shape), a, b)

    def ne(self, a, b) -> Term:
        return self._binary_op(_IType.LIA.Ne, Bool(*_dtype(a).shape), a, b)

    def sin(self, a) -> Term:
        raise ValueError("LIA does not support sin")

    def cos(self, a) -> Term:
        raise ValueError("LIA does not support cos")

    def tanh(self, a) -> Term:
        raise ValueError("LIA does not support tanh")

    def mul(self, a, b) -> Term:
        result = self._try_mul_as_linear(a, b, "ConstInt", torch.int64)
        if result is not None:
            return result
        raise ValueError("LIA does not support mul with non-constant operands")

    def matmul(self, a, b) -> Term:
        a_wire, b_wire = _wire(a), _wire(b)
        out_shape = [a_wire.dtype.shape[0], b_wire.dtype.shape[1]]
        if isinstance(a, Term) and a.itype.op_name == "ConstInt":
            no_bias = torch.empty(0, 0)
            return Term(_IType.LIA.Linear(a.itype.const_data, no_bias), [Wire(DType.Int(out_shape))], [b_wire])
        raise RuntimeError("LIA matmul requires a constant left operand; use builder.linear() instead")

    def const_for_value(self, value, output_wire=None) -> Term:
        if isinstance(value, bool):
            return self.const_bool(value, output_wire=output_wire)
        tensor = torch.tensor([int(value)], dtype=torch.int64)
        return self.const(tensor, output_wire=output_wire)

    def const(self, tensor, output_wire=None) -> Term:
        if tensor.dtype == torch.bool:
            shape = _normalize_shape(list(tensor.size()))
            w = output_wire or Wire(Bool(*shape))
            return Term(_IType.LIA.ConstBool(tensor), [w])
        shape = _normalize_shape(list(tensor.size()))
        w = output_wire or Wire(Int(*shape))
        return Term(_IType.LIA.ConstInt(tensor), [w])


class BVTermBuilder(TermBuilder):
    _ns = _IType.BV

    def _numeric_wire(self, shape: list) -> Wire:
        return Wire(DType.BV(32).reshape(shape))

    def python_type_to_dtype(self, python_type: type, shape: list):
        if python_type is bool:
            return DType.Bool(shape)
        # FIXME: make 32 a parameter
        return DType.BV(32).reshape(shape)

    def add(self, a, b) -> Term:
        return self._binary_op(_IType.BV.Add, _dtype(a), a, b)

    def sub(self, a, b) -> Term:
        return self._binary_op(_IType.BV.Sub, _dtype(a), a, b)

    def mul(self, a, b) -> Term:
        return self._binary_op(_IType.BV.Mul, _dtype(a), a, b)

    # TODO: BV comparisons are signed/unsigned per-op in the theory, but DType.BV
    # carries no signedness yet, so we default to the *unsigned* variants. Once
    # signedness is represented Python-side, pick U*/S* accordingly.
    def lt(self, a, b) -> Term:
        return self._binary_op(_IType.BV.ULt, Bool(*_dtype(a).shape), a, b)

    def le(self, a, b) -> Term:
        return self._binary_op(_IType.BV.ULe, Bool(*_dtype(a).shape), a, b)

    def gt(self, a, b) -> Term:
        return self._binary_op(_IType.BV.UGt, Bool(*_dtype(a).shape), a, b)

    def ge(self, a, b) -> Term:
        return self._binary_op(_IType.BV.UGe, Bool(*_dtype(a).shape), a, b)

    def eq(self, a, b) -> Term:
        return self._binary_op(_IType.BV.Eq, Bool(*_dtype(a).shape), a, b)

    def ne(self, a, b) -> Term:
        return self._binary_op(_IType.BV.Ne, Bool(*_dtype(a).shape), a, b)

    def sin(self, a) -> Term:
        raise ValueError("BV does not support sin")

    def cos(self, a) -> Term:
        raise ValueError("BV does not support cos")

    def tanh(self, a) -> Term:
        raise ValueError("BV does not support tanh")

    def neg(self, a) -> Term:
        return self._unary_op(_IType.BV.Neg, a)

    def abs_(self, a) -> Term:
        return self._unary_op(_IType.BV.Abs, a)

    def matmul(self, a, b) -> Term:
        a_w, b_w = _wire(a), _wire(b)
        out_shape = [a_w.dtype.shape[0], b_w.dtype.shape[1]]
        out = Wire(DType.BV(a_w.dtype.bv_bitwidth()).reshape(out_shape))
        return Term(_IType.BV.MatMul(), [out], [a_w, b_w])

    def bit_select(self, a, high: int, low: int) -> Term:
        bw = high - low + 1
        out = Wire(DType.BV(bw))
        return Term(_IType.BV.BitSelect(high, low), [out], [_wire(a)])

    def extend(self, a, extra: int) -> Term:
        a_w = _wire(a)
        new_bw = a_w.dtype.bv_bitwidth() + extra
        out = Wire(DType.BV(new_bw))
        return Term(_IType.BV.Extend(extra), [out], [a_w])

    def const_for_value(self, value, output_wire=None) -> Term:
        if isinstance(value, bool):
            tensor = torch.tensor([value], dtype=torch.bool)
        else:
            tensor = torch.tensor([int(value)], dtype=torch.int64)
        return self.const(tensor, output_wire=output_wire)

    def const(self, tensor, output_wire=None) -> Term:
        if tensor.dtype == torch.bool:
            w = output_wire or Wire(DType.BV(1))
            return Term(_IType.BV.Const(tensor), [w])
        out = output_wire or Wire(DType.BV(32))
        return Term(_IType.BV.Const(tensor), [out])


def builder_for(theory=None) -> "TermBuilder":
    """Create the appropriate TermBuilder for the given theory.

    theory: IType.LRA, IType.LIA, IType.BV, or None (defaults to LRA)
    """
    if theory is None or theory is _IType.LRA:
        return LRATermBuilder()
    if theory is _IType.LIA:
        return LIATermBuilder()
    if theory is _IType.BV:
        return BVTermBuilder()
    raise ValueError(
        f"Unknown theory: {theory!r}. Use IType.LRA, IType.LIA, or IType.BV."
    )
