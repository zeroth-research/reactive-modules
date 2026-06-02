"""Stateless Term factories, one per theory."""
import torch
from .zrth import Wire, Term, IType as _IType, DType
from . import Bool, Float, Int


def _wire(t) -> Wire:
    """Get the output wire from a Term or a bare Wire."""
    if isinstance(t, Term):
        return t.write[0]
    return t


def _dtype(t) -> DType:
    return _wire(t).dtype


class TermBuilder:
    """Abstract base. Subclasses set _ns = IType.LIA / IType.LRA / IType.BV."""
    _ns = None

    # --- helpers (public for callers in visitor) ---
    def wire(self, t) -> Wire:
        return _wire(t)

    def dtype(self, t) -> DType:
        return _dtype(t)

    # --- theory-independent ops ---

    def id_(self, src, output_wire=None) -> Term:
        w = output_wire or Wire(_dtype(src))
        return Term(self._ns.Id(), [w], [_wire(src)])

    def ite(self, cond, a, b, output_wire=None) -> Term:
        w = output_wire or Wire(_dtype(a))
        return Term(self._ns.Ite(), [w], [_wire(cond), _wire(a), _wire(b)])

    def not_(self, a) -> Term:
        return Term(self._ns.Not(), [Wire(Bool(1, 1))], [_wire(a)])

    def const_bool(self, value: bool, output_wire=None) -> Term:
        w = output_wire or Wire(Bool(1, 1))
        return Term(self._ns.ConstBool(value), [w])

    def transpose(self, a) -> Term:
        w = _wire(a)
        s = w.dtype.shape
        return Term(self._ns.Transpose(), [Wire(w.dtype.reshape([s[1], s[0]]))], [w])

    def relu(self, a) -> Term:
        return Term(self._ns.ReLU(), [Wire(_dtype(a))], [_wire(a)])

    def argmax(self, a) -> Term:
        w = _wire(a)
        # Output has same dtype family as input, reshaped to [1, 1]
        return Term(self._ns.Argmax(), [Wire(w.dtype.reshape([1, 1]))], [w])

    def uninterpreted(self, name: str, dtype) -> Term:
        return Term(self._ns.Uninterpreted(name), [Wire(dtype)])

    def linear(self, x, weight, bias) -> list:
        """Emit [Transpose-in, Linear, Transpose-out]. Last write[0] is the output wire."""
        x_wire = _wire(x)
        shape = x_wire.dtype.shape  # [1, in_features] row-major
        in_features = shape[-1]
        out_features = weight.shape[0]

        input_col_wire = Wire(x_wire.dtype.reshape([in_features, 1]))
        t_in = Term(self._ns.Transpose(), [input_col_wire], [x_wire])

        linear_out_wire = Wire(Float(out_features, 1))
        t_lin = Term(self._ns.Linear(weight, bias), [linear_out_wire], [input_col_wire])

        out_wire = Wire(Float(out_features))
        t_out = Term(self._ns.Transpose(), [out_wire], [linear_out_wire])

        return [t_in, t_lin, t_out]

    def linear_symbolic(self, x, out_features: int):
        """Linear with unknown (symbolic) weights. Returns (term, weight_wire, bias_wire)."""
        x_wire = _wire(x)
        in_features = x_wire.dtype.shape[-1]
        weight_wire = Wire(Float(out_features, in_features))
        bias_wire = Wire(Float(out_features, 1))
        out_wire = Wire(Float(out_features))
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
    def sin(self, a) -> Term:
        raise NotImplementedError
    def cos(self, a) -> Term:
        raise NotImplementedError
    def tanh(self, a) -> Term:
        raise NotImplementedError


class LRATermBuilder(TermBuilder):
    _ns = _IType.LRA

    def add(self, a, b) -> Term:
        return Term(_IType.LRA.Add(), [Wire(_dtype(a))], [_wire(a), _wire(b)])
    def sub(self, a, b) -> Term:
        return Term(_IType.LRA.Sub(), [Wire(_dtype(a))], [_wire(a), _wire(b)])
    def lt(self, a, b) -> Term:
        return Term(_IType.LRA.Lt(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def le(self, a, b) -> Term:
        return Term(_IType.LRA.Le(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def gt(self, a, b) -> Term:
        return Term(_IType.LRA.Gt(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def ge(self, a, b) -> Term:
        return Term(_IType.LRA.Ge(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def eq(self, a, b) -> Term:
        return Term(_IType.LRA.Eq(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def ne(self, a, b) -> Term:
        return Term(_IType.LRA.Ne(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def sin(self, a) -> Term:
        return Term(_IType.LRA.Sin(), [Wire(_dtype(a))], [_wire(a)])
    def cos(self, a) -> Term:
        return Term(_IType.LRA.Cos(), [Wire(_dtype(a))], [_wire(a)])
    def tanh(self, a) -> Term:
        return Term(_IType.LRA.Tanh(), [Wire(_dtype(a))], [_wire(a)])

    def mul(self, a, b) -> Term:
        result = self._try_mul_as_linear(a, b)
        if result is not None:
            return result
        raise ValueError("LRA does not support mul with non-constant operands")

    def _try_mul_as_linear(self, a, b):
        for const_t, var_t in [(a, b), (b, a)]:
            if not isinstance(const_t, Term):
                continue
            if const_t.itype.op_name != "ConstReal":
                continue
            data = const_t.itype.const_data
            if data.numel() != 1:
                continue
            var_wire = _wire(var_t)
            s = var_wire.dtype.shape
            if len(s) < 2 or s[-1] != 1:
                continue
            scalar = data.item()
            A = torch.tensor([[scalar]], dtype=torch.float32)
            b_bias = torch.zeros(1, 1, dtype=torch.float32)
            return Term(_IType.LRA.Linear(A, b_bias), [Wire(var_wire.dtype)], [var_wire])
        return None

    def const(self, tensor, output_wire=None) -> Term:
        if tensor.dtype == torch.bool:
            w = output_wire or Wire(Bool(1, 1))
            return Term(_IType.LRA.ConstBool(tensor), [w])
        shape = list(tensor.size())
        if len(shape) == 0: shape = [1, 1]
        elif len(shape) == 1: shape = [1, shape[0]]
        w = output_wire or Wire(Float(*shape))
        return Term(_IType.LRA.ConstReal(tensor), [w])


class LIATermBuilder(TermBuilder):
    _ns = _IType.LIA

    def add(self, a, b) -> Term:
        return Term(_IType.LIA.Add(), [Wire(_dtype(a))], [_wire(a), _wire(b)])
    def sub(self, a, b) -> Term:
        return Term(_IType.LIA.Sub(), [Wire(_dtype(a))], [_wire(a), _wire(b)])
    def lt(self, a, b) -> Term:
        return Term(_IType.LIA.Lt(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def le(self, a, b) -> Term:
        return Term(_IType.LIA.Le(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def gt(self, a, b) -> Term:
        return Term(_IType.LIA.Gt(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def ge(self, a, b) -> Term:
        return Term(_IType.LIA.Ge(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def eq(self, a, b) -> Term:
        return Term(_IType.LIA.Eq(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def ne(self, a, b) -> Term:
        return Term(_IType.LIA.Ne(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def sin(self, a) -> Term:
        raise ValueError("LIA does not support sin")
    def cos(self, a) -> Term:
        raise ValueError("LIA does not support cos")
    def tanh(self, a) -> Term:
        raise ValueError("LIA does not support tanh")

    def mul(self, a, b) -> Term:
        result = self._try_mul_as_linear(a, b)
        if result is not None:
            return result
        raise ValueError("LIA does not support mul with non-constant operands")

    def _try_mul_as_linear(self, a, b):
        for const_t, var_t in [(a, b), (b, a)]:
            if not isinstance(const_t, Term):
                continue
            if const_t.itype.op_name != "ConstInt":
                continue
            data = const_t.itype.const_data
            if data.numel() != 1:
                continue
            var_wire = _wire(var_t)
            s = var_wire.dtype.shape
            if len(s) < 2 or s[-1] != 1:
                continue
            scalar = data.item()
            A = torch.tensor([[scalar]], dtype=torch.int64)
            b_bias = torch.zeros(1, 1, dtype=torch.int64)
            return Term(_IType.LIA.Linear(A, b_bias), [Wire(var_wire.dtype)], [var_wire])
        return None

    def const(self, tensor, output_wire=None) -> Term:
        if tensor.dtype == torch.bool:
            w = output_wire or Wire(Bool(1, 1))
            return Term(_IType.LIA.ConstBool(tensor), [w])
        shape = list(tensor.size())
        if len(shape) == 0: shape = [1, 1]
        elif len(shape) == 1: shape = [1, shape[0]]
        w = output_wire or Wire(Int(*shape))
        return Term(_IType.LIA.ConstInt(tensor), [w])


class BVTermBuilder(TermBuilder):
    _ns = _IType.BV

    def add(self, a, b) -> Term:
        return Term(_IType.BV.Add(), [Wire(_dtype(a))], [_wire(a), _wire(b)])
    def sub(self, a, b) -> Term:
        return Term(_IType.BV.Sub(), [Wire(_dtype(a))], [_wire(a), _wire(b)])
    def mul(self, a, b) -> Term:
        return Term(_IType.BV.Mul(), [Wire(_dtype(a))], [_wire(a), _wire(b)])
    def lt(self, a, b) -> Term:
        return Term(_IType.BV.Lt(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def le(self, a, b) -> Term:
        return Term(_IType.BV.Le(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def gt(self, a, b) -> Term:
        return Term(_IType.BV.Gt(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def ge(self, a, b) -> Term:
        return Term(_IType.BV.Ge(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def eq(self, a, b) -> Term:
        return Term(_IType.BV.Eq(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def ne(self, a, b) -> Term:
        return Term(_IType.BV.Ne(), [Wire(Bool(1, 1))], [_wire(a), _wire(b)])
    def sin(self, a) -> Term:
        raise ValueError("BV does not support sin")
    def cos(self, a) -> Term:
        raise ValueError("BV does not support cos")
    def tanh(self, a) -> Term:
        raise ValueError("BV does not support tanh")
    def neg(self, a) -> Term:
        return Term(_IType.BV.Neg(), [Wire(_dtype(a))], [_wire(a)])
    def abs_(self, a) -> Term:
        return Term(_IType.BV.Abs(), [Wire(_dtype(a))], [_wire(a)])
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
    def const(self, tensor, output_wire=None) -> Term:
        if tensor.dtype == torch.bool:
            w = output_wire or Wire(DType.BV(1))
            return Term(_IType.BV.Const(tensor), [w])
        out = output_wire or Wire(DType.BV(32))
        return Term(_IType.BV.Const(tensor), [out])
