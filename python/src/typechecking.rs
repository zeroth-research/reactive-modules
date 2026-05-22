use crate::types::{DType, IType};
use DType::*;

/// get next arguments from an iterator with arguments (a helper fun)
fn next_arg<I, D>(iter: &mut I, op: &str, i: usize) -> Result<DType, String>
where
    I: Iterator<Item = D>,
    D: TryInto<DType>,
{
    if let Some(d) = iter.next() {
        d.try_into()
            .map_err(|_| format!("{op}: arg {i} not compatible with DType"))
    } else {
        Err(format!("{op}: arg {i} expected but got none"))
    }
}

/// translate type's shape into matrix shape, return Err if the shape
/// is not matrix shape
fn shape2(shape: &[usize]) -> Result<(usize, usize), ()> {
    match shape {
        [n] => Ok((1, *n)),
        [m, n] => Ok((*m, *n)),
        _ => Err(()),
    }
}

/// Type-check IType, this function will be called from the `impl Theory for IType`
pub(crate) fn type_check<'a, R, W, D>(itype: &IType, read: R, write: W) -> Result<(), String>
where
    D: TryInto<DType>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
{
    let mut rd = read.into_iter();
    let mut wr = write.into_iter();
    let op_str = itype.to_string();
    let op = op_str.as_str();

    // helper accessor macros
    macro_rules! rd {
        ($i:expr) => {
            next_arg(&mut rd, op, $i)?
        };
    }
    macro_rules! wr {
        ($i:expr) => {
            next_arg(&mut wr, op, $i)?
        };
    }
    macro_rules! no_more_args {
        () => {
            if rd.next().is_some() {
                return Err(format!("{op}: too many read args"));
            }
            if wr.next().is_some() {
                return Err(format!("{op}: too many write args"));
            }
        };
    }

    match itype {
        // -- identity / ite
        IType::Id() => {
            let r = rd!(0);
            let w = wr!(0);
            no_more_args!();
            if r != w {
                return Err(format!("{op}: read type {r} != write type {w}"));
            }
            Ok(())
        }
        IType::Ite() => {
            // cond, true-value, false-value
            let (r0, r1, r2) = (rd!(0), rd!(1), rd!(2));
            let w0 = wr!(0);
            no_more_args!();
            if !r0.is_bool() {
                return Err(format!("{op}: condition must be Bool, got {r0}"));
            };
            if !r0.is_scalar() {
                return Err(format!("{op}: condition must be Bool scalar, got {r0}"));
            }
            if r1 != r2 {
                return Err(format!(
                    "{op}: branches must have same type, got {r1} and {r2}"
                ));
            }
            if w0 != r1 {
                return Err(format!("{op}: output {w0} must match branch type {r1}"));
            }
            Ok(())
        }

        // -- comparison
        IType::Eq() | IType::Neq() | IType::Lt() | IType::Le() | IType::Gt() | IType::Ge() => {
            let (r0, r1) = (rd!(0), rd!(1));
            let w0 = wr!(0);
            no_more_args!();
            if r0 != r1 {
                return Err(format!(
                    "{op}: inputs must have the same type, got {r0} and {r1}"
                ));
            }
            if r0.is_bool() {
                return Err(format!("{op}: inputs cannot be Bool"));
            }
            if !w0.is_bool() {
                return Err(format!("{op}: output must be Bool, got {w0}"));
            }
            if w0.shape() != r0.shape() {
                return Err(format!(
                    "{op}: output Bool shape {:?} != input shape {:?}",
                    w0.shape(),
                    r0.shape()
                ));
            }
            Ok(())
        }

        // -- logical
        IType::And() | IType::Or() | IType::Xor() | IType::Xnor() | IType::Implies() => {
            let (r0, r1) = (rd!(0), rd!(1));
            let w0 = wr!(0);
            no_more_args!();
            if !r0.is_bool() || !r1.is_bool() || !w0.is_bool() {
                return Err(format!("{op}: all operands must be Bool"));
            }
            if r0 != r1 {
                return Err(format!(
                    "{op}: inputs must have the same shape, got {r0} and {r1}"
                ));
            }
            if w0 != r0 {
                return Err(format!("{op}: output {w0} must match input type {r0}"));
            }
            Ok(())
        }
        IType::Not() => {
            let r = rd!(0);
            let w = wr!(0);
            no_more_args!();
            if !r.is_bool() || !w.is_bool() {
                return Err(format!("{op}: operands must be Bool"));
            }
            if r != w {
                return Err(format!(
                    "{op}: input {r} and output {w} must have the same shape"
                ));
            }
            Ok(())
        }

        // -- arithmetic (binary)
        IType::Add() | IType::Sub() | IType::Mul() | IType::Div() | IType::Mod() => {
            let (r0, r1) = (rd!(0), rd!(1));
            let w0 = wr!(0);
            no_more_args!();
            if r0.is_bool() {
                return Err(format!("{op}: inputs must be numeric, not Bool"));
            }
            if r0 != r1 {
                return Err(format!(
                    "{op}: inputs must have the same type, got {r0} and {r1}"
                ));
            }
            if w0 != r0 {
                return Err(format!("{op}: output {w0} must match input type {r0}"));
            }
            Ok(())
        }

        // -- arithmetic (unary)
        IType::Neg() | IType::Abs() => {
            let r = rd!(0);
            let w = wr!(0);
            no_more_args!();
            if r.is_bool() {
                return Err(format!("{op}: input must be numeric, not Bool"));
            }
            if r != w {
                return Err(format!(
                    "{op}: input {r} and output {w} must have the same type"
                ));
            }
            Ok(())
        }

        // -- matrix multiply
        IType::MatMul() => {
            let (r0, r1) = (rd!(0), rd!(1));
            let w0 = wr!(0);
            no_more_args!();
            if !r0.is_same_kind(&r1) || !r0.is_same_kind(&w0) {
                return Err(format!("{op}: all operands must have the same base type"));
            }
            if r0.is_bool() || r0.is_bv() {
                return Err(format!("{op}: operands must be numeric (Int/Float/Real)"));
            }
            let r0_shape = r0.shape();
            let r1_shape = r1.shape();
            let w0_shape = w0.shape();
            let (m, k) = shape2(&r0_shape).map_err(|_| format!("{op}: lhs must be 1-D or 2-D"))?;
            let (k2, n) = if r1_shape.len() == 1 {
                (r1_shape[0], 1)
            } else {
                shape2(&r1_shape).map_err(|_| format!("{op}: rhs must be 1-D or 2-D"))?
            };
            let (om, on) = if w0_shape.len() == 1 {
                (w0_shape[0], 1)
            } else {
                shape2(&w0_shape).map_err(|_| format!("{op}: output must be 1-D or 2-D"))?
            };
            if k != k2 {
                return Err(format!("{op}: inner dimensions mismatch: {k} != {k2}"));
            }
            if m != om || n != on {
                return Err(format!("{op}: output shape ({om},{on}) must be ({m},{n})"));
            }
            Ok(())
        }

        // -- transcendental
        IType::Sin() | IType::Cos() => {
            let r = rd!(0);
            let w = wr!(0);
            no_more_args!();
            if !r.is_real() {
                return Err(format!("{op}: input must be Real, got {r}"));
            }
            if r != w {
                return Err(format!(
                    "{op}: input {r} and output {w} must have the same type"
                ));
            }
            Ok(())
        }

        // -- NN
        IType::ReLU() | IType::Tanh() => {
            let r = rd!(0);
            let w = wr!(0);
            no_more_args!();
            if !r.is_float() && !r.is_real() {
                return Err(format!("{op}: input must be Float or Real, got {r}"));
            }
            if r != w {
                return Err(format!(
                    "{op}: input {r} and output {w} must have the same type"
                ));
            }
            Ok(())
        }
        // -- Linear layer
        IType::Linear() => {
            let (r0, r1, r2) = (rd!(0), rd!(1), rd!(2)); // input, weight, bias
            let w0 = wr!(0);
            no_more_args!();
            if !r0.is_float() || !r1.is_float() || !r2.is_float() || !w0.is_float() {
                return Err(format!("{op}: all operands must be Float"));
            }
            let (inp_s, wgt_s, bias_s, out_s) = (r0.shape(), r1.shape(), r2.shape(), w0.shape());
            let (batch, in_f) =
                shape2(&inp_s).map_err(|_| format!("{op}: input must be 1-D or 2-D"))?;
            let (out_f, wgt_in) =
                shape2(&wgt_s).map_err(|_| format!("{op}: weight must be 1-D or 2-D"))?;
            let (_, bias_cols) =
                shape2(&bias_s).map_err(|_| format!("{op}: bias must be 1-D or 2-D"))?;
            let (out_batch, out_cols) =
                shape2(&out_s).map_err(|_| format!("{op}: output must be 1-D or 2-D"))?;
            if in_f != wgt_in {
                return Err(format!(
                    "{op}: input features {in_f} != weight cols {wgt_in}"
                ));
            }
            if out_f != bias_cols {
                return Err(format!(
                    "{op}: weight rows {out_f} != bias cols {bias_cols}"
                ));
            }
            if batch != out_batch || out_f != out_cols {
                return Err(format!(
                    "{op}: output must be Float({batch}, {out_f}), got Float({out_batch}, {out_cols})"
                ));
            }
            Ok(())
        }

        // -- Tensor ops
        IType::Argmax() => {
            let _r = rd!(0);
            let w0 = wr!(0);
            no_more_args!();
            if !w0.is_int() || !w0.is_scalar() {
                return Err(format!("{op}: output must be Int scalar, got {w0}"));
            }
            Ok(())
        }
        IType::TensorSum() | IType::TensorMean() | IType::TensorMax() => {
            let r0 = rd!(0);
            let w0 = wr!(0);
            no_more_args!();
            if !r0.is_same_kind(&w0) || r0.is_bool() || r0.is_bv() || !w0.is_scalar() {
                return Err(format!(
                    "{op}: output must be scalar of same base type as input, got {w0}"
                ));
            }
            Ok(())
        }
        IType::TensorGet() => {
            let (r0, r1) = (rd!(0), rd!(1)); // tensor, index
            let w0 = wr!(0);
            no_more_args!();
            if !r1.is_int() || !r1.is_scalar() {
                return Err(format!("{op}: index must be Int scalar, got {r1}"));
            }
            if !r0.is_same_kind(&w0) || r0.is_bool() || r0.is_bv() || !w0.is_scalar() {
                return Err(format!(
                    "{op}: output must be scalar of same base type as tensor input, got {w0}"
                ));
            }
            Ok(())
        }
        IType::TensorSet() => {
            let (r0, r1, r2) = (rd!(0), rd!(1), rd!(2)); // tensor, index, value
            let w0 = wr!(0);
            no_more_args!();
            if !r1.is_int() || !r1.is_scalar() {
                return Err(format!("{op}: index must be Int scalar, got {r1}"));
            }
            if !r0.is_same_kind(&r2) || r0.is_bool() || r0.is_bv() || !r2.is_scalar() {
                return Err(format!(
                    "{op}: value must be scalar of same base type as tensor, got {r2}"
                ));
            }
            if r0 != w0 {
                return Err(format!(
                    "{op}: output {w0} must match input tensor type {r0}"
                ));
            }
            Ok(())
        }

        // -- BV / casting
        IType::BitSelect(h, l) => {
            let r0 = rd!(0);
            let w0 = wr!(0);
            no_more_args!();
            if !r0.is_bv() {
                return Err(format!("{op}: input must be BV, got {r0}"));
            }
            let out_bw = h - l + 1;
            if w0 != BV(out_bw) {
                return Err(format!("{op}: output must be BV({out_bw}), got {w0}"));
            }
            Ok(())
        }
        IType::Extend(w) => {
            let r0 = rd!(0);
            let w0 = wr!(0);
            no_more_args!();
            let expected = match r0 {
                BV(bw) => {
                    if bw > *w {
                        return Err(format!("{op}: input width {bw} > target {w}"));
                    }
                    BV(*w)
                }
                _ => return Err(format!("{op}: input must be BV, got {r0}")),
            };
            if w0 != expected {
                return Err(format!("{op}: output must be {expected}, got {w0}"));
            }
            Ok(())
        }
        IType::BVToBool() => {
            let r0 = rd!(0);
            let w0 = wr!(0);
            no_more_args!();
            if !r0.is_bv() {
                return Err(format!("{op}: input must be BV, got {r0}"));
            }
            if !w0.is_bool() || !w0.is_scalar() {
                return Err(format!("{op}: output must be Bool scalar, got {w0}"));
            }
            Ok(())
        }
        IType::BoolToBV() => {
            let r0 = rd!(0);
            let w0 = wr!(0);
            no_more_args!();
            if !r0.is_bool() {
                return Err(format!("{op}: input must be Bool, got {r0}"));
            }
            if w0 != BV(1) {
                return Err(format!("{op}: output must be BV(1), got {w0}"));
            }
            Ok(())
        }

        // -- Constants
        IType::Tensor(t) => {
            let w0 = wr!(0);
            no_more_args!();
            let sz = t.tensor.size();
            let expected_shape: Vec<usize> = sz.iter().map(|&x| x as usize).collect();
            match w0 {
                Float(s) | Int(s) | Real(s) | Bool(s) => {
                    if s != expected_shape {
                        return Err(format!(
                            "{op}: write shape {s:?} doesn't match tensor shape {expected_shape:?}"
                        ));
                    }
                }
                _ => return Err(format!("{op}: write type must be Float, Int, or Real")),
            }
            Ok(())
        }
        IType::ConstBool(_) => {
            let w0 = wr!(0);
            no_more_args!();
            if !w0.is_bool() || !w0.is_scalar() {
                return Err(format!("{op}: output must be Bool scalar, got {w0}"));
            }
            Ok(())
        }
        IType::ConstInt(_) => {
            let w0 = wr!(0);
            no_more_args!();
            if !(w0.is_int() && w0.is_scalar()) && !(w0.is_real() && w0.is_scalar()) && !w0.is_bv()
            {
                return Err(format!(
                    "{op}: output must be Int/Real scalar or BV, got {w0}"
                ));
            }
            Ok(())
        }

        // -- Uninterpreted
        IType::Uninterpreted(_) => Ok(()),
    }
}
