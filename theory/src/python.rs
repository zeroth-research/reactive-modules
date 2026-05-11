use crate::{CmpOp, FlowOp, Theory, read_nxt, write_nxt};
use crate::{bool, bv, float, int, real};
use std::fmt;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Type {
    Bool(bool::Bool),
    Int(int::Int),
    Float(float::Float),
    Real(real::Real),
    BV32(bv::BV),
}

impl Type {
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Type::Bool(b) => b.shape(),
            Type::Int(i) => i.shape(),
            Type::Float(f) => f.shape(),
            Type::Real(r) => r.shape(),
            Type::BV32(bv) => bv.shape(),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Bool(t) => t.fmt(f),
            Type::Int(t) => t.fmt(f),
            Type::Float(t) => t.fmt(f),
            Type::Real(t) => t.fmt(f),
            Type::BV32(t) => match t {
                crate::bv::BV::U(bw, r, c) => write!(f, "UWord({bw}, {r}, {c})"),
                crate::bv::BV::S(bw, r, c) => write!(f, "SWord({bw}, {r}, {c})"),
            },
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum NNOp {
    ReLU,
    Tanh,
    // Linear layer: output = input @ weight + bias
    // Reads: [input, weight, bias], Writes: [output]
    Linear,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TensorOp {
    Get,
    Set,
    Sum,
    Mean,
    Max,
    // index of maximal value in the **flattened** tensor
    Argmax,
}

impl fmt::Display for NNOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NNOp::ReLU => write!(f, "ReLU"),
            NNOp::Tanh => write!(f, "Tanh"),
            NNOp::Linear => write!(f, "Linear"),
        }
    }
}

impl fmt::Display for TensorOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorOp::Get => write!(f, "Get"),
            TensorOp::Set => write!(f, "Set"),
            TensorOp::Sum => write!(f, "Sum"),
            TensorOp::Mean => write!(f, "Mean"),
            TensorOp::Max => write!(f, "Max"),
            TensorOp::Argmax => write!(f, "Argmax"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum IType {
    // Arithmetic operations
    Bool(bool::BoolOp),
    Int(int::ArithInt),
    Float(float::ArithFloat),
    Real(real::ArithReal),
    Cmp(CmpOp),
    Flow(FlowOp),

    // activation functions
    NN(NNOp),

    // Tensor operations
    Tensor(TensorOp),

    // Bit-vector operations
    BV(bv::BVTheory),

    // Casting
    BVToBool,
    BVToWord1,
    ToUnsigned,
    ToSigned,

    // Bit-vector bit selection: extract bits [high..low] from a BV
    BitSelect(usize, usize),

    // Bit-vector zero/sign extension
    Extend(usize),

    // Symbol referring to uninterpreted constants or functions,
    // whose signature is known in the context, i.e., the current theory
    Uninterpreted(String),
}

impl fmt::Display for IType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IType::Bool(x) => x.fmt(f),
            IType::Int(x) => x.fmt(f),
            IType::Float(x) => x.fmt(f),
            IType::Real(x) => x.fmt(f),
            IType::Cmp(x) => x.fmt(f),
            IType::Flow(x) => x.fmt(f),
            IType::NN(x) => x.fmt(f),
            IType::Tensor(x) => x.fmt(f),
            IType::BV(t) => write!(f, "BV({t:?})"),
            IType::BVToBool => write!(f, "BVToBool"),
            IType::BVToWord1 => write!(f, "BVToWord1"),
            IType::ToUnsigned => write!(f, "ToUnsigned"),
            IType::ToSigned => write!(f, "ToSigned"),
            IType::BitSelect(h, l) => write!(f, "BitSelect({h}, {l})"),
            IType::Extend(w) => write!(f, "Extend({w})"),
            IType::Uninterpreted(t) => write!(f, "{t}"),
        }
    }
}

// ============================================================================
// TryFrom conversions: &'a Type -> &'a SubType  (needed for sub-theory delegation)
// ============================================================================

impl<'a> TryFrom<&'a Type> for &'a bool::Bool {
    type Error = ();
    fn try_from(t: &'a Type) -> Result<Self, ()> {
        match t {
            Type::Bool(b) => Ok(b),
            _ => Err(()),
        }
    }
}

impl<'a> TryFrom<&'a Type> for &'a int::Int {
    type Error = ();
    fn try_from(t: &'a Type) -> Result<Self, ()> {
        match t {
            Type::Int(i) => Ok(i),
            _ => Err(()),
        }
    }
}

impl<'a> TryFrom<&'a Type> for &'a float::Float {
    type Error = ();
    fn try_from(t: &'a Type) -> Result<Self, ()> {
        match t {
            Type::Float(f) => Ok(f),
            _ => Err(()),
        }
    }
}

impl<'a> TryFrom<&'a Type> for &'a real::Real {
    type Error = ();
    fn try_from(t: &'a Type) -> Result<Self, ()> {
        match t {
            Type::Real(r) => Ok(r),
            _ => Err(()),
        }
    }
}

impl<'a> TryFrom<&'a Type> for &'a bv::BV {
    type Error = ();
    fn try_from(t: &'a Type) -> Result<Self, ()> {
        match t {
            Type::BV32(bv) => Ok(bv),
            _ => Err(()),
        }
    }
}

// "Unwrap" impls: make Result<&'a SubType, E> usable as D in sub-theory type_check calls.
// Bool and Int are already covered by identical impls in lia.rs.

impl<'a, E> TryFrom<Result<&'a float::Float, E>> for &'a float::Float {
    type Error = E;
    fn try_from(v: Result<&'a float::Float, E>) -> Result<Self, E> {
        v
    }
}

impl<'a, E> TryFrom<Result<&'a real::Real, E>> for &'a real::Real {
    type Error = E;
    fn try_from(v: Result<&'a real::Real, E>) -> Result<Self, E> {
        v
    }
}

impl<'a, E> TryFrom<Result<&'a bv::BV, E>> for &'a bv::BV {
    type Error = E;
    fn try_from(v: Result<&'a bv::BV, E>) -> Result<Self, E> {
        v
    }
}

fn iter_try_into_subtype<'a, D, S, I>(
    iter: I,
    msg: &'static str,
) -> impl Iterator<Item = Result<&'a S, &'static str>>
where
    D: TryInto<&'a Type>,
    &'a Type: TryInto<&'a S, Error = ()>,
    I: IntoIterator<Item = D>,
    S: 'a,
{
    iter.into_iter().map(move |d| {
        d.try_into()
            .map_err(|_| ())
            .and_then(|t: &'a Type| t.try_into())
            .map_err(|_| msg)
    })
}

// ============================================================================
// Theory impl
// ============================================================================

impl Theory for IType {
    type DType = Type;

    fn type_check<'a, R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<&'a Self::DType>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
        Self::DType: 'a,
    {
        match self {
            IType::Bool(op) => op.type_check(
                iter_try_into_subtype(read, "expected Bool type"),
                iter_try_into_subtype(write, "expected Bool type"),
            ),
            IType::Int(op) => op.type_check(
                iter_try_into_subtype(read, "expected Int type"),
                iter_try_into_subtype(write, "expected Int type"),
            ),
            IType::Float(op) => op.type_check(
                iter_try_into_subtype(read, "expected Float type"),
                iter_try_into_subtype(write, "expected Float type"),
            ),
            IType::Real(op) => op.type_check(
                iter_try_into_subtype(read, "expected Real type"),
                iter_try_into_subtype(write, "expected Real type"),
            ),
            IType::BV(op) => op.type_check(
                iter_try_into_subtype(read, "expected BV type"),
                iter_try_into_subtype(write, "expected BV type"),
            ),
            IType::Cmp(op) => op.type_check(read, write),
            IType::Flow(op) => op.type_check(read, write),
            IType::NN(op) => check_nn(op, read, write),
            IType::Tensor(op) => check_tensor(op, read, write),
            IType::BVToBool => check_bv_to_bool(read, write),
            IType::BVToWord1 => check_bv_to_word1(read, write),
            IType::ToUnsigned => check_to_unsigned(read, write),
            IType::ToSigned => check_to_signed(read, write),
            IType::BitSelect(high, low) => check_bit_select(*high, *low, read, write),
            IType::Extend(width) => check_extend(*width, read, write),
            IType::Uninterpreted(_) => Ok(()),
        }
    }
}

// ============================================================================
// Type-check helpers
// ============================================================================

fn check_nn<'a, R, W, D>(op: &NNOp, read: R, write: W) -> Result<(), String>
where
    D: TryInto<&'a Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
    Type: 'a,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        NNOp::ReLU | NNOp::Tanh => {
            let (r0, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                return Err(format!("{:?}: must read exactly one value", op));
            };
            let (w0, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                return Err(format!("{:?}: must write exactly one value", op));
            };
            if matches!(r0, Type::Bool(_) | Type::Int(_)) {
                return Err(format!(
                    "{:?}: input must be Float or Real, got {:?}",
                    op, r0
                ));
            }
            if r0 != w0 {
                return Err(format!(
                    "{:?}: input and output must have the same type",
                    op
                ));
            }
            Ok(())
        }
        NNOp::Linear => {
            // output = input @ weight.T + bias
            // input:  Float(batch, in_f)
            // weight: Float(out_f, in_f)
            // bias:   Float(1,     out_f)  [or Float(batch, out_f) for matrix bias]
            // output: Float(batch, out_f)
            let r0 = read_nxt(&mut read, 0)?; // input
            let r1 = read_nxt(&mut read, 1)?; // weight
            let r2 = read_nxt(&mut read, 2)?; // bias
            if read.next().is_some() {
                return Err("Linear: must read exactly three values".into());
            }
            let (w0, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                return Err("Linear: must write exactly one value".into());
            };
            let (
                Type::Float(float::Float(batch, in_f)),
                Type::Float(float::Float(out_f, wgt_in)),
                Type::Float(float::Float(_, bias_cols)),
                Type::Float(float::Float(out_batch, out_cols)),
            ) = (r0, r1, r2, w0)
            else {
                return Err("Linear: all operands must be Float".into());
            };
            if in_f != wgt_in {
                return Err(format!(
                    "Linear: input features {} != weight cols {}",
                    in_f, wgt_in
                ));
            }
            if out_f != bias_cols {
                return Err(format!(
                    "Linear: weight rows {} != bias cols {}",
                    out_f, bias_cols
                ));
            }
            if batch != out_batch || out_f != out_cols {
                return Err(format!(
                    "Linear: output must be Float({}, {}), got Float({}, {})",
                    batch, out_f, out_batch, out_cols
                ));
            }
            Ok(())
        }
    }
}

fn check_tensor<'a, R, W, D>(op: &TensorOp, read: R, write: W) -> Result<(), String>
where
    D: TryInto<&'a Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
    Type: 'a,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        TensorOp::Argmax => {
            let (_r0, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                return Err("Argmax: must read exactly one value".into());
            };
            let (w0, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                return Err("Argmax: must write exactly one value".into());
            };
            if *w0 != Type::Int(int::Int(1, 1)) {
                return Err(format!("Argmax: output must be Int(1, 1), got {:?}", w0));
            }
            Ok(())
        }
        TensorOp::Sum | TensorOp::Mean | TensorOp::Max => {
            let (r0, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                return Err(format!("{:?}: must read exactly one value", op));
            };
            let (w0, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                return Err(format!("{:?}: must write exactly one value", op));
            };
            // output is a scalar of the same base type
            let expected = match r0 {
                Type::Float(_) => Type::Float(float::Float(1, 1)),
                Type::Int(_) => Type::Int(int::Int(1, 1)),
                Type::Real(_) => Type::Real(real::Real(1, 1)),
                _ => return Err(format!("{:?}: input must be Float, Int, or Real", op)),
            };
            if *w0 != expected {
                return Err(format!(
                    "{:?}: output must be {:?}, got {:?}",
                    op, expected, w0
                ));
            }
            Ok(())
        }
        TensorOp::Get => {
            // Get: read[tensor, index], write[scalar element]
            let r0 = read_nxt(&mut read, 0)?; // tensor
            let r1 = read_nxt(&mut read, 1)?; // index
            if read.next().is_some() {
                return Err("Get: must read exactly two values".into());
            }
            let (w0, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                return Err("Get: must write exactly one value".into());
            };
            if *r1 != Type::Int(int::Int(1, 1)) {
                return Err(format!("Get: index must be Int(1, 1), got {:?}", r1));
            }
            let expected = match r0 {
                Type::Float(_) => Type::Float(float::Float(1, 1)),
                Type::Int(_) => Type::Int(int::Int(1, 1)),
                Type::Real(_) => Type::Real(real::Real(1, 1)),
                _ => {
                    return Err(format!(
                        "Get: tensor must be Float, Int, or Real, got {:?}",
                        r0
                    ));
                }
            };
            if *w0 != expected {
                return Err(format!("Get: output must be {:?}, got {:?}", expected, w0));
            }
            Ok(())
        }
        TensorOp::Set => {
            // Set: read[tensor, index, value], write[tensor]
            let r0 = read_nxt(&mut read, 0)?; // tensor
            let r1 = read_nxt(&mut read, 1)?; // index
            let r2 = read_nxt(&mut read, 2)?; // value
            if read.next().is_some() {
                return Err("Set: must read exactly three values".into());
            }
            let (w0, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                return Err("Set: must write exactly one value".into());
            };
            if *r1 != Type::Int(int::Int(1, 1)) {
                return Err(format!("Set: index must be Int(1, 1), got {:?}", r1));
            }
            let scalar = match r0 {
                Type::Float(_) => Type::Float(float::Float(1, 1)),
                Type::Int(_) => Type::Int(int::Int(1, 1)),
                Type::Real(_) => Type::Real(real::Real(1, 1)),
                _ => {
                    return Err(format!(
                        "Set: tensor must be Float, Int, or Real, got {:?}",
                        r0
                    ));
                }
            };
            if *r2 != scalar {
                return Err(format!("Set: value must be {:?}, got {:?}", scalar, r2));
            }
            if r0 != w0 {
                return Err(format!(
                    "Set: output must have the same type as input tensor, got {:?}",
                    w0
                ));
            }
            Ok(())
        }
    }
}

fn check_bv_to_bool<'a, R, W, D>(read: R, write: W) -> Result<(), String>
where
    D: TryInto<&'a Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
    Type: 'a,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let (r0, None) = (read_nxt(&mut read, 0)?, read.next()) else {
        return Err("BVToBool: must read exactly one value".into());
    };
    let (w0, None) = (write_nxt(&mut write, 0)?, write.next()) else {
        return Err("BVToBool: must write exactly one value".into());
    };
    let Type::BV32(bv) = r0 else {
        return Err(format!("BVToBool: input must be BV, got {:?}", r0));
    };
    let (rows, cols) = bv.shape();
    if *w0 != Type::Bool(bool::Bool(rows, cols)) {
        return Err(format!(
            "BVToBool: output must be Bool({}, {}), got {:?}",
            rows, cols, w0
        ));
    }
    Ok(())
}

fn check_bv_to_word1<'a, R, W, D>(read: R, write: W) -> Result<(), String>
where
    D: TryInto<&'a Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
    Type: 'a,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let (r0, None) = (read_nxt(&mut read, 0)?, read.next()) else {
        return Err("BVToWord1: must read exactly one value".into());
    };
    let (w0, None) = (write_nxt(&mut write, 0)?, write.next()) else {
        return Err("BVToWord1: must write exactly one value".into());
    };
    let Type::BV32(bv) = r0 else {
        return Err(format!("BVToWord1: input must be BV, got {:?}", r0));
    };
    let (rows, cols) = bv.shape();
    let expected = Type::BV32(bv::BV::U(1, rows, cols));
    if *w0 != expected {
        return Err(format!(
            "BVToWord1: output must be {:?}, got {:?}",
            expected, w0
        ));
    }
    Ok(())
}

fn check_to_unsigned<'a, R, W, D>(read: R, write: W) -> Result<(), String>
where
    D: TryInto<&'a Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
    Type: 'a,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let (r0, None) = (read_nxt(&mut read, 0)?, read.next()) else {
        return Err("ToUnsigned: must read exactly one value".into());
    };
    let (w0, None) = (write_nxt(&mut write, 0)?, write.next()) else {
        return Err("ToUnsigned: must write exactly one value".into());
    };
    let Type::BV32(bv) = r0 else {
        return Err(format!("ToUnsigned: input must be BV, got {:?}", r0));
    };
    let (bw, rows, cols) = (bv.bw(), bv.shape().0, bv.shape().1);
    let expected = Type::BV32(bv::BV::U(bw, rows, cols));
    if *w0 != expected {
        return Err(format!(
            "ToUnsigned: output must be {:?}, got {:?}",
            expected, w0
        ));
    }
    Ok(())
}

fn check_to_signed<'a, R, W, D>(read: R, write: W) -> Result<(), String>
where
    D: TryInto<&'a Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
    Type: 'a,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let (r0, None) = (read_nxt(&mut read, 0)?, read.next()) else {
        return Err("ToSigned: must read exactly one value".into());
    };
    let (w0, None) = (write_nxt(&mut write, 0)?, write.next()) else {
        return Err("ToSigned: must write exactly one value".into());
    };
    let Type::BV32(bv) = r0 else {
        return Err(format!("ToSigned: input must be BV, got {:?}", r0));
    };
    let (bw, rows, cols) = (bv.bw(), bv.shape().0, bv.shape().1);
    let expected = Type::BV32(bv::BV::S(bw, rows, cols));
    if *w0 != expected {
        return Err(format!(
            "ToSigned: output must be {:?}, got {:?}",
            expected, w0
        ));
    }
    Ok(())
}

fn check_bit_select<'a, R, W, D>(high: usize, low: usize, read: R, write: W) -> Result<(), String>
where
    D: TryInto<&'a Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
    Type: 'a,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let (r0, None) = (read_nxt(&mut read, 0)?, read.next()) else {
        return Err("BitSelect: must read exactly one value".into());
    };
    let (w0, None) = (write_nxt(&mut write, 0)?, write.next()) else {
        return Err("BitSelect: must write exactly one value".into());
    };
    let Type::BV32(bv) = r0 else {
        return Err(format!("BitSelect: input must be BV, got {:?}", r0));
    };
    if high < low {
        return Err(format!("BitSelect: high ({}) < low ({})", high, low));
    }
    if high >= bv.bw() {
        return Err(format!(
            "BitSelect: high ({}) >= input bitwidth ({})",
            high,
            bv.bw()
        ));
    }
    let out_bw = high - low + 1;
    let (rows, cols) = bv.shape();
    let expected = Type::BV32(bv::BV::U(out_bw, rows, cols));
    if *w0 != expected {
        return Err(format!(
            "BitSelect: output must be {:?}, got {:?}",
            expected, w0
        ));
    }
    Ok(())
}

fn check_extend<'a, R, W, D>(width: usize, read: R, write: W) -> Result<(), String>
where
    D: TryInto<&'a Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
    Type: 'a,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let (r0, None) = (read_nxt(&mut read, 0)?, read.next()) else {
        return Err("Extend: must read exactly one value".into());
    };
    let (w0, None) = (write_nxt(&mut write, 0)?, write.next()) else {
        return Err("Extend: must write exactly one value".into());
    };
    let Type::BV32(bv) = r0 else {
        return Err(format!("Extend: input must be BV, got {:?}", r0));
    };
    let (rows, cols) = bv.shape();
    let out_bw = bv.bw() + width;
    let expected = match bv {
        bv::BV::U(_, _, _) => Type::BV32(bv::BV::U(out_bw, rows, cols)),
        bv::BV::S(_, _, _) => Type::BV32(bv::BV::S(out_bw, rows, cols)),
    };
    if *w0 != expected {
        return Err(format!(
            "Extend: output must be {:?}, got {:?}",
            expected, w0
        ));
    }
    Ok(())
}
