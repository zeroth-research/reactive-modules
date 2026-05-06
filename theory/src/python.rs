use crate::Theory;
use crate::{bool, bv, float, int, real};
use std::fmt;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Type {
    // at this moment, we keep the DType flat and encode the type
    // of elements in the names
    Bool(bool::Bool),
    Int(int::Int),
    Float(float::Float),
    Real(real::Real),
    BV32(bv::BV),
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

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum CmpOp {
    Le,
    Lt,
    Ge,
    Gt,
    Eq,
    Ne,
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

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CmpOp::Eq => write!(f, "Eq"),
            CmpOp::Ne => write!(f, "Neq"),
            CmpOp::Lt => write!(f, "Lt"),
            CmpOp::Le => write!(f, "Le"),
            CmpOp::Gt => write!(f, "Gt"),
            CmpOp::Ge => write!(f, "Ge"),
        }
    }
}

#[derive(Clone, PartialEq, Debug, Eq)]
pub enum FlowOp {
    Ite,
    Id, // this could probably be in the top-level enum directly..
}

impl fmt::Display for FlowOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlowOp::Ite => write!(f, "Ite"),
            FlowOp::Id => write!(f, "Id"),
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

    // Control flow
    Ite,
    // Special operations
    Id,

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
            IType::Ite => write!(f, "Ite"),
            IType::Id => write!(f, "Id"),
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

impl Theory for IType {
    type DType = Type;

    fn type_check<'a, R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<&'a Self::DType>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
        Self::DType: 'a,
    {
        // FIXME
        // TODO: don't forget to check that BV are 32-bit
        Ok(())
    }
}
