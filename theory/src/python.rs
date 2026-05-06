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
            Type::BV32(_t) => unimplemented!(),
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

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CmpOp {
    Le,
    Lt,
    Ge,
    Gt,
    Eq,
    Ne,
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

#[derive(Clone, PartialEq, Debug)]
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

#[derive(Debug, Clone)]
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

    // Casting
    BVToBool,
    BVToWord1,

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
            IType::NN(_x) => todo!(),
            IType::Tensor(_x) => todo!(),
            IType::Ite => write!(f, "Ite"),
            IType::Id => write!(f, "Id"),
            IType::BVToBool => write!(f, "BVToBool"),
            IType::BVToWord1 => write!(f, "BVToWord1"),
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
