use crate::val::Val;
use std::fmt;

#[derive(Debug, Copy, Clone)]
pub enum CmpOp {
    Eq,
    Lt,
    Le,
}

#[derive(Debug, Copy, Clone)]
pub enum LogicalOp {
    And,
    Or,
    Not,
}

#[derive(Debug, Copy, Clone)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone)]
pub enum IType {
    // constant and identity terms
    Const(Val),
    Id,
    // arith
    Arith(ArithOp),
    // Comparisons
    Cmp(CmpOp),
    // Logical ops
    Logical(LogicalOp),
    // If-then-else
    Ite,
    // Takes a bool and a value and acts as Id if the bool is true, otherwise returns Val::None.
    // That is, it is fundamentally `Ite(cond, val, None)` (but without having to create None
    // via a term)
    IfThen,
    // Choose non-deterministically one of the values that is not Val::None
    Choose,
}

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CmpOp::Eq => write!(f, "Eq"),
            CmpOp::Lt => write!(f, "Lt"),
            CmpOp::Le => write!(f, "Le"),
        }
    }
}

impl fmt::Display for LogicalOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogicalOp::And => write!(f, "And"),
            LogicalOp::Or => write!(f, "Or"),
            LogicalOp::Not => write!(f, "Not"),
        }
    }
}

impl fmt::Display for ArithOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArithOp::Add => write!(f, "Add"),
            ArithOp::Sub => write!(f, "Sub"),
            ArithOp::Mul => write!(f, "Mul"),
            ArithOp::Div => write!(f, "Div"),
        }
    }
}

impl fmt::Display for IType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IType::Const(v) => write!(f, "Const({})", v),
            IType::Id => write!(f, "Id"),
            IType::Logical(op) => write!(f, "Logical::{}", op),
            IType::Arith(op) => write!(f, "Arith::{}", op),
            IType::Cmp(op) => write!(f, "Cmp::{}", op),
            IType::Ite => write!(f, "Ite"),
            IType::Choose => write!(f, "Choose"),
            IType::IfThen => write!(f, "IfThen"),
        }
    }
}

impl std::str::FromStr for IType {
    type Err = String;

    fn from_str(ins: &str) -> Result<Self, Self::Err> {
        match ins {
            "Id" => Ok(IType::Id),
            "Ite" => Ok(IType::Ite),
            "Choose" => Ok(IType::Choose),
            "IfThen" => Ok(IType::IfThen),
            "Logical::And" => Ok(IType::Logical(LogicalOp::And)),
            "Logical::Or" => Ok(IType::Logical(LogicalOp::Or)),
            "Logical::Not" => Ok(IType::Logical(LogicalOp::Not)),
            "Arith::Add" => Ok(IType::Arith(ArithOp::Add)),
            "Arith::Sub" => Ok(IType::Arith(ArithOp::Sub)),
            "Arith::Mul" => Ok(IType::Arith(ArithOp::Mul)),
            "Arith::Div" => Ok(IType::Arith(ArithOp::Div)),
            "Cmp::Eq" => Ok(IType::Cmp(CmpOp::Eq)),
            "Cmp::Le" => Ok(IType::Cmp(CmpOp::Le)),
            "Cmp::Lt" => Ok(IType::Cmp(CmpOp::Lt)),
            _ => Err(format!("Cannot convert `{}` to IType", ins)),
        }
    }
}
