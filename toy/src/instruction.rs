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

#[derive(Debug, Copy, Clone)]
pub enum Instruction {
    // constant and identity terms
    Const(Val),
    Id,
    // Comparisons
    Cmp(CmpOp),
    // Logical ops
    Logical(LogicalOp),
    Ite,
    // arith
    Arith(ArithOp),
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

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Const(v) => write!(f, "Const({})", v),
            Instruction::Id => write!(f, "Id"),
            Instruction::Logical(op) => write!(f, "Logical::{}", op),
            Instruction::Arith(op) => write!(f, "Arith::{}", op),
            Instruction::Cmp(op) => write!(f, "Cmp::{}", op),
            Instruction::Ite => write!(f, "Ite"),
        }
    }
}

impl std::str::FromStr for Instruction {
    type Err = String;

    fn from_str(ins: &str) -> Result<Self, Self::Err> {
        match ins {
            "Id" => Ok(Instruction::Id),
            "Ite" => Ok(Instruction::Ite),
            "Logical::And" => Ok(Instruction::Logical(LogicalOp::And)),
            "Logical::Or" => Ok(Instruction::Logical(LogicalOp::Or)),
            "Logical::Not" => Ok(Instruction::Logical(LogicalOp::Not)),
            "Arith::Add" => Ok(Instruction::Arith(ArithOp::Add)),
            "Arith::Sub" => Ok(Instruction::Arith(ArithOp::Sub)),
            "Arith::Mul" => Ok(Instruction::Arith(ArithOp::Mul)),
            "Arith::Div" => Ok(Instruction::Arith(ArithOp::Div)),
            "Cmp::Eq" => Ok(Instruction::Cmp(CmpOp::Eq)),
            "Cmp::Le" => Ok(Instruction::Cmp(CmpOp::Le)),
            "Cmp::Lt" => Ok(Instruction::Cmp(CmpOp::Lt)),
            _ => Err(format!("Cannot convert `{}` to Instruction", ins)),
        }
    }
}
