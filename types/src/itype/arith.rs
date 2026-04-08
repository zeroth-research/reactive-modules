use std::fmt;

/// Element-wise arithmetic operations.
///
/// Operands must have compatible shapes. This is enforced
/// by the type checker, not by the enum itself.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,
    Abs,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Add => write!(f, "Add"),
            Op::Sub => write!(f, "Sub"),
            Op::Mul => write!(f, "Mul"),
            Op::Div => write!(f, "Div"),
            Op::Mod => write!(f, "Mod"),
            Op::Neg => write!(f, "Neg"),
            Op::Abs => write!(f, "Abs"),
        }
    }
}

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CmpOp::Eq => write!(f, "Eq"),
            CmpOp::Neq => write!(f, "Neq"),
            CmpOp::Lt => write!(f, "Lt"),
            CmpOp::Le => write!(f, "Le"),
            CmpOp::Gt => write!(f, "Gt"),
            CmpOp::Ge => write!(f, "Ge"),
        }
    }
}
