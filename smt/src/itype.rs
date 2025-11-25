use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Val {
    None, // no value
    Real(f64),
    Int(i64),
    Bool(bool),
}

#[derive(Debug, Copy, Clone)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Copy, Clone)]
pub enum LogicalOp {
    Not,
    And,
    Or,
}

#[derive(Debug, Copy, Clone)]
pub enum CmpOp {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Copy, Clone)]
pub enum IType {
    // Constant
    Const(Val),
    // Arithmetic operations
    Arith(ArithOp),
    // Logical operations
    Logical(LogicalOp),
    // Comparisons
    Cmp(CmpOp),
    // Identity
    Id,
    // Conditional expression (ternary operator)
    Cond,
}

impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Val::None => write!(f, "None"),
            Val::Real(x) => write!(f, "{}", x),
            Val::Int(x) => write!(f, "{}", x),
            Val::Bool(x) => write!(f, "{}", x),
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

impl fmt::Display for LogicalOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogicalOp::Not => write!(f, "Not"),
            LogicalOp::And => write!(f, "And"),
            LogicalOp::Or => write!(f, "Or"),
        }
    }
}

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CmpOp::Eq => write!(f, "Eq"),
            CmpOp::Lt => write!(f, "Lt"),
            CmpOp::Le => write!(f, "Le"),
            CmpOp::Gt => write!(f, "Gt"),
            CmpOp::Ge => write!(f, "Ge"),
        }
    }
}

impl fmt::Display for IType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IType::Const(v) => write!(f, "{}", v),
            IType::Arith(op) => write!(f, "{}", op),
            IType::Logical(op) => write!(f, "{}", op),
            IType::Cmp(op) => write!(f, "{}", op),
            IType::Id => write!(f, "Id"),
            IType::Cond => write!(f, "Cond"),
        }
    }
}
