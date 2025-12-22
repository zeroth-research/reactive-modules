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
    // Constant number
    Num(Val),
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
            IType::Num(v) => write!(f, "{}", v),
            IType::Arith(op) => write!(f, "{}", op),
            IType::Logical(op) => write!(f, "{}", op),
            IType::Cmp(op) => write!(f, "{}", op),
            IType::Id => write!(f, "Id"),
            IType::Cond => write!(f, "Cond"),
        }
    }
}

impl std::str::FromStr for IType {
    type Err = String;

    fn from_str(ins: &str) -> Result<Self, Self::Err> {
        match ins {
            "Id" => Ok(IType::Id),
            "Cond" => Ok(IType::Cond),
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
            "Cmp::Ge" => Ok(IType::Cmp(CmpOp::Ge)),
            "Cmp::Gt" => Ok(IType::Cmp(CmpOp::Gt)),
            _ => Err(format!("Cannot convert `{}` to IType", ins)),
        }
    }
}
