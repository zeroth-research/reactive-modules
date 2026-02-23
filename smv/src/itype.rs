use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IType {
    // Values and variables
    ConstBool(bool),
    ConstInt(i64),

    // Arithmetic operations
    Add,
    Sub,
    Mul,
    Div,

    // Boolean operations
    Not,
    And,
    Or,

    // Comparisons
    Lt,
    Le,
    Gt,
    Ge,
    Eq,

    // Conditional expression (ternary operator)
    Cond,

    // Absolute value (unary)
    Abs,

    // Temporal / assignment constructs
    Next,
    Init,
    Assign,
}

impl fmt::Display for IType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IType::ConstBool(v) => write!(f, "Const: {}", v),
            IType::ConstInt(v) => write!(f, "Const: {}", v),
            IType::Add => write!(f, "Add"),
            IType::Sub => write!(f, "Sub"),
            IType::Mul => write!(f, "Mul"),
            IType::Div => write!(f, "Div"),
            IType::Not => write!(f, "Not"),
            IType::And => write!(f, "And"),
            IType::Or => write!(f, "Or"),
            IType::Lt => write!(f, "Lt"),
            IType::Le => write!(f, "Le"),
            IType::Gt => write!(f, "Gt"),
            IType::Ge => write!(f, "Ge"),
            IType::Eq => write!(f, "Eq"),
            IType::Cond => write!(f, "Cond"),
            IType::Abs => write!(f, "Abs"),
            IType::Next => write!(f, "Next"),
            IType::Init => write!(f, "Init"),
            IType::Assign => write!(f, "Assign"),
        }
    }
}
