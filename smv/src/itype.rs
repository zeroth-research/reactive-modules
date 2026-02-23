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
    Mod,
    Neg,

    // Boolean operations
    Not,
    And,
    Or,
    Xor,
    Xnor,
    Implies,

    // Comparisons
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Neq,

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
            IType::Mod => write!(f, "Mod"),
            IType::Neg => write!(f, "Neg"),
            IType::Not => write!(f, "Not"),
            IType::And => write!(f, "And"),
            IType::Or => write!(f, "Or"),
            IType::Xor => write!(f, "Xor"),
            IType::Xnor => write!(f, "Xnor"),
            IType::Implies => write!(f, "Implies"),
            IType::Lt => write!(f, "Lt"),
            IType::Le => write!(f, "Le"),
            IType::Gt => write!(f, "Gt"),
            IType::Ge => write!(f, "Ge"),
            IType::Eq => write!(f, "Eq"),
            IType::Neq => write!(f, "Neq"),
            IType::Cond => write!(f, "Cond"),
            IType::Abs => write!(f, "Abs"),
            IType::Next => write!(f, "Next"),
            IType::Init => write!(f, "Init"),
            IType::Assign => write!(f, "Assign"),
        }
    }
}
