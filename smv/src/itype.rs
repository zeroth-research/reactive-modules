#[derive(Clone, Debug)]
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

    // Temporal / assignment constructs
    Next,
    Init,
    Assign,
}
