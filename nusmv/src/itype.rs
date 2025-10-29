#[derive(Clone, Debug)]
pub enum IType {
    // Values and variables
    ConstBool(bool),
    ConstInt(i64),
    VarRef(String),

    // Arithmetic operations
    Add(Box<IType>, Box<IType>),
    Sub(Box<IType>, Box<IType>),
    Mul(Box<IType>, Box<IType>),
    Div(Box<IType>, Box<IType>),

    // Boolean operations
    Not(Box<IType>),
    And(Box<IType>, Box<IType>),
    Or(Box<IType>, Box<IType>),

    // Comparisons
    Lt(Box<IType>, Box<IType>),
    Le(Box<IType>, Box<IType>),
    Gt(Box<IType>, Box<IType>),
    Ge(Box<IType>, Box<IType>),
    Eq(Box<IType>, Box<IType>),

    // Conditional expression (ternary operator)
    Cond(Box<IType>, Box<IType>, Box<IType>),

    // Temporal / assignment constructs
    Next(Box<IType>),
    Init(Box<IType>),
    Assign(Box<IType>, Box<IType>),
}