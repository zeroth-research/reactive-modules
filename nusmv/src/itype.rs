#[derive(Clone, Debug)]
pub enum IType {
    // Values and variables
    ConstBool(bool),
    ConstInt(i64),
    VarRef(String),

    // Boolean operations
    Not(Box<IType>),
    And(Box<IType>, Box<IType>),
    Or(Box<IType>, Box<IType>),

    // Arithmetic operations
    Add(Box<IType>, Box<IType>),
    Sub(Box<IType>, Box<IType>),
    Mul(Box<IType>, Box<IType>),
    Div(Box<IType>, Box<IType>),

    // Temporal / assignment constructs
    Next(Box<IType>),
    Init(Box<IType>),
    Assign(Box<IType>, Box<IType>),
}