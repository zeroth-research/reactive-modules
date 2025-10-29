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

    // Temporal / assignment constructs
    Next(Box<IType>),
    Init(Box<IType>),
    Assign(Box<IType>, Box<IType>),
}