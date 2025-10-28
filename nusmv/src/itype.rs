use base::term::Instruction;

#[derive(Clone, Debug)]
pub enum IType {
    // Values and variables
    ConstBool(bool),
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

impl Instruction for IType {
    fn arity(&self) -> Option<usize> {
        match self {
            IType::ConstBool(_) => Some(0),
            IType::VarRef(_) => Some(0),
            IType::Not(_) => Some(1),
            IType::And(_, _) => Some(2),
            IType::Or(_, _) => Some(2),
            IType::Next(_) => Some(1),
            IType::Init(_) => Some(1),
            IType::Assign(_, _) => Some(2),
        }
    }
}
