use base::term::Term;

#[derive(Debug)]
pub enum TorchOp {
    // constants are special terms
    Const(tch::Tensor),
    // comparisons
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
    // arithmetic
    Sum,
    Add,
    Mul,
    Sub,
    Div,
    // Guard
    Guard,
    // either boolean negation or negating a number
    // (depending on the type of inputs)
    Neg,
}

pub type TorchDType = &'static str;
pub type TorchTerm = Term<TorchDType, TorchOp>;

impl TorchOp {
    pub fn from_str(s: &str) -> Self {
        match s {
            "Eq" => TorchOp::Eq,
            "Neq" => TorchOp::Neq,
            "Lt" => TorchOp::Lt,
            "Le" => TorchOp::Le,
            "Gt" => TorchOp::Gt,
            "Ge" => TorchOp::Ge,
            // -----
            "Add" => TorchOp::Add,
            "Sub" => TorchOp::Sub,
            "Mul" => TorchOp::Mul,
            "Div" => TorchOp::Div,
            "Sum" => TorchOp::Sum,
            // -----
            "Guard" => TorchOp::Guard,
            "Neg" => TorchOp::Neg,
            // -----
            "Const" => panic!("Const cannot be constructed from a &str"),
            oth => panic!("Invalid TorchOp: {} (maybe just not added yet)", oth),
        }
    }
}

impl Clone for TorchOp {
    fn clone(&self) -> Self {
        match self {
            TorchOp::Eq => TorchOp::Eq,
            TorchOp::Neq => TorchOp::Neq,
            TorchOp::Lt => TorchOp::Lt,
            TorchOp::Le => TorchOp::Le,
            TorchOp::Gt => TorchOp::Gt,
            TorchOp::Ge => TorchOp::Ge,
            TorchOp::Add => TorchOp::Add,
            TorchOp::Sub => TorchOp::Sub,
            TorchOp::Mul => TorchOp::Mul,
            TorchOp::Div => TorchOp::Div,
            TorchOp::Sum => TorchOp::Sum,
            TorchOp::Guard => TorchOp::Guard,
            TorchOp::Neg => TorchOp::Neg,
            TorchOp::Const(v) => Self::Const(v.shallow_clone()),
        }
    }
}
