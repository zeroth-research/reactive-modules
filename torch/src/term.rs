#[derive(Debug)]
pub enum TorchOp {
    // constants are special terms
    Const,
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

impl TorchOp {
    pub fn from_str(s: &str) -> Self {
        match s {
            "Const" => TorchOp::Const,
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
            oth => panic!("Invalid TorchOp: {} (maybe just not added yet)", oth),
        }
    }
}
