use base::term::Term;
use std::fmt;

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
    // Id term
    Id,
    // If-then-else and Choose
    Ite,
    Choose,
    // Boolean terms
    //
    Neg,
    Or,
    And,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorchDType {
    Tensor,
}

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
            "Id" => TorchOp::Id,
            // -----
            "Ite" => TorchOp::Ite,
            "Choose" => TorchOp::Choose,
            // -----
            "Neg" => TorchOp::Neg,
            "Or" => TorchOp::Or,
            "And" => TorchOp::And,
            // -----
            "Const" => panic!("Const cannot be constructed from a &str"),
            oth => panic!("Invalid TorchOp: {} (maybe just not added yet)", oth),
        }
    }
}

impl Clone for TorchOp {
    fn clone(&self) -> Self {
        match self {
            TorchOp::Const(v) => Self::Const(v.shallow_clone()),
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
            TorchOp::Id => TorchOp::Id,
            TorchOp::Ite => TorchOp::Ite,
            TorchOp::Choose => TorchOp::Choose,
            TorchOp::Neg => TorchOp::Neg,
            TorchOp::And => TorchOp::And,
            TorchOp::Or => TorchOp::Or,
        }
    }
}

impl fmt::Display for TorchOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TorchOp::Eq => write!(f, "Eq"),
            TorchOp::Neq => write!(f, "Neq"),
            TorchOp::Lt => write!(f, "Lt"),
            TorchOp::Le => write!(f, "Le"),
            TorchOp::Gt => write!(f, "Gt"),
            TorchOp::Ge => write!(f, "Ge"),
            TorchOp::Add => write!(f, "Add"),
            TorchOp::Sub => write!(f, "Sub"),
            TorchOp::Mul => write!(f, "Mul"),
            TorchOp::Div => write!(f, "Div"),
            TorchOp::Sum => write!(f, "Sum"),
            TorchOp::Id => write!(f, "Id"),
            TorchOp::Ite => write!(f, "Ite"),
            TorchOp::Choose => write!(f, "Choose"),
            TorchOp::Neg => write!(f, "Neg"),
            TorchOp::And => write!(f, "And"),
            TorchOp::Or => write!(f, "Or"),
            TorchOp::Const(t) => {
                let flat = t.view([-1]);

                if let Ok(vals) = Vec::<f64>::try_from(&flat) {
                    let _ = write!(f, "Const([");
                    for (n, v) in vals.iter().take(3).enumerate() {
                        if n == 0 {
                            let _ = write!(f, "{}", v);
                        } else {
                            let _ = write!(f, " {}", v);
                        }
                    }
                    if flat.numel() > 3 {
                        let _ = write!(f, " ...");
                    }
                    write!(f, "])")
                } else {
                    write!(f, "Const({})", flat)
                }
            }
        }
    }
}

impl fmt::Display for TorchDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TorchDType::Tensor => write!(f, "Tensor"),
        }
    }
}
