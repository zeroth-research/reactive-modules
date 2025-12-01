use base::term::Term;
use std::fmt;

#[derive(Debug)]
pub enum IType {
    // constants are special terms
    Const(tch::Tensor),
    // comparisons (element-wise)
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
    // element-wise product and sum
    Prod,
    Sum,
    // standard arithmetic
    Add,
    Mul,
    Sub,
    Div,
    // Id term
    Id,
    // If-then-else and Choose
    Ite,
    Choose,
    // Boolean operations
    // Tensor([1]) is true and Tensor([0]) is false atm.
    // We might want to add explicit Bool type
    Neg,
    Or,
    And,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Tensor,
}

pub type TorchTerm = Term<DType, IType>;

impl IType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "Eq" => IType::Eq,
            "Neq" => IType::Neq,
            "Lt" => IType::Lt,
            "Le" => IType::Le,
            "Gt" => IType::Gt,
            "Ge" => IType::Ge,
            // -----
            "Add" => IType::Add,
            "Sub" => IType::Sub,
            "Mul" => IType::Mul,
            "Div" => IType::Div,
            "Sum" => IType::Sum,
            "Prod" => IType::Prod,
            // -----
            "Id" => IType::Id,
            // -----
            "Ite" => IType::Ite,
            "Choose" => IType::Choose,
            // -----
            "Neg" => IType::Neg,
            "Or" => IType::Or,
            "And" => IType::And,
            // -----
            "Const" => panic!("Const cannot be constructed from a &str"),
            oth => panic!("Invalid IType: {} (maybe just not added yet)", oth),
        }
    }
}

impl Clone for IType {
    fn clone(&self) -> Self {
        match self {
            IType::Const(v) => Self::Const(v.shallow_clone()),
            IType::Eq => IType::Eq,
            IType::Neq => IType::Neq,
            IType::Lt => IType::Lt,
            IType::Le => IType::Le,
            IType::Gt => IType::Gt,
            IType::Ge => IType::Ge,
            IType::Add => IType::Add,
            IType::Sub => IType::Sub,
            IType::Mul => IType::Mul,
            IType::Div => IType::Div,
            IType::Sum => IType::Sum,
            IType::Prod => IType::Prod,
            IType::Id => IType::Id,
            IType::Ite => IType::Ite,
            IType::Choose => IType::Choose,
            IType::Neg => IType::Neg,
            IType::And => IType::And,
            IType::Or => IType::Or,
        }
    }
}

impl fmt::Display for IType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IType::Eq => write!(f, "Eq"),
            IType::Neq => write!(f, "Neq"),
            IType::Lt => write!(f, "Lt"),
            IType::Le => write!(f, "Le"),
            IType::Gt => write!(f, "Gt"),
            IType::Ge => write!(f, "Ge"),
            IType::Add => write!(f, "Add"),
            IType::Sub => write!(f, "Sub"),
            IType::Mul => write!(f, "Mul"),
            IType::Div => write!(f, "Div"),
            IType::Sum => write!(f, "Sum"),
            IType::Prod => write!(f, "Prod"),
            IType::Id => write!(f, "Id"),
            IType::Ite => write!(f, "Ite"),
            IType::Choose => write!(f, "Choose"),
            IType::Neg => write!(f, "Neg"),
            IType::And => write!(f, "And"),
            IType::Or => write!(f, "Or"),
            IType::Const(t) => {
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

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Tensor => write!(f, "Tensor"),
        }
    }
}
