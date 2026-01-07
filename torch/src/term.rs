use base::term::Term;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DType {
    None,               // No value
    Tensor(Vec<usize>), // Tensor with a given shape
    Bool,               // Boolean
}

impl std::str::FromStr for DType {
    type Err = String;

    fn from_str(ty: &str) -> Result<Self, Self::Err> {
        match ty {
            "None" => Ok(DType::None),
            "Bool" => Ok(DType::Bool),
            //"Tensor" => Ok(DType::Tensor(todo!())),
            _ => Err(format!("Cannot convert `{}` to DType", ty)),
        }
    }
}

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
    // product and sum of elements of the tensor
    Prod,
    Sum,
    // standard arithmetic
    Add,
    Mul,
    Sub,
    Div,
    MatMul,
    // Id term
    Id,
    // If-then-else construct
    Ite,
    // IfThen is the same as Ite(cond, x, None). The thing is that we do not need to
    // create the None constant which may be very beneficial in some situations.
    // This construct is at this moment used exclusively to input values into `Choose`
    // and `ChooseOr` statements
    IfThen,
    // Non-deterministic choice from input values which are not None
    Choose,
    // Non-deterministic choice from input values which are not None
    // the last argument is the default values and *cannot* be None
    ChooseOr,
    // Boolean operations
    Not,
    Or,
    And,
}

pub type TorchTerm = Term<DType, IType>;

impl std::str::FromStr for IType {
    type Err = String;

    fn from_str(s: &str) -> Result<IType, Self::Err> {
        match s {
            "Eq" => Ok(IType::Eq),
            "Neq" => Ok(IType::Neq),
            "Lt" => Ok(IType::Lt),
            "Le" => Ok(IType::Le),
            "Gt" => Ok(IType::Gt),
            "Ge" => Ok(IType::Ge),
            // -----
            "Add" => Ok(IType::Add),
            "Sub" => Ok(IType::Sub),
            "Mul" => Ok(IType::Mul),
            "Div" => Ok(IType::Div),
            "MatMul" => Ok(IType::MatMul),
            "Sum" => Ok(IType::Sum),
            "Prod" => Ok(IType::Prod),
            // -----
            "Id" => Ok(IType::Id),
            // -----
            "Ite" => Ok(IType::Ite),
            "Choose" => Ok(IType::Choose),
            "ChooseOr" => Ok(IType::ChooseOr),
            "IfThen" => Ok(IType::IfThen),
            // -----
            "Not" => Ok(IType::Not),
            "Or" => Ok(IType::Or),
            "And" => Ok(IType::And),
            // -----
            "Const" => Err("Const cannot be constructed from a &str".into()),
            oth => Err(format!("Invalid IType: {} (maybe just not added yet)", oth)),
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
            IType::MatMul => IType::MatMul,
            IType::Sum => IType::Sum,
            IType::Prod => IType::Prod,
            IType::Id => IType::Id,
            IType::Ite => IType::Ite,
            IType::Choose => IType::Choose,
            IType::ChooseOr => IType::ChooseOr,
            IType::IfThen => IType::IfThen,
            IType::Not => IType::Not,
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
            IType::MatMul => write!(f, "MatMul"),
            IType::Div => write!(f, "Div"),
            IType::Sum => write!(f, "Sum"),
            IType::Prod => write!(f, "Prod"),
            IType::Id => write!(f, "Id"),
            IType::Ite => write!(f, "Ite"),
            IType::Choose => write!(f, "Choose"),
            IType::ChooseOr => write!(f, "ChooseOr"),
            IType::IfThen => write!(f, "IfThen"),
            IType::Not => write!(f, "Not"),
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
            DType::Tensor(shape) => write!(
                f,
                "Tensor({})",
                shape
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            DType::Bool => write!(f, "Bool"),
            DType::None => write!(f, "None"),
        }
    }
}
