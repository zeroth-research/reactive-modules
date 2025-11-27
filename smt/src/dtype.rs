use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub enum DType {
    Real,
    Int,
    Bool,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Real => write!(f, "Real"),
            DType::Int => write!(f, "Int"),
            DType::Bool => write!(f, "Bool"),
        }
    }
}

impl std::str::FromStr for DType {
    type Err = String;

    fn from_str(ty: &str) -> Result<Self, Self::Err> {
        match ty {
            "Real" => Ok(DType::Real),
            "Int" => Ok(DType::Int),
            "Bool" => Ok(DType::Bool),
            _ => Err(format!("Cannot convert `{}` to DType", ty)),
        }
    }
}
