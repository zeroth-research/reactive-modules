use std::fmt;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Type {
    Real,
    Int,
    Bool,
}

impl std::str::FromStr for Type {
    type Err = String;

    fn from_str(ty: &str) -> Result<Self, Self::Err> {
        match ty {
            "Real" => Ok(Type::Real),
            "Int" => Ok(Type::Int),
            "Bool" => Ok(Type::Bool),
            _ => Err(format!("Cannot convert `{}` to Type", ty)),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Real => write!(f, "Real"),
            Type::Int => write!(f, "Int"),
            Type::Bool => write!(f, "Bool"),
        }
    }
}
