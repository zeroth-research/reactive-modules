use std::fmt;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Type {
    Real,
    NReal,
    Int,
    NInt,
    Bool,
}

impl Type {
    pub fn from_str(ty: &str) -> Result<Self, &'static str> {
        match ty {
            "Real" => Ok(Type::Real),
            "NReal" => Ok(Type::NReal),
            "Int" => Ok(Type::Int),
            "NInt" => Ok(Type::NInt),
            "Bool" => Ok(Type::Bool),
            _ => Err("Invalid Type (cannot convert from this str)"),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Real => write!(f, "Real"),
            Type::NReal => write!(f, "NReal"),
            Type::Int => write!(f, "Int"),
            Type::NInt => write!(f, "NInt"),
            Type::Bool => write!(f, "Bool"),
        }
    }
}
