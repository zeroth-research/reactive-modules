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