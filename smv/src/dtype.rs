use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub enum DType {
    Bool,
    Int,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Bool => write!(f, "Bool"),
            DType::Int => write!(f, "Int"),
        }
    }
}