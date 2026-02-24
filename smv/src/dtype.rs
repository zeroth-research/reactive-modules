use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub enum DType {
    Bool,
    Int,
    UWord(u32),
    SWord(u32),
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Bool => write!(f, "Bool"),
            DType::Int => write!(f, "Int"),
            DType::UWord(n) => write!(f, "UWord{}", n),
            DType::SWord(n) => write!(f, "SWord{}", n),
        }
    }
}
