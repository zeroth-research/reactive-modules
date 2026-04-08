use std::fmt;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Op {
    Sin,
    Cos,
    Tanh,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Sin => write!(f, "Sin"),
            Op::Cos => write!(f, "Cos"),
            Op::Tanh => write!(f, "Tanh"),
        }
    }
}
