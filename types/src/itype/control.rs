use std::fmt;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Op {
    /// Identity: passes input through unchanged
    Id,
    /// If-then-else: conditional expression
    Ite,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Id => write!(f, "Id"),
            Op::Ite => write!(f, "Ite"),
        }
    }
}
