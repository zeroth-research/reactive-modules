use std::fmt;

/// Element-wise propositional logic operations.
///
/// Operands must have compatible shapes. This is enforced
/// by the type checker, not by the enum itself.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Op {
    Not,
    And,
    Or,
    Xor,
    Xnor,
    Implies,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Not => write!(f, "Not"),
            Op::And => write!(f, "And"),
            Op::Or => write!(f, "Or"),
            Op::Xor => write!(f, "Xor"),
            Op::Xnor => write!(f, "Xnor"),
            Op::Implies => write!(f, "Implies"),
        }
    }
}
