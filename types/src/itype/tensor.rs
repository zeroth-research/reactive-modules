use std::fmt;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Op {
    /// Constant tensor. The actual data payload is held at the
    /// term level by each downstream crate (e.g. `tch::Tensor`,
    /// `PyTensor`), not in this enum.
    Const,
    /// Matrix multiplication. Inputs must be 2-dimensional;
    /// enforced by the type checker.
    MatMul,
    Mean,
    Max,
    Argmax,
    ReLU,
    /// Linear layer: output = input @ weight + bias.
    /// Weight and bias are parameters attached at the term level.
    Linear,
    Prod,
    Sum,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Const => write!(f, "Tensor"),
            Op::MatMul => write!(f, "MatMul"),
            Op::Mean => write!(f, "Mean"),
            Op::Max => write!(f, "Max"),
            Op::Argmax => write!(f, "Argmax"),
            Op::ReLU => write!(f, "ReLU"),
            Op::Linear => write!(f, "Linear"),
            Op::Prod => write!(f, "Prod"),
            Op::Sum => write!(f, "Sum"),
        }
    }
}
