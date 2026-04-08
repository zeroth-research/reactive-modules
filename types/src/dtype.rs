use std::fmt;

/// Data types for wires/terms.
///
/// Every variant carries a shape (`Vec<usize>`). Scalars are
/// represented as 0-dimensional, i.e. an empty shape `vec![]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DType {
    Bool(Vec<usize>),
    Int(Vec<usize>),
    Float(Vec<usize>),
    Real(Vec<usize>),
    /// Unsigned bitvector: bit-width + shape.
    UWord(u32, Vec<usize>),
    /// Signed bitvector: bit-width + shape.
    SWord(u32, Vec<usize>),
}

fn fmt_shape(f: &mut fmt::Formatter<'_>, shape: &[usize]) -> fmt::Result {
    write!(f, "(")?;
    for (i, dim) in shape.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{dim}")?;
    }
    write!(f, ")")
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Bool(s) => { write!(f, "Bool")?; fmt_shape(f, s) }
            DType::Int(s) => { write!(f, "Int")?; fmt_shape(f, s) }
            DType::Float(s) => { write!(f, "Float")?; fmt_shape(f, s) }
            DType::Real(s) => { write!(f, "Real")?; fmt_shape(f, s) }
            DType::UWord(n, s) => { write!(f, "UWord<{n}>")?; fmt_shape(f, s) }
            DType::SWord(n, s) => { write!(f, "SWord<{n}>")?; fmt_shape(f, s) }
        }
    }
}
