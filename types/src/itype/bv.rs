use std::fmt;

/// Bitvector operations.
///
/// TODO: expand with more operations from the MathSAT `make_bv` API:
/// https://mathsat.fbk.eu/apireference.html
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    BitSelect(u32, u32),
    Extend(u32),
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::BitSelect(h, l) => write!(f, "BitSelect[{}:{}]", h, l),
            Op::Extend(n) => write!(f, "Extend({})", n),
        }
    }
}
