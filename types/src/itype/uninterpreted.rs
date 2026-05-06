use std::fmt;

/// An uninterpreted constant or function, whose signature
/// is known in the context of the current theory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Uninterpreted(pub String);

//impl theory::Operation for Uninterpreted {}

impl fmt::Display for Uninterpreted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
