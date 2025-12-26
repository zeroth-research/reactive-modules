mod term;
mod wire;

pub use term::Term;
pub use wire::Wire;

use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum IType {
    A,
    B,
}

#[pyclass]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DType {
    C,
    D,
}
