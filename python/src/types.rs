use pyo3::prelude::*;
use std::fmt;

#[pyclass]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MyTensor {
    data: Vec<usize>,
}

#[pymethods]
impl MyTensor {
    #[new]
    pub fn new(data: Vec<usize>) -> Self {
        Self { data }
    }
}

#[pyclass]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum IType {
    A(),
    B(),
    C(MyTensor),
}

#[pyclass]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DType {
    C,
    D,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::C => write!(f, "C"),
            DType::D => write!(f, "D"),
        }
    }
}

impl fmt::Display for IType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IType::A() => write!(f, "A"),
            IType::B() => write!(f, "B"),
            IType::C(_) => write!(f, "C(tensor)"),
        }
    }
}
