use pyo3::prelude::*;

use std::fmt;

#[pyclass]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DType(torch::DType);

impl From<torch::DType> for DType {
    fn from(t: torch::DType) -> Self {
        DType(t)
    }
}

impl Into<torch::DType> for DType {
    fn into(self) -> torch::DType {
        self.0
    }
}

#[pymethods]
impl DType {
    #[staticmethod]
    fn tensor(shape: Vec<usize>) -> Self {
        DType(torch::DType::Tensor(shape))
    }

    #[staticmethod]
    fn bool() -> Self {
        DType(torch::DType::Bool)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}
