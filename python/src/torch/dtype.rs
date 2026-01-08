use pyo3::exceptions::PyValueError;
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

    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        let dtype = s.parse().map_err(PyValueError::new_err)?;
        Ok(DType(dtype))
    }

    /// Get the data dimensions of this data type
    fn dims(&self) -> Vec<usize> {
        match &self.0 {
            torch::DType::None => vec![0],
            torch::DType::Bool => vec![1],
            torch::DType::Tensor(shape) => shape.clone(),
        }
    }

    fn is_tensor(&self) -> bool {
        match &self.0 {
            torch::DType::Tensor(_) => true,
            _ => false,
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}
