#![allow(non_snake_case)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt;

fn parse_dims(dims: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<(usize, usize)> {
    match dims.len() {
        1 => Ok((1, dims.get_item(0)?.extract::<usize>()?)),
        2 => Ok((
            dims.get_item(0)?.extract::<usize>()?,
            dims.get_item(1)?.extract::<usize>()?,
        )),
        n => Err(PyValueError::new_err(format!(
            "expected 1 or 2 dimensions, got {n}"
        ))),
    }
}

// ============================================================================
// DType
// ============================================================================

#[pyclass(frozen, eq, str)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DType(pub(crate) theory::python::Type);

#[pymethods]
impl DType {
    #[staticmethod]
    #[pyo3(signature = (*dims))]
    fn Bool(dims: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        let (rows, cols) = parse_dims(dims)?;
        Ok(DType(theory::python::Type::Bool(theory::bool::Bool(
            rows, cols,
        ))))
    }

    #[staticmethod]
    #[pyo3(signature = (*dims))]
    fn Int(dims: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        let (rows, cols) = parse_dims(dims)?;
        Ok(DType(theory::python::Type::Int(theory::int::Int(
            rows, cols,
        ))))
    }

    #[staticmethod]
    #[pyo3(signature = (*dims))]
    fn Float(dims: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        let (rows, cols) = parse_dims(dims)?;
        Ok(DType(theory::python::Type::Float(theory::float::Float(
            rows, cols,
        ))))
    }

    #[staticmethod]
    #[pyo3(signature = (*dims))]
    fn Real(dims: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        let (rows, cols) = parse_dims(dims)?;
        Ok(DType(theory::python::Type::Real(theory::real::Real(
            rows, cols,
        ))))
    }

    /// Get the data dimensions of this data type
    #[getter]
    fn shape(&self) -> Vec<usize> {
        todo!()
    }

    /// Create the same (Tensor) dtype but with a different shape
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        todo!()
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<theory::python::Type> for DType {
    fn from(t: theory::python::Type) -> DType {
        DType(t)
    }
}

impl From<DType> for theory::python::Type {
    fn from(t: DType) -> theory::python::Type {
        t.0
    }
}

