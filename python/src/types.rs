use crate::pytensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt;
use theory::Theory;

// ============================================================================
// DType
// ============================================================================

#[pyclass(frozen, eq, str)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DType(pub(crate) theory::python::Type);

#[pymethods]
impl DType {
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

// ============================================================================
// IType
// ============================================================================

// FIXME #[pyclass(str, frozen)]
#[pyclass(frozen)]
#[derive(Debug, Clone)]
pub struct IType(theory::python::IType);

impl From<theory::python::IType> for IType {
    fn from(t: theory::python::IType) -> IType {
        IType(t)
    }
}

impl From<IType> for theory::python::IType {
    fn from(t: IType) -> theory::python::IType {
        t.0
    }
}
