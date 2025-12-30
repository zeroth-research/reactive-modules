mod module;
mod term;
mod wire;

use super::pytensor::PyTensor;
pub use module::Module;
pub use term::Term;
pub use wire::Wire;

use pyo3::PyClass;
use pyo3::prelude::*;

use std::fmt;

#[pyclass]
#[derive(Debug, Clone)]
pub enum IType {
    A(),
    B(),
    C(PyTensor),
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

pub(crate) fn try_iter_borrow<'py, P>(
    iter: &Bound<'py, PyAny>,
) -> PyResult<impl Iterator<Item = PyResult<PyRef<'py, P>>>>
where
    P: PyClass,
{
    let iter = iter
        .try_iter()?
        .map(|i| i?.extract::<PyRef<P>>().map_err(PyErr::from));
    Ok(iter)
}

pub(crate) fn try_iter_borrow2<'py, P>(
    iter: &Bound<'py, PyAny>,
) -> PyResult<impl Iterator<Item = PyResult<[PyRef<'py, P>; 2]>>>
where
    P: PyClass,
{
    let iter = iter
        .try_iter()?
        .map(|i| i?.extract::<[PyRef<'py, P>; 2]>().map_err(PyErr::from));
    Ok(iter)
}
