mod module;
mod term;
mod wire;

use super::pytensor::PyTensor;
pub use module::Module;
pub use term::Term;
pub use wire::Wire;

use pyo3::PyClass;
use pyo3::prelude::*;

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

pub(crate) fn try_iter_extract<P>(
    iter: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = PyResult<P>>>
where
    P: Clone + PyClass,
{
    let iter = iter
        .try_iter()?
        .map(|i| i?.extract::<P>().map_err(PyErr::from));
    Ok(iter)
}

pub(crate) fn try_iter_pair_extract<P>(
    iter: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = PyResult<(P, P)>>>
where
    P: Clone + PyClass,
{
    let iter = iter
        .try_iter()?
        .map(|i| i?.extract::<(P, P)>().map_err(PyErr::from));
    Ok(iter)
}
