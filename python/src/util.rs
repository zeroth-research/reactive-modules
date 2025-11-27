use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn str_to_pyerr(e: &'static str) -> PyErr {
    PyErr::new::<PyValueError, _>(e)
}
