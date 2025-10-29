use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyInt};

pub use crate::pytensor::PyTensor;

// A wrapper around a value that carries also the type of the value.
// We need it to know what we pass.
// XXX: Once we have the types in our structures, this enum will likely change
#[derive(Debug, Clone)]
#[pyclass]
pub enum PyVal {
    // `Sym` is an identifier (a symbol) in a wire.
    // Every symbol is represented by an `usize` number (see [Wire]).
    Sym(usize),
    // values
    Tensor(PyTensor),
    I64(i64),
    F64(f64),
}

#[pymethods]
impl PyVal {
    #[new]
    fn new(obj: &Bound<'_, PyAny>) -> PyResult<PyVal> {
        if obj.is_instance_of::<PyInt>() {
            let val: i64 = obj.extract()?;
            Ok(PyVal::I64(val))
        } else if obj.is_instance_of::<PyFloat>() {
            let val: f64 = obj.extract()?;
            Ok(PyVal::F64(val))
        } else if let Ok(tensor) = obj.extract::<PyTensor>() {
            Ok(PyVal::Tensor(tensor))
        } else {
            panic!("Unknown PyVal argument type: {:?}", obj.get_type());
        }
    }

    #[staticmethod]
    fn sym(val: usize) -> PyResult<PyVal> {
        Ok(PyVal::Sym(val))
    }

    fn __repr__(&self) -> String {
        format!("PyVal::{:?}", self)
    }
}
