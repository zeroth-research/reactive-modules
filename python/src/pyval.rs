use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt};

#[cfg(feature = "pytorch")]
use crate::torch::pytensor::PyTensor;

// A wrapper around a value that carries also the type of the value.
// We need it to know what we pass.
#[cfg(feature = "pytorch")]
#[derive(Debug, Clone)]
#[pyclass]
pub enum PyVal {
    // `Sym` is an identifier (a symbol) in a wire.
    // Every symbol is represented by an `usize` number (see [Wire]).
    // It has also associated the type of the value which is the second parameter
    Sym(usize, String),
    Int(i64),
    Bool(bool),
    Tensor(PyTensor),
}

#[cfg(not(feature = "pytorch"))]
#[derive(Debug, Clone)]
#[pyclass]
pub enum PyVal {
    // `Sym` is an identifier (a symbol) in a wire.
    // Every symbol is represented by an `usize` number (see [Wire]).
    // It has also associated the type of the value which is the second parameter
    Sym(usize, String),
    Int(i64),
    Bool(bool),
}

#[pymethods]
impl PyVal {
    #[new]
    fn new(obj: &Bound<'_, PyAny>) -> PyResult<PyVal> {
        #[cfg(feature = "pytorch")]
        {
            if let Ok(tensor) = obj.extract::<PyTensor>() {
                return Ok(PyVal::Tensor(tensor));
            }
        }

        if obj.is_instance_of::<PyInt>() {
            let val: i64 = obj.extract()?;
            Ok(PyVal::Int(val))
        //} else if obj.is_instance_of::<PyFloat>() {
        //    let val: f64 = obj.extract()?;
        //    Ok(PyVal::F64(val))
        //}
        } else if obj.is_instance_of::<PyBool>() {
            let val: bool = obj.extract()?;
            Ok(PyVal::Bool(val))
        } else {
            panic!("Unknown PyVal argument type: {:?}", obj.get_type());
        }
    }

    #[staticmethod]
    fn sym(val: usize, ty: &str) -> PyResult<PyVal> {
        Ok(PyVal::Sym(val, ty.to_string()))
    }

    #[staticmethod]
    fn int(val: i64) -> PyResult<PyVal> {
        Ok(PyVal::Int(val))
    }

    #[staticmethod]
    fn bool(val: bool) -> PyResult<PyVal> {
        Ok(PyVal::Bool(val))
    }

    #[cfg(feature = "pytorch")]
    #[staticmethod]
    fn tensor(val: PyTensor) -> PyResult<PyVal> {
        Ok(PyVal::Tensor(val))
    }

    fn ty(&self) -> String {
        match self {
            PyVal::Int(_) => "Int".to_string(),
            PyVal::Bool(_) => "Bool".to_string(),
            PyVal::Sym(_, ty) => ty.clone(),
            #[cfg(feature = "pytorch")]
            PyVal::Tensor(_) => "Tensor".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("PyVal::{:?}", self)
    }
}
