mod atom;
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

// This is a copy of torch::DType.
// It is easier to duplicate the code that embedding torch::DType.
#[pyclass]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DType {
    Bool(),
    Tensor(Vec<usize>),
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Bool() => write!(f, "Bool"),
            DType::Tensor(shape) => write!(
                f,
                "Tensor<{}>",
                shape
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
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

fn try_iter_borrow<'py, P>(
    iter: &'py Bound<'py, PyAny>,
) -> PyResult<impl Iterator<Item = PyResult<PyRef<'py, P>>>>
where
    P: PyClass,
{
    let iter = iter
        .try_iter()?
        .map(|i| i?.extract::<PyRef<P>>().map_err(PyErr::from));
    Ok(iter)
}

fn try_array2_iter_borrow<'py, P>(
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

fn try_wire2_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = [base::Wire<DType>; 2]>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_array2_iter_borrow::<Wire>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| r.map(|r| r.base().clone()));
    Ok(seq)
}

fn try_term_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = base::Term<DType, IType>>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_iter_borrow::<Term>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| r.base().clone());
    Ok(seq)
}

fn try_wire_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = base::Wire<DType>>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_iter_borrow::<Wire>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| r.base().clone());
    Ok(seq)
}
