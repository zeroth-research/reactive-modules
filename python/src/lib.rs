#[cfg(feature = "enable-smt")]
mod smt;
mod toy;
mod util;

#[cfg(feature = "enable-torch")]
mod torch;

pub mod pyval;
pub use pyval::PyVal;

use pyo3::prelude::*;

#[pymodule]
fn zrth(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVal>()?;

    let toy = PyModule::new(py, "toy")?;
    toy.add_class::<toy::WrappedTerm>()?;
    toy.add_class::<toy::WrappedModule>()?;
    toy.add_class::<toy::WrappedContext>()?;

    m.add_submodule(&toy)?;

    #[cfg(feature = "enable-smt")]
    {
        let smt = PyModule::new(py, "smt")?;
        smt.add_class::<smt::WrappedTerm>()?;
        smt.add_class::<smt::WrappedModule>()?;
        smt.add_class::<smt::WrappedContext>()?;

        m.add_submodule(&smt)?;
    }

    #[cfg(feature = "enable-torch")]
    {
        let torch = PyModule::new(py, "torch")?;
        torch.add_class::<torch::WrappedTerm>()?;
        torch.add_class::<torch::WrappedModule>()?;
        torch.add_class::<torch::WrappedContext>()?;

        m.add_submodule(&torch)?;
    }

    m.add_class::<IType>()?;
    m.add_class::<DType>()?;
    m.add_class::<Wire>()?;
    m.add_class::<Term>()?;
    m.add_class::<Module>()?;

    m.add_class::<MyTensor>()?;

    Ok(())
}

mod atom;
mod module;
mod term;
mod wire;

use crate::module::Module;
use crate::term::Term;
use crate::wire::Wire;
use pyo3::PyClass;
use std::fmt;

#[pyclass]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MyTensor {
    data: Vec<usize>,
}

#[pymethods]
impl MyTensor {
    #[new]
    pub fn new(data: Vec<usize>) -> Self {
        Self { data }
    }
}

#[pyclass]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum IType {
    A(),
    B(),
    C(MyTensor),
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
