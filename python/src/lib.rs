#[cfg(feature = "enable-smt")]
mod smt;

use pyo3::prelude::*;

mod atom;
mod context;
mod module;
mod pytensor;
mod term;
mod types;
mod wire;

use crate::context::RustContext;
use crate::module::Module;
use crate::term::Term;
use crate::types::{DType, IType};
use crate::wire::Wire;
use pyo3::PyClass;

#[pymodule]
fn zrth(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "enable-smt")]
    {
        let smt = PyModule::new(py, "smt")?;
        smt.add_class::<smt::WrappedTerm>()?;
        smt.add_class::<smt::WrappedModule>()?;
        smt.add_class::<smt::WrappedContext>()?;
        smt.add_class::<smt::PyVal>()?;

        m.add_submodule(&smt)?;
    }

    m.add_class::<IType>()?;
    m.add_class::<DType>()?;
    m.add_class::<Wire>()?;
    m.add_class::<Term>()?;
    m.add_class::<Module>()?;
    m.add_class::<RustContext>()?;

    Ok(())
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
