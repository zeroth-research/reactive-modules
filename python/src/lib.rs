use crate::theory::Sort;
use ::theory::bv::BV;
use ::theory::lia::LIA;
use ::theory::lra::LRA;
use pyo3::PyClass;
use pyo3::prelude::*;
use pyo3::types::PyAny;

mod atom;
mod module;
mod term;
mod theory;
mod wire;

use crate::atom::Atom;
use crate::module::Module;
use crate::term::Term;
use crate::wire::Wire;

#[pymodule]
fn zrth(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Sort>()?;
    m.add_class::<Wire>()?;
    m.add_class::<Atom>()?;
    m.add_class::<Term>()?;
    m.add_class::<Module>()?;

    m.add_class::<LRA>()?;
    m.add_class::<LIA>()?;
    m.add_class::<BV>()?;

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
) -> PyResult<impl Iterator<Item = [base::Wire<theory::Sort>; 2]>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_array2_iter_borrow::<Wire>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| r.map(|r| r.base().clone()));
    Ok(seq)
}

fn try_term_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = base::Term<theory::Any>>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_iter_borrow::<Term>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| r.base().clone());
    Ok(seq)
}

fn try_wire_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = base::Wire<theory::Sort>>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_iter_borrow::<Wire>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| r.base().clone());
    Ok(seq)
}
