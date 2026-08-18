use ::theory::bv::BV;
use ::theory::lia::LIA;
use ::theory::lra::LRA;
use pyo3::PyClass;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::fmt::Debug;
use theory::Theory;
use theory::any::{Any, Sort};

mod atom;
mod module;
mod term;
mod var;
mod wire;

use crate::atom::Atom;
use crate::module::Module;
use crate::term::Term;
use crate::var::Var;
use crate::wire::Wire;

#[pymodule]
fn zrth(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Sort>()?;
    m.add_class::<Wire>()?;
    m.add_class::<Var>()?;
    m.add_class::<Atom>()?;
    m.add_class::<Term>()?;
    m.add_class::<Module>()?;

    m.add_class::<LRA>()?;
    m.add_class::<LIA>()?;
    m.add_class::<BV>()?;

    m.add_function(wrap_pyfunction!(var::x, m)?)?;
    m.add_function(wrap_pyfunction!(var::d, m)?)?;

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

fn try_var_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = base::Variable<theory::any::Sort>>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_iter_borrow::<Var>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| *r.base());
    Ok(seq)
}

fn try_term_iter_cloned<T>(seq: &Bound<'_, PyAny>) -> PyResult<impl Iterator<Item = base::Term<T>>>
where
    T: Theory,
    base::Term<Any>: TryInto<base::Term<T>>,
    <base::Term<Any> as TryInto<base::Term<T>>>::Error: Debug,
{
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_iter_borrow::<Term>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| r.base().clone().try_into().unwrap());
    Ok(seq)
}

fn try_wire_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = base::Wire<theory::any::Sort>>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = seq.try_iter()?.map(|i| -> PyResult<_> {
        let i = i?;
        if let Ok(wire) = i.extract::<PyRef<Wire>>() {
            return Ok(wire.base().clone());
        }
        // a variable stands for its latched wire in a wire position
        if let Ok(var) = i.extract::<PyRef<Var>>() {
            return Ok(base::Wire::from(*var.base()));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected a Wire or a Variable",
        ))
    });
    let seq = seq.map(Result::unwrap);
    Ok(seq)
}
