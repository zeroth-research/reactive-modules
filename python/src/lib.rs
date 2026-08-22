use ::theory::bv::BV;
use ::theory::lia::LIA;
use ::theory::lra::LRA;
use pyo3::PyClass;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::borrow::Borrow;
use std::collections::HashSet;
use std::fmt::Debug;
use std::fmt::Write;
use theory::Theory;
use theory::any::{Any, Combinatorial, Differential, Sequential, Sort};

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

    m.add_class::<Combinatorial>()?;
    m.add_class::<Sequential>()?;
    m.add_class::<Differential>()?;
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
) -> PyResult<impl Iterator<Item = base::Var<theory::any::Sort>>> {
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

/// The python-facing `hide` argument: either a callable predicate over
/// variables (e.g. a lambda), or any iterable of variables (preferably a
/// set); `None` hides nothing.
pub(crate) enum Hide<'py> {
    Set(HashSet<base::Var<Sort>>),
    Call(
        &'py Bound<'py, PyAny>,
        // errors raised by the callable, surfaced after the base call
        std::cell::RefCell<Option<PyErr>>,
    ),
}

impl<'py> Hide<'py> {
    pub(crate) fn try_from(hide: &'py Bound<'py, PyAny>) -> PyResult<Self> {
        if hide.is_callable() {
            Ok(Hide::Call(hide, Default::default()))
        } else if let Ok(iter) = try_var_iter_cloned(hide) {
            Ok(Hide::Set(iter.collect()))
        } else {
            Err(PyException::new_err(
                "hide must be either callable or iterable",
            ))
        }
    }

    /// The predicate handed to the base constructors.
    pub(crate) fn as_fn(&self) -> impl Fn(&base::Var<Sort>) -> bool + '_ {
        move |var| match self {
            Hide::Set(set) => set.contains(var),
            Hide::Call(hide, err) => {
                if err.borrow().is_some() {
                    return false;
                }
                match hide.call1((Var::from(*var),)).and_then(|r| r.is_truthy()) {
                    Ok(hidden) => hidden,
                    Err(e) => {
                        *err.borrow_mut() = Some(e);
                        false
                    }
                }
            }
        }
    }

    /// Propagates any error the callable raised inside the predicate; must
    /// be checked before trusting the base result.
    pub(crate) fn err(self) -> PyResult<()> {
        match self {
            Hide::Call(_, err) => err.into_inner().map_or(Ok(()), Err),
            Hide::Set(_) => Ok(()),
        }
    }
}

/// Renders wires untyped and comma-separated. Accepts references to wires
/// and to variables alike: a `Var` borrows as its latched `Wire`.
pub(crate) fn wires_untyped_to_repr<'a, S, W, I>(wires: I) -> String
where
    S: Debug + 'a,
    W: Borrow<base::Wire<S>> + 'a,
    I: IntoIterator<Item = &'a W>,
{
    let mut s = String::from("[");
    for (i, wire) in wires.into_iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        write!(s, "{:?}", wire.borrow().typed()).unwrap(); // we know it's infallible
    }
    s.push_str("]");
    s
}
