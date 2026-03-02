use pyo3::PyClass;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

mod atom;
mod lean;
mod module;
mod pytensor;
mod term;
mod types;
mod wire;

use crate::atom::Atom;
use crate::module::Module;
use crate::term::Term;
use crate::types::{DType, IType};
use crate::wire::Wire;

use base::Interface;

#[pymodule]
fn zrth(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IType>()?;
    m.add_class::<DType>()?;
    m.add_class::<Wire>()?;
    m.add_class::<Term>()?;
    m.add_class::<Module>()?;
    m.add_class::<Transition>()?;

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

/// List of terms represented as a transition.
#[pyclass(frozen)]
pub struct Transition(pub(crate) common::transition::Transition<DType, IType>);

#[pymethods]
impl Transition {
    #[new]
    pub fn new(
        vars_in: &Bound<'_, PyAny>,
        vars_env: &Bound<'_, PyAny>,
        vars_env_new: &Bound<'_, PyAny>,
        vars_out: &Bound<'_, PyAny>,
        terms: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let intf_in =
            Interface::sequence(try_wire_iter_cloned(vars_in)?).map_err(PyValueError::new_err)?;
        let intf_out =
            Interface::sequence(try_wire_iter_cloned(vars_out)?).map_err(PyValueError::new_err)?;
        let tmp_env = Interface::try_from_iter(
            try_wire_iter_cloned(vars_env)?.zip(try_wire_iter_cloned(vars_env_new)?),
        )
        .map_err(PyValueError::new_err)?;

        let intf_env = if tmp_env.is_empty() {
            None
        } else {
            Some(tmp_env)
        };

        let terms = try_term_iter_cloned(&terms)?.collect::<Vec<base::Term<DType, IType>>>();

        Ok(Self(common::transition::Transition::new(
            intf_in, intf_env, intf_out, terms,
        )))
    }

    fn intf_in(slf: PyRef<'_, Self>) -> PyResult<TransitionInterface> {
        let py = slf.py();
        let transition = slf.into_pyobject(py)?.unbind();
        Ok(TransitionInterface {
            transition,
            interface: TransitionInterfaceType::In,
        })
    }

    fn intf_out(slf: PyRef<'_, Self>) -> PyResult<TransitionInterface> {
        let py = slf.py();
        let transition = slf.into_pyobject(py)?.unbind();
        Ok(TransitionInterface {
            transition,
            interface: TransitionInterfaceType::Out,
        })
    }

    fn intf_env(slf: PyRef<'_, Self>) -> PyResult<TransitionEnvInterface> {
        let py = slf.py();
        let transition = slf.into_pyobject(py)?.unbind();
        Ok(TransitionEnvInterface { transition })
    }

    // pub fn dbg(&self) {
    //     println!("---------------------------");
    //     println!("Transition:");
    //     if let Some(intf_env) = self.0.intf_env() {
    //         println!("In: {}", self.0.intf_in());
    //         println!("Env: {}", intf_env);
    //     } else {
    //         println!("In: {}", self.0.intf_in());
    //     }
    //     println!("---------------------------");
    //     for term in &self.0.transition {
    //         println!("{}", term);
    //     }
    //     println!("---------------------------");
    //     println!("Out: {}", self.0.intf_out());
    // }
}

#[derive(Clone)]
enum TransitionInterfaceType {
    In,
    Out,
}

#[pyclass(sequence)]
struct TransitionInterface {
    transition: Py<Transition>,
    interface: TransitionInterfaceType,
}

#[pymethods]
impl TransitionInterface {
    fn __getitem__(&self, index: usize) -> PyResult<Wire> {
        let transition = &self.transition.get().0;
        // item.and_then(|i| Some(i.map(Clone::clone).map(Wire::from)))
        //     .ok_or(PyIndexError::new_err("index out of bounds"))
        match self.interface {
            TransitionInterfaceType::In => transition
                .intf_in()
                .entry(index)
                .and_then(|w| Some(Wire::from(w[0].clone())))
                .ok_or(PyIndexError::new_err("index out of bounds")),

            TransitionInterfaceType::Out => transition
                .intf_out()
                .entry(index)
                .and_then(|w| Some(Wire::from(w[0].clone())))
                .ok_or(PyIndexError::new_err("index out of bounds")),
        }
    }

    fn __len__(&self) -> usize {
        let transition = &self.transition.get().0;
        match self.interface {
            TransitionInterfaceType::In => transition.intf_in().len(),
            TransitionInterfaceType::Out => transition.intf_out().len(),
        }
    }

    fn __str__(&self) -> String {
        let transition = &self.transition.get().0;
        match self.interface {
            TransitionInterfaceType::In => transition.intf_in().to_string(),
            TransitionInterfaceType::Out => transition.intf_out().to_string(),
        }
    }
}

#[pyclass(sequence)]
struct TransitionEnvInterface {
    transition: Py<Transition>,
}

#[pymethods]
impl TransitionEnvInterface {
    fn __getitem__(&self, index: usize) -> PyResult<(Wire, Wire)> {
        match &self.transition.get().0.intf_env() {
            None => Err(PyIndexError::new_err("interface is empty")),
            Some(intf) => intf
                .entry(index)
                .map(|w| (w[0].clone().into(), w[1].clone().into()))
                .ok_or_else(|| PyIndexError::new_err("index out of bounds")),
        }
    }

    fn __len__(&self) -> usize {
        self.transition
            .get()
            .0
            .intf_env()
            .map_or(0, |intf| intf.len())
    }

    fn __str__(&self) -> String {
        self.transition
            .get()
            .0
            .intf_env()
            .map_or("()".to_string(), |intf| intf.to_string())
    }
}
