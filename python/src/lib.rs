use pyo3::PyClass;
use pyo3::prelude::*;
use pyo3::types::PyAny;

mod atom;
mod itype;
mod module;
mod pytensor;
mod term;
mod types;
mod wire;

use crate::atom::Atom;
use crate::itype::{ArithOp, ArithOpVal, BVIType, BoolIType, CmpIType, FlowIType, FloatIType, IntIType, IType, NNIType, RealIType, TensorIType};
use crate::module::Module;
use crate::term::Term;
use crate::types::DType;
use crate::wire::Wire;

#[pymodule]
fn zrth(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IType>()?;
    // namespace helper classes — must be registered so Python can inspect them
    m.add_class::<BoolIType>()?;
    m.add_class::<IntIType>()?;
    m.add_class::<FloatIType>()?;
    m.add_class::<RealIType>()?;
    m.add_class::<CmpIType>()?;
    m.add_class::<NNIType>()?;
    m.add_class::<TensorIType>()?;
    m.add_class::<FlowIType>()?;
    m.add_class::<BVIType>()?;
    m.add_class::<ArithOp>()?;
    m.add_class::<ArithOpVal>()?;
    m.add_class::<DType>()?;
    m.add_class::<Wire>()?;
    m.add_class::<Term>()?;
    m.add_class::<Module>()?;

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
) -> PyResult<impl Iterator<Item = [base::Wire<theory::python::Type>; 2]>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_array2_iter_borrow::<Wire>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| r.map(|r| r.base().clone()));
    Ok(seq)
}

fn try_term_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = base::Term<theory::python::IType>>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_iter_borrow::<Term>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| r.base().clone());
    Ok(seq)
}

fn try_wire_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = base::Wire<theory::python::Type>>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_iter_borrow::<Wire>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| r.base().clone());
    Ok(seq)
}
