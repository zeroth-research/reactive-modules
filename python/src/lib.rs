use pyo3::PyClass;
use pyo3::prelude::*;
use pyo3::types::PyAny;

//mod atom;
mod lia;
//mod module;
mod pytensor;
// mod term;
mod types;
// mod wire;

// use crate::atom::Atom;
// use crate::module::Module;
// use crate::term::Term;
use crate::types::IType;
// use crate::wire::Wire;

#[pymodule]
fn zrth(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IType>()?;
    //m.add_class::<DType>()?;
    //m.add_class::<Wire>()?;
    //m.add_class::<Term>()?;
    //m.add_class::<Module>()?;

    let lia = PyModule::new(py, "lia")?;
    lia.add_class::<lia::Type>()?;
    lia.add_class::<lia::Wire>()?;
    lia.add_class::<lia::Term>()?;
    lia.add_class::<lia::Atom>()?;
    lia.add_class::<lia::Int>()?;
    lia.add_class::<lia::Bool>()?;

    m.add_submodule(&lia)?;
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
