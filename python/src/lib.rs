use pyo3::PyClass;
use pyo3::prelude::*;
use pyo3::types::PyAny;

mod itype;
mod lia;
mod pytensor;

use crate::itype::IType;

#[pymodule]
fn zrth(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IType>()?;

    let lia = PyModule::new(py, "lia")?;
    lia.add_class::<lia::Type>()?;
    lia.add_class::<lia::Wire>()?;
    lia.add_class::<lia::Term>()?;
    lia.add_class::<lia::Atom>()?;
    lia.add_class::<lia::Module>()?;
    lia.add_function(wrap_pyfunction!(lia::Int, &lia)?)?;
    lia.add_function(wrap_pyfunction!(lia::Bool, &lia)?)?;

    // allow to use `zrth.lia` in imports
    m.add_submodule(&lia)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("zrth.lia", &lia)?;
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
