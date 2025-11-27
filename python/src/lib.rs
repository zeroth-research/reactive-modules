mod toy;
mod util;

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_zrth")]
fn _zrth(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let toy = PyModule::new(py, "toy")?;

    //#[pymodule_export]
    toy.add_class::<toy::PyVal>()?;
    toy.add_class::<toy::WrappedTerm>()?;
    toy.add_class::<toy::WrappedAtom>()?;
    toy.add_class::<toy::WrappedModule>()?;
    toy.add_class::<toy::WrappedContext>()?;

    m.add_submodule(&toy)?;

    Ok(())
}
