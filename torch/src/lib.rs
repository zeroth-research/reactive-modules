mod context;
mod pytensor;
mod pyval;
mod term;
mod wrappedatom;
mod wrappedmodule;
mod wrappedterm;

use pyo3::prelude::*;

pub use context::Context;
pub use pytensor::PyTensor;
pub use pyval::PyVal;
pub use wrappedatom::WrappedAtom;
pub use wrappedmodule::WrappedModule;
pub use wrappedterm::WrappedTerm;

#[pymodule]
#[pyo3(name = "zrm_torch")]
fn zrm_torch(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVal>()?;
    m.add_class::<WrappedTerm>()?;
    m.add_class::<WrappedAtom>()?;
    m.add_class::<WrappedModule>()?;
    m.add_class::<Context>()?;

    py.import("torch")?;
    //m.add_function(wrap_pyfunction!(print_pyterm, m)?)?;
    Ok(())
}
