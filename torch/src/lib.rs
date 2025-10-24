mod context;
mod pyatom;
mod pytensor;
mod pyterm;
mod pyval;
mod term;

use pyo3::prelude::*;
use pyterm::print_pyterm;

pub use context::Context;
pub use pyatom::PyAtom;
pub use pytensor::PyTensor;
pub use pyterm::PyTerm;
pub use pyval::PyVal;

#[pymodule]
#[pyo3(name = "zrm_torch")]
fn zrm_torch(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    //#[pymodule_export]
    m.add_class::<PyVal>()?;
    m.add_class::<PyTerm>()?;
    m.add_class::<PyAtom>()?;
    m.add_class::<Context>()?;

    py.import("torch")?;
    m.add_function(wrap_pyfunction!(print_pyterm, m)?)?;
    Ok(())
}
