#[cfg(feature = "enable-smt")]
mod smt;
mod toy;
mod util;

#[cfg(feature = "enable-torch")]
mod torch;

pub mod pyval;
pub use pyval::PyVal;

use pyo3::prelude::*;

#[pymodule]
fn zrth(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVal>()?;

    let toy = PyModule::new(py, "toy")?;
    toy.add_class::<toy::WrappedTerm>()?;
    toy.add_class::<toy::WrappedModule>()?;
    toy.add_class::<toy::WrappedContext>()?;

    m.add_submodule(&toy)?;

    #[cfg(feature = "enable-smt")]
    {
        let smt = PyModule::new(py, "smt")?;
        smt.add_class::<smt::WrappedTerm>()?;
        smt.add_class::<smt::WrappedModule>()?;
        smt.add_class::<smt::WrappedContext>()?;

        m.add_submodule(&smt)?;
    }

    #[cfg(feature = "enable-torch")]
    {
        let torch = PyModule::new(py, "torch")?;
        torch.add_class::<torch::WrappedTerm>()?;
        torch.add_class::<torch::WrappedModule>()?;
        torch.add_class::<torch::WrappedContext>()?;

        m.add_submodule(&torch)?;
    }

    m.add_class::<IType>()?;
    m.add_class::<DType>()?;
    m.add_class::<Wire>()?;
    m.add_class::<Term>()?;

    Ok(())
}

mod term;
mod wire;

use crate::term::Term;
use crate::wire::Wire;
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone, Eq, PartialEq)]
enum IType {
    A,
    B,
}

#[pyclass]
#[derive(Debug, Clone, Eq, PartialEq)]
enum DType {
    C,
    D,
}
