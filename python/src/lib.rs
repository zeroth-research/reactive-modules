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
#[pyo3(name = "_zrth")]
fn _zrth(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
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
        // the high-level API for Torch integration
        let torch = PyModule::new(py, "torch")?;
        torch.add_class::<torch::WrappedTerm>()?;
        torch.add_class::<torch::WrappedModule>()?;
        torch.add_class::<torch::WrappedContext>()?;

        // add the low-level API
        let ll = PyModule::new(py, "ll")?;
        ll.add_class::<torch::ll::IType>()?;
        ll.add_class::<torch::ll::DType>()?;
        ll.add_class::<torch::ll::Wire>()?;
        ll.add_class::<torch::ll::Term>()?;

        // _zrth.torch.ll
        torch.add_submodule(&ll)?;
        m.add_submodule(&torch)?;
    }

    Ok(())
}
