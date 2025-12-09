use crate::torch::wrappedcontext::WrappedContext;
use std::iter::zip;
use torch::{DType, IType, TorchTerm};

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::torch::wrappedatom::{vars_to_wiring, wterms_to_torchterms};
use base::Module;

#[pyclass]
pub struct WrappedModule {
    module: Module<DType, IType>,
}

// It is safe to share this struct for the same reasons as for PyTensor
unsafe impl Sync for WrappedModule {}

#[pymethods]
impl WrappedModule {
    /// Construct a new (fully observable) module with a single atom
    /// defined by `init` and `update` lists of terms
    #[new]
    fn new(
        ctx: &Bound<'_, WrappedContext>,
        // current-state variables
        latched: &Bound<'_, PyList>,
        // next-state variables
        next: &Bound<'_, PyList>,
        // init terms
        init: &Bound<'_, PyList>,
        // update terms
        update: &Bound<'_, PyList>,
    ) -> Self {
        let latched = vars_to_wiring(latched).unwrap();
        let next = vars_to_wiring(next).unwrap();

        let ctx: &mut WrappedContext = &mut ctx.borrow_mut();
        let init = wterms_to_torchterms(ctx, init).unwrap();
        let update = wterms_to_torchterms(ctx, update).unwrap();

        Self {
            module: Module::sequential(zip(latched, next).map(|([l], [n])| [l, n]), init, update)
                .expect("Failed creating module"),
        }
    }

    ///
    /// Dump the module to HTML using the `visual` crate
    // #[cfg(feature = "visual-html")]
    // fn to_html(&self, ctx: &Bound<'_, WrappedContext>, path: &str) {
    //     let ctx: &WrappedContext = &ctx.borrow();
    //     //visual::html::write_to_html(self.module, path, Some(ctx))
    //     visual::html::write_to_html(&self.module, path, Some(ctx));
    // }

    //fn set_name(&mut self, name: &str) {
    //    self.module.set_name(name);
    //}

    fn dbg(&self) {
        println!("{}", self.module);
    }
}
