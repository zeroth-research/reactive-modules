use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::smt::wrappedcontext::WrappedContext;
use crate::smt::{vars_to_wiring, wterms_to_torchterms};

use base::Module;
use smt::dtype::DType;
use smt::itype::IType;

use std::iter::zip;

#[pyclass]
pub struct WrappedModule {
    pub(crate) module: Module<DType, IType>,
}

// It is safe to share this struct for the same reasons as for PyTensor
unsafe impl Sync for WrappedModule {}

#[pymethods]
impl WrappedModule {
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

    fn to_smtlib(&self) -> String {
        smt::smt::module_to_smtlib(&self.module)
    }

    #[cfg(feature = "visual-html")]
    fn to_html(&self, ctx: &Bound<'_, WrappedContext>, path: &str) {
        let ctx: &WrappedContext = &ctx.borrow();
        let _ =
            visual::html::write_to_html(&self.module, path, Some(&ctx.to_smt_ctx(&self.module)));
    }

    fn dbg(&self) {
        println!("{}", self.module);
    }
}
