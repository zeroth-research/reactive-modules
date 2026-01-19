use pyo3::prelude::*;
use pyo3::types::PyList;
use std::iter::zip;

use crate::toy::wrappedcontext::WrappedContext;
use crate::toy::{vars_to_wiring, wterms_to_terms};

use base::Module;
use toy::ToyModule;

#[pyclass]
pub struct WrappedModule {
    module: ToyModule,
}

// It is safe to share this struct for the same reasons as for PyTensor
unsafe impl Sync for WrappedModule {}

#[pymethods]
impl WrappedModule {
    #[new]
    fn sequential(
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
        let init = wterms_to_terms(ctx, init).unwrap();
        let update = wterms_to_terms(ctx, update).unwrap();

        Self {
            module: Module::sequential_observable(
                zip(latched, next).map(|([l], [n])| [l, n]),
                init,
                update,
            )
            .expect("Failed creating module"),
        }
    }

    #[cfg(feature = "visual-html")]
    fn to_html(&self, ctx: &Bound<'_, WrappedContext>, path: &str) {
        let ctx: &WrappedContext = &ctx.borrow();
        //visual::html::write_to_html(self.module, path, Some(ctx))
        let _ = visual::html::write_to_html(
            &self.module,
            path,
            Some(&toy::visual::html::HTMLDescriptor::new(&ctx.ctx)),
        );
    }

    #[cfg(feature = "enable-smt")]
    fn translate_to(&self, ty: &str) -> crate::smt::WrappedModule {
        match ty {
            "smt" => {
                let smt_module: base::Module<smt::dtype::DType, smt::itype::IType> =
                    toy::conversions::to_smt(&self.module).unwrap();
                return crate::smt::WrappedModule { module: smt_module };
            }
            _ => panic!("Cannot tranlate to {}", ty),
        }
    }

    fn dbg(&self) {
        println!("{}", self.module);
    }
}
