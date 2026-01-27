use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::smt::wrappedcontext::WrappedContext;
use crate::smt::{vars_to_wiring, wterms_to_terms};

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

    #[pyo3(signature = (what = None))]
    fn to_smtlib(&self, what: Option<&str>) -> String {
        match what {
            Some("module") | None => smt::smt::module_to_smtlib(&self.module),
            Some("variables") => smt::smt::module_variables_to_smtlib(&self.module),
            Some("init") => smt::smt::module_init_to_smtlib(&self.module),
            Some("update") => smt::smt::module_update_to_smtlib(&self.module),
            _ => panic!("Invalid `what` argument: `{}`", what.unwrap()),
        }
    }

    #[cfg(feature = "visual-html")]
    fn to_html(&self, ctx: &Bound<'_, WrappedContext>, path: &str) {
        let ctx: &WrappedContext = &ctx.borrow();
        let _ = visual::html::module::write_to_html(
            &self.module,
            path,
            Some(&ctx.to_smt_ctx(&self.module)),
        );
    }

    fn dbg(&self) {
        println!("{}", self.module);
    }

    #[staticmethod]
    fn parallel(items: &Bound<'_, PyList>) -> PyResult<WrappedModule> {
        // get Rust references to modules
        let refs = items.iter().map(|item| {
            let py_ref: PyRef<WrappedModule> =
                item.extract().expect("List item is not WrappedModule");
            py_ref.module.clone()
        });

        Ok(WrappedModule {
            module: base::Module::parallel(refs).map_err(PyException::new_err)?,
        })
    }

    fn __str__(&self) -> String {
        self.module.to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.module)
    }
}
