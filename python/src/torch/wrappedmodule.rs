use crate::torch::wrappedcontext::WrappedContext;
use torch::{DType, IType, TorchTerm};

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::torch::wrappedatom::{WrappedAtom, vars_to_wiring};
use base::Module;

#[pyclass]
pub struct WrappedModule {
    module: Module<DType, IType>,
}

// It is safe to share this struct for the same reasons as for PyTensor
unsafe impl Sync for WrappedModule {}

#[pymethods]
impl WrappedModule {
    #[new]
    fn new(
        _ctx: &Bound<'_, WrappedContext>,
        // current-state variables
        latched: &Bound<'_, PyList>,
        // next-state variables
        next: &Bound<'_, PyList>,
        // list of atoms
        atom: &Bound<'_, WrappedAtom>,
    ) -> Self {
        let latched = vars_to_wiring(latched).unwrap();
        let next = vars_to_wiring(next).unwrap();

        let atom: &WrappedAtom = &atom.borrow();

        Self {
            module: Module::sequential(
                [latched, next],
                atom.atom
                    .init()
                    .iter()
                    .map(|term| {
                        TorchTerm::new(
                            term.itype().clone(),
                            term.writes().clone(),
                            term.reads().clone(),
                        )
                    })
                    .collect::<Vec<TorchTerm>>(),
                atom.atom
                    .update()
                    .iter()
                    .map(|term| {
                        TorchTerm::new(
                            term.itype().clone(),
                            term.writes().clone(),
                            term.reads().clone(),
                        )
                    })
                    .collect::<Vec<TorchTerm>>(),
            )
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
