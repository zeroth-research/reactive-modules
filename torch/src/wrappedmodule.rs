use std::fmt;

use crate::context::Context;
use crate::pyval::PyVal;
use crate::term::{TorchDType, TorchOp, TorchTerm};

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::wrappedatom::{WrappedAtom, vars_to_wiring};
use base::{Atom, Module, Wire};

#[cfg(feature = "visual-html")]
use crate::html::*;

type WireTy = Wire<TorchDType>;

#[pyclass]
pub struct WrappedModule {
    module: Module<TorchDType, TorchOp>,
}

// It is safe to share this struct for the same reasons as for PyTensor
unsafe impl Sync for WrappedModule {}

#[pymethods]
impl WrappedModule {
    #[new]
    fn new(
        ctx: &Bound<'_, Context>,
        // current-state variables
        latched: &Bound<'_, PyList>,
        // next-state variables
        next: &Bound<'_, PyList>,
        // list of atoms
        atom: &Bound<'_, WrappedAtom>,
    ) -> Self {
        let latched = vars_to_wiring(latched).unwrap();
        let next = vars_to_wiring(next).unwrap();

        let ctx: &mut Context = &mut ctx.borrow_mut();
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
    #[cfg(feature = "visual-html")]
    fn to_html(&self, ctx: &Bound<'_, Context>, path: &str) {
        let ctx: &Context = &ctx.borrow();
        //visual::html::write_to_html(self.module, path, Some(ctx))
        visual::html::write_to_html(&self.module, path, Some(ctx));
    }

    fn set_name(&mut self, name: &str) {
        self.module.set_name(name);
    }

    fn dbg(&self) {
        println!("{}", self.module);
    }
}
