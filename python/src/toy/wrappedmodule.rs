use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::toy::wrappedatom::{WrappedAtom, vars_to_wiring};
use crate::toy::wrappedcontext::WrappedContext;

use base::Module;
use toy::{ToyModule, ToyTerm};

#[pyclass]
pub struct WrappedModule {
    module: ToyModule,
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
                        ToyTerm::new(
                            term.itype().clone(),
                            term.writes().clone(),
                            term.reads().clone(),
                        )
                    })
                    .collect::<Vec<ToyTerm>>(),
                atom.atom
                    .update()
                    .iter()
                    .map(|term| {
                        ToyTerm::new(
                            term.itype().clone(),
                            term.writes().clone(),
                            term.reads().clone(),
                        )
                    })
                    .collect::<Vec<ToyTerm>>(),
            )
            .expect("Failed creating module"),
        }
    }

    #[cfg(feature = "visual-html")]
    fn to_html(&self, ctx: &Bound<'_, WrappedContext>, path: &str) {
        let ctx: &WrappedContext = &ctx.borrow();
        //visual::html::write_to_html(self.module, path, Some(ctx))
        let _ = visual::html::write_to_html(&self.module, path, Some(&ctx.ctx));
    }

    fn translate_to(&self, ty: &str) -> crate::smt::WrappedModule {
        match ty {
            "smt" => {
                let smt_module: base::Module<smt::dtype::DType, smt::itype::IType> =
                    toy::conversions::ModuleConverter(&self.module)
                        .try_into()
                        .unwrap();
                return crate::smt::WrappedModule { module: smt_module };
            }
            _ => panic!("Cannot tranlate to {}", ty),
        }
    }

    fn set_name(&mut self, name: &str) {
        self.module.set_name(name);
    }

    fn dbg(&self) {
        println!("{}", self.module);
    }
}
