use pyo3::prelude::*;

// the context in `toy` crate is generic,
// we'll use it until we have the context in `base`.
use smt::dtype::DType;
use smt::itype::IType;

use bmc::transition::WiredTransitions;
use bmc::unrolling::ModuleUnrolling;

use super::wrappedcontext::WrappedContext;
use super::wrappedmodule::WrappedModule;

/// Context for generating Atoms and Modules from Python
#[pyclass]
pub struct WrappedWiredTransitions {
    pub(crate) transitions: WiredTransitions<DType, IType>,
}

impl Default for WrappedWiredTransitions {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl WrappedWiredTransitions {
    #[new]
    pub fn new() -> Self {
        Self {
            transitions: WiredTransitions::new(),
        }
    }

    pub fn init(
        self_: Bound<'_, Self>,
        module: &Bound<'_, WrappedModule>,
        ctx: &Bound<'_, WrappedContext>,
    ) {
        let module = module.borrow();
        let mut ctx = ctx.borrow_mut();

        let mut unroll = ModuleUnrolling::<DType, IType>::new(&module.module, &mut ctx.ctx);
        unroll.init_ref(&mut self_.borrow_mut().transitions);
    }

    pub fn step(
        self_: Bound<'_, Self>,
        module: &Bound<'_, WrappedModule>,
        ctx: &Bound<'_, WrappedContext>,
    ) {
        let module = module.borrow();
        let mut ctx = ctx.borrow_mut();

        let mut unroll = ModuleUnrolling::<DType, IType>::new(&module.module, &mut ctx.ctx);
        unroll.step_ref(&mut self_.borrow_mut().transitions);
    }

    pub fn dbg(&self) {
        for transition in &self.transitions {
            println!("---------------------------");
            if let Some(intf_env) = transition.intf_env() {
                println!("In: {}", transition.intf_in());
                println!("Env: {}", intf_env);
            } else {
                println!("In: {}", transition.intf_in());
            }
            println!("---------------------------");
            for term in transition {
                println!("{}", term);
            }
            println!("---------------------------");
            println!("Out: {}", transition.intf_out());
        }
    }
}
//
// impl WrappedContext {
//     /// Translate data from `toy::Context` to `smt::html::Context`. This is necessary
//     /// to be able to dump the smt module into HTML (while keeping using the toy::Context
//     /// when interacting with python).
//     pub(crate) fn to_smt_ctx(&self, module: &base::Module<DType, IType>) -> smt::html::Context {
//         let ctx = smt::html::Context::new(module);
//         for (id, name) in self.ctx.names() {
//             ctx.add_name(*id, name.as_str());
//         }
//         ctx
//     }
// }
