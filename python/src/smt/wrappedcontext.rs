use pyo3::prelude::*;

use crate::smt::PyVal;

use common::context::Context;
use smt::dtype::DType;
use smt::itype::IType;

/// Context for generating Atoms and Modules from Python
#[pyclass]
pub struct WrappedContext {
    pub(crate) ctx: Context<DType>,
}

impl Default for WrappedContext {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl WrappedContext {
    #[new]
    pub fn new() -> Self {
        Self {
            ctx: Context::<DType>::new(),
        }
    }

    /// Get or create a new named wire (symbol)
    pub fn sym(&mut self, name: &str, ty: &str) -> PyVal {
        let (id, _) = self.ctx.var(name, ty.parse().expect("Invalid type str"));
        PyVal::Sym(id, ty.to_string())
    }

    /// Create and get a new unnamed wire (symbol)
    pub fn tmp_sym(&mut self, ty: &str) -> PyVal {
        PyVal::Sym(self.ctx.tmp_id(), ty.to_string())
    }

    /// Get the *existing* named wire (symbol) in the contet
    pub fn get(&mut self, name: &str) -> PyVal {
        let (id, ty) = self.ctx.get(name).unwrap();
        PyVal::Sym(id, ty.to_string())
    }

    /// Get the ID of an *existing* named wire in the contet
    pub fn get_wire_id(&mut self, name: &str) -> usize {
        self.ctx.get(name).unwrap().0
    }
}

impl WrappedContext {
    /// Translate data from `toy::Context` to `smt::html::Context`. This is necessary
    /// to be able to dump the smt module into HTML (while keeping using the toy::Context
    /// when interacting with python).
    pub(crate) fn to_smt_ctx(&self, module: &base::Module<DType, IType>) -> smt::html::Context {
        let ctx = smt::html::Context::new(module);
        for (id, name) in self.ctx.names() {
            ctx.add_name(*id, name.as_str());
        }
        ctx
    }
}
