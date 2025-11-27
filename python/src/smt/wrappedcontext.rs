use pyo3::prelude::*;

use crate::PyVal;

// the context in `toy` crate is generic,
// we'll use it until we have the context in `base`.
use smt::dtype::DType;
use toy::context::Context;

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

    pub fn tmp_sym(&mut self, ty: &str) -> PyVal {
        let id = self.ctx.tmp_var(ty.parse().expect("Invalid type str"));
        PyVal::Sym(id, ty.to_string())
    }

    pub fn get(&mut self, name: &str) -> PyVal {
        let (id, ty) = self.ctx.get(name);
        PyVal::Sym(id, ty.to_string())
    }

    pub fn get_sym(&mut self, name: &str, ty: &str) -> PyVal {
        let (id, _) = self.ctx.var(name, ty.parse().expect("Invalid type str"));
        PyVal::Sym(id, ty.to_string())
    }
}
