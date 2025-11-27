use pyo3::prelude::*;

use crate::toy::pyval::PyVal;

/// Context for generating Atoms and Modules from Python
#[pyclass]
pub struct WrappedContext {
    pub(crate) ctx: toy::ToyContext,
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
            ctx: toy::context::Context::new(),
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
