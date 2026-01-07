use pyo3::prelude::*;

use crate::PyVal;

// the context in `toy` crate is generic,
// we'll use it until we have the context in `base`.
use common::context::Context;
use torch::DType;

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

    pub fn fresh_var(&mut self) -> usize {
        self.ctx.tmp_id()
    }

    pub fn tmp_var(&mut self) -> PyVal {
        PyVal::Sym(self.ctx.tmp_id(), "Tensor".to_string())
    }

    pub fn get(&mut self, name: &str) -> usize {
        self.ctx.get(name).unwrap().0
    }

    //pub fn var(&mut self, name: &str) -> PyVal {
    //    let (id, _) = self.ctx.var(name, DType::Tensor);
    //    PyVal::Sym(id, "Tensor".to_string())
    //}
}
