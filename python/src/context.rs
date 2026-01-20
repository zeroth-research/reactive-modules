use pyo3::prelude::*;

// the context in `toy` crate is generic,
// we'll use it until we have the context in `base`.
use crate::{DType, Wire};
use common::context::Context;

/// Context for generating Atoms and Modules from Python
#[pyclass]
pub struct RustContext {
    pub(crate) ctx: Context<DType>,
}

impl Default for RustContext {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl RustContext {
    #[new]
    pub fn new() -> Self {
        Self {
            ctx: Context::<DType>::new(),
        }
    }

    fn fresh_wire_id(&mut self) -> usize {
        self.ctx.tmp_id()
    }

    fn tmp_wire(&mut self, dtype: DType) -> Wire {
        Wire::new(self.ctx.tmp_id(), dtype)
    }

    //pub fn get(&mut self, name: &str, dtype: DType) -> PyResult<Wire> {
    //    self.ctx.get(name).unwrap().0
    //}

    fn wire(&mut self, name: &str, dtype: DType) -> Wire {
        let (id, ty) = self.ctx.var(name, dtype.into());
        Wire::new(id, DType::from(ty))
    }
}
