use super::DType;
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct Wire {
    base: base::Wire<DType>,
}

#[pymethods]
impl Wire {
    #[staticmethod]
    pub fn new(id: usize, dtype: DType) -> Self {
        let base = base::Wire::new(id, dtype);
        Self { base }
    }
}

impl Wire {
    pub(crate) fn base(&self) -> &base::Wire<DType> {
        &self.base
    }
}
