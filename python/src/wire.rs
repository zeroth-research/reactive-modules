use crate::DType;
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct Wire {
    base: base::Wire<DType>,
}

#[pymethods]
impl Wire {
    #[new]
    pub fn new(dtype: DType, id: usize) -> Self {
        let base = base::Wire::new(id, dtype);
        Self { base }
    }
}

impl From<Wire> for base::Wire<DType> {
    fn from(wire: Wire) -> Self {
        wire.base
    }
}
