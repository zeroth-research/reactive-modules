use super::DType;
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Eq, PartialEq)]
pub struct Wire {
    base: base::Wire<DType>,
}

#[pymethods]
impl Wire {
    #[new]
    fn new(dtype: DType, id: usize) -> Self {
        let base = base::Wire::new(id, dtype);
        Self { base }
    }

    fn __str__(&self) -> String {
        self.base.to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.base == other.base
    }
}

impl Wire {
    pub(crate) fn base(&self) -> &base::Wire<DType> {
        &self.base
    }
}

impl From<base::Wire<DType>> for Wire {
    fn from(base: base::Wire<DType>) -> Self {
        Self { base }
    }
}
