use crate::DType;
use pyo3::prelude::*;

#[pyclass(frozen)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct Wire {
    base: base::Wire<DType>,
}

#[pymethods]
impl Wire {
    #[new]
    pub(crate) fn new(dtype: DType) -> Self {
        let base = base::Wire::new(dtype);
        Self { base }
    }

    fn id(&self) -> usize {
        self.base.id()
    }

    fn dtype(&self) -> DType {
        self.base.dtype().clone()
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

    fn __hash__(&self) -> usize {
        self.id()
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
