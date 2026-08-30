use pyo3::prelude::*;
use theory::any::Sort;

#[pyclass(frozen, eq, hash, ord)]
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub(crate) struct Wire {
    base: base::Wire<Sort>,
}

#[pymethods]
impl Wire {
    #[new]
    pub(crate) fn new(dtype: Sort) -> Self {
        let base = base::Wire::new(dtype);
        Self { base }
    }

    #[getter]
    fn id(&self) -> usize {
        self.base.id()
    }

    #[getter]
    fn dtype(&self) -> Sort {
        self.base.dtype().clone()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base.typed())
    }
}

impl Wire {
    pub(crate) fn base(&self) -> &base::Wire<Sort> {
        &self.base
    }
}

impl From<base::Wire<Sort>> for Wire {
    fn from(base: base::Wire<Sort>) -> Self {
        Self { base }
    }
}
