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
    #[pyo3(signature = (dtype, degree = 0))]
    pub(crate) fn new(dtype: Sort, degree: u8) -> Self {
        let base = base::Wire::new(dtype, degree);
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

    #[getter]
    fn degree(&self) -> u8 {
        self.base.degree()
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
