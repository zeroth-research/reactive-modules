use pyo3::prelude::*;
use std::fmt;
use theory::any::Sort;

#[pyclass(frozen, eq, hash, str)]
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
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

    fn __repr__(&self) -> String {
        format!("{:?}", self.base)
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

impl fmt::Display for Wire {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.base.fmt(f)
    }
}
