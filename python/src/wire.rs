use crate::DType;
use pyo3::prelude::*;
use std::fmt;

#[pyclass(frozen, eq, hash, str)]
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub(crate) struct Wire {
    base: base::Wire<theory::python::Type>,
}

#[pymethods]
impl Wire {
    #[new]
    pub(crate) fn new(dtype: DType) -> Self {
        let base = base::Wire::new(dtype.0);
        Self { base }
    }

    #[getter]
    fn id(&self) -> usize {
        self.base.id()
    }

    #[getter]
    fn dtype(&self) -> DType {
        self.base.dtype().clone().into()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base)
    }
}

impl Wire {
    pub(crate) fn base(&self) -> &base::Wire<theory::python::Type> {
        &self.base
    }
}

impl From<base::Wire<theory::python::Type>> for Wire {
    fn from(base: base::Wire<theory::python::Type>) -> Self {
        Self { base }
    }
}

impl fmt::Display for Wire {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.base.fmt(f)
    }
}
