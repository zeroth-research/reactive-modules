use crate::wire::Wire;
use pyo3::prelude::*;
use std::fmt;
use theory::any::Sort;

#[pyclass(frozen, eq, hash, str)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Var {
    base: base::Variable<Sort>,
}

#[pymethods]
impl Var {
    #[new]
    pub(crate) fn new(dtype: Sort) -> Self {
        let base = base::Variable::new(dtype);
        Self { base }
    }

    #[getter]
    fn id(&self) -> usize {
        self.base.id()
    }

    #[getter]
    fn dtype(&self) -> Sort {
        *self.base.dtype()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base)
    }
}

impl Var {
    pub(crate) fn base(&self) -> &base::Variable<Sort> {
        &self.base
    }
}

impl From<base::Variable<Sort>> for Var {
    fn from(base: base::Variable<Sort>) -> Self {
        Self { base }
    }
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.base.fmt(f)
    }
}

/// The next wire of the variable.
#[pyfunction]
#[pyo3(name = "X")]
pub(crate) fn x(var: PyRef<'_, Var>) -> Wire {
    base::variable::X(var.base).into()
}

/// The derivative wire of the variable.
#[pyfunction]
pub(crate) fn d(var: PyRef<'_, Var>) -> Wire {
    base::variable::d(var.base).into()
}
