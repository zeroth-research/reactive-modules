use crate::wire::Wire;
use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};
use theory::any::Sort;

#[pyclass(frozen, str)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub(crate) struct Var {
    base: base::Var<Sort>,
}

#[pymethods]
impl Var {
    #[new]
    pub(crate) fn new(dtype: Sort) -> Self {
        let base = base::Var::new(dtype);
        Self { base }
    }

    /// Attribute fallback to the latched wire, mirroring Rust's `Deref`:
    /// any attribute `Var` does not define itself is answered by the wire
    /// (`id`, `dtype`, `degree`, ...).
    fn __getattr__(slf: PyRef<'_, Self>, name: &str) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let wire: Wire = base::Wire::from(*slf.base()).into();
        wire.into_pyobject(py)?.getattr(name).map(Into::into)
    }

    /// Comparison: variables are totally ordered among themselves, and
    /// equality additionally holds against the latched wire, consistent
    /// with the deref view — a variable *is* its latched wire.
    fn __richcmp__(
        &self,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        if let Ok(other) = other.extract::<PyRef<Var>>() {
            let matches = op.matches(self.base.cmp(&other.base));
            return Ok(matches.into_pyobject(py)?.to_owned().into_any().unbind());
        }
        if let Ok(wire) = other.extract::<PyRef<Wire>>() {
            let eq = &base::Wire::from(self.base) == wire.base();
            let matches = match op {
                CompareOp::Eq => eq,
                CompareOp::Ne => !eq,
                _ => return Ok(py.NotImplemented()),
            };
            return Ok(matches.into_pyobject(py)?.to_owned().into_any().unbind());
        }
        Ok(py.NotImplemented())
    }

    /// Hashing matches equality: the latched wire is the bearing element,
    /// so a variable and its latched wire hash alike and are
    /// interchangeable as dictionary keys.
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.base.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base)
    }
}

impl Var {
    pub(crate) fn base(&self) -> &base::Var<Sort> {
        &self.base
    }
}

impl From<base::Var<Sort>> for Var {
    fn from(base: base::Var<Sort>) -> Self {
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
    base::var::X(&var.base).into()
}

/// The derivative wire of the variable.
#[pyfunction]
pub(crate) fn d(var: PyRef<'_, Var>) -> Wire {
    base::var::d(&var.base).into()
}
