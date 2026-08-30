use crate::wire::Wire;
use pyo3::BoundObject;
use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::PyNotImplemented;
use std::hash::{DefaultHasher, Hash, Hasher};
use theory::any::Sort;

#[pyclass(frozen)] // eq, hash, ord are implemented manually for compat with polymorphic __richcmp__
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
    /// (`id`, `dtype`, ...).
    fn __getattr__(slf: PyRef<'_, Self>, name: &str) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let wire: Wire = base::Wire::from(*slf.base()).into();
        wire.into_pyobject(py)?.getattr(name).map(Into::into)
    }

    /// Comparison through the latched wire, consistent with the deref view:
    /// a variable compares as its latched wire, against variables and wires
    /// alike, under the wires' total order.
    fn __richcmp__<'py>(
        &self,
        other: &Bound<'py, PyAny>,
        op: CompareOp,
        py: Python<'py>,
    ) -> PyResult<Borrowed<'py, 'py, PyAny>> {
        let ordering = if let Ok(var) = other.extract::<PyRef<Var>>() {
            self.base.cmp(&var.base)
        } else if let Ok(wire) = other.extract::<PyRef<Wire>>() {
            let ltc: &base::Wire<Sort> = &self.base;
            ltc.cmp(wire.base())
        } else {
            return Ok(PyNotImplemented::get(py).into_any());
        };

        Ok(op.matches(ordering).into_pyobject(py)?.into_any())
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
        format!("{:?}", self.base.typed())
    }
}

impl Var {
    pub(crate) fn base(&self) -> &base::Var<Sort> {
        &self.base
    }
}

/// For keying purposes, a python `Var` is its base var: the borrowed view
/// hashes and compares identically (the wrapper's `Hash`/`Eq`/`Ord` all
/// delegate to the base), so maps keyed by `Var` can be probed with a
/// `&base::Var<Sort>` directly.
impl std::borrow::Borrow<base::Var<Sort>> for Var {
    fn borrow(&self) -> &base::Var<Sort> {
        &self.base
    }
}

impl From<base::Var<Sort>> for Var {
    fn from(base: base::Var<Sort>) -> Self {
        Self { base }
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
