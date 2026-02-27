use crate::DType;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[pyclass(frozen)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct Wire {
    base: base::Wire<DType>,
}

static NEXT: AtomicUsize = AtomicUsize::new(0);

#[pymethods]
impl Wire {
    #[new]
    pub(crate) fn new(dtype: DType) -> Self {
        let base = base::Wire::new(NEXT.fetch_add(1, Ordering::Relaxed), dtype);
        if base.id() == usize::MAX {
            panic!("wire id overflow");
        }
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

    // Wires should be identified by their ID.
    // If there are two wires with same IDs and different types
    // that are used by the same code, it is an error.
    // So hash only by ID
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
