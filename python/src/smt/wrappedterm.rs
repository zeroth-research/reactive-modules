use pyo3::prelude::*;

use crate::smt::PyVal;
use smt::itype::IType;

#[pyclass]
pub struct WrappedTerm {
    pub op: IType,
    pub reads: Vec<PyVal>,
    pub writes: Vec<PyVal>,
}

#[pymethods]
impl WrappedTerm {
    #[new]
    fn new(op: &str, reads: Vec<Py<PyVal>>, writes: Vec<Py<PyVal>>) -> Self {
        // here we copy the PyVal created (and owned) by Python so that we can
        // work with them independently of Python.  If the copying becomes
        // expensive (IHMO unlikely), we can keep `Py<PyVal>` in `WrappedTerm`
        // instead -- the memory would be owned by Python and we would have only
        // references to the values.
        Self {
            op: op.parse().expect("Invalid instruction"),
            reads: reads.iter().map(|item| item.get().clone()).collect(),
            writes: writes.iter().map(|item| item.get().clone()).collect(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "WrappedTerm({:?}, {:?}, {:?})",
            self.op, self.reads, self.writes
        )
    }

    fn __str__(&self) -> String {
        format!("{:?} - {:?} -> {:?}", self.reads, self.op, self.writes)
    }

    fn print(&self) {
        // FIXME: use `colored` crate
        println!(
            "\x1b[1;34m{:?}\x1b[0m - \x1b[1;31m{:?}\x1b[0m -> \x1b[1;32m{:?}\x1b[0m",
            self.reads, self.op, self.writes
        )
    }
}

// It is safe to share WrappedTerm for the same reasons as for PyTensor
unsafe impl Sync for WrappedTerm {}
