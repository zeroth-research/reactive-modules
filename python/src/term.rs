use crate::wire::Wire;
use crate::{DType, IType, try_iter_borrow};
use pyo3::exceptions::{PyException, PyIndexError};
use pyo3::prelude::*;

#[pyclass]
pub(crate) struct Term {
    base: base::Term<DType, IType>,
}

#[pymethods]
impl Term {
    #[staticmethod]
    fn function(itype: IType, write: &Bound<'_, PyAny>, read: &Bound<'_, PyAny>) -> PyResult<Self> {
        let write = try_iter_borrow::<Wire>(write)?;
        let read = try_iter_borrow::<Wire>(read)?;

        // TODO: make the base take result iterators to avoid unwrap
        let write = write.into_iter().map(Result::unwrap);
        let read = read.into_iter().map(Result::unwrap);

        let write = write.map(|r| r.base().clone());
        let read = read.map(|r| r.base().clone());

        match base::Term::function(itype, write, read) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn constant(itype: IType, write: &Bound<'_, PyAny>) -> PyResult<Self> {
        let write = try_iter_borrow::<Wire>(write)?;
        // TODO: make the base take result iterators to avoid unwrap
        let write = write.into_iter().map(Result::unwrap);
        let write = write.map(|r| r.base().clone());

        match base::Term::constant(itype, write) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[new]
    #[pyo3(signature = (itype, write, read = None))]
    fn new(
        itype: IType,
        write: &Bound<'_, PyAny>,
        read: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        match read {
            Some(read) => Self::function(itype, write, read),
            None => Self::constant(itype, write),
        }
    }

    fn write(slf: PyRef<'_, Self>) -> PyResult<TermInterface> {
        Self::interface(slf, TermInterfaceType::Write)
    }

    fn read(slf: PyRef<'_, Self>) -> PyResult<TermInterface> {
        Self::interface(slf, TermInterfaceType::Read)
    }

    fn __str__(&self) -> String {
        self.base.to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base)
    }
}

impl Term {
    pub(crate) fn base(&self) -> &base::Term<DType, IType> {
        &self.base
    }

    fn interface(slf: PyRef<'_, Self>, titype: TermInterfaceType) -> PyResult<TermInterface> {
        let py = slf.py();
        Ok(TermInterface {
            term: slf.into_pyobject(py)?.unbind(),
            interface: titype,
        })
    }
}

impl From<base::Term<DType, IType>> for Term {
    fn from(base: base::Term<DType, IType>) -> Self {
        Self { base }
    }
}

#[derive(Clone)]
pub(crate) enum TermInterfaceType {
    Read,
    Write,
}
#[pyclass(sequence)]
struct TermInterface {
    term: Py<Term>,
    interface: TermInterfaceType,
}

#[pymethods]
impl TermInterface {
    fn __str__<'py>(&self, py: Python<'py>) -> String {
        let base_term = &self.term.borrow(py).base;
        match self.interface {
            TermInterfaceType::Read => base_term.read().to_string(),
            TermInterfaceType::Write => base_term.write().to_string(),
        }
    }

    fn __eq__<'py>(&self, other: &Bound<'py, PyAny>) -> bool {
        let py = other.py();
        let this = &self.term.borrow(py).base;
        let other = match try_iter_borrow::<Wire>(other) {
            Ok(other) => other,
            Err(_) => return false,
        };

        let this = match self.interface {
            TermInterfaceType::Read => this.read(),
            TermInterfaceType::Write => this.write(),
        };

        let mut this = this.wires();
        let mut other = other.into_iter();
        loop {
            match (this.next(), other.next()) {
                (Some(this), Some(Ok(other))) => {
                    if this != other.base() {
                        return false;
                    }
                }
                (None, None) => return true,
                _ => return false,
            }
        }
    }

    fn __getitem__<'py>(&self, index: usize, py: Python<'py>) -> PyResult<Wire> {
        let term = &self.term.borrow(py).base;
        let interface = match self.interface {
            TermInterfaceType::Read => term.read(),
            TermInterfaceType::Write => term.write(),
        };
        if index < interface.len() {
            Ok(interface.wire(0, index).clone().into())
        } else {
            Err(PyIndexError::new_err("index out of bounds"))
        }
    }

    fn __len__<'py>(&self, py: Python<'py>) -> usize {
        let term = &self.term.borrow(py).base;
        match self.interface {
            TermInterfaceType::Read => term.read().len(),
            TermInterfaceType::Write => term.write().len(),
        }
    }
}
