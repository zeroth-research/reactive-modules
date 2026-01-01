use super::wire::Wire;
use super::{DType, IType, try_iter_borrow};
use pyo3::exceptions::{PyException, PyIndexError};
use pyo3::prelude::*;

#[pyclass(frozen)]
pub struct Term {
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

impl TermInterface {
    fn base(&self) -> &base::Interface<DType> {
        let base = &self.term.get().base;
        match self.interface {
            TermInterfaceType::Read => base.read(),
            TermInterfaceType::Write => base.write(),
        }
    }
}

#[pymethods]
impl TermInterface {
    fn __str__(&self) -> String {
        self.base().to_string()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        let other = match try_iter_borrow::<Wire>(other) {
            Ok(other) => other,
            Err(_) => return false,
        };

        let mut this = self.base().wires();
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

    fn __getitem__(&self, index: usize) -> PyResult<Wire> {
        let item = self.base().wire(0, index);
        item.and_then(|w| Some(w.clone().into()))
            .ok_or(PyIndexError::new_err("index out of bounds"))
    }

    fn __len__(&self) -> usize {
        self.base().len()
    }
}
