use crate::wire::Wire;
use crate::*;
use pyo3::exceptions::{PyException, PyIndexError};

#[pyclass(frozen)]
pub(crate) struct Term {
    base: base::Term<theory::any::Any>,
}

#[pymethods]
impl Term {
    #[staticmethod]
    fn function(
        itype: &Bound<'_, PyAny>,
        write: &Bound<'_, PyAny>,
        read: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let itype = itype.extract()?;
        let write = try_wire_iter_cloned(write)?;
        let read = try_wire_iter_cloned(read)?;

        match base::Term::function(itype, write, read) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn constant(itype: &Bound<'_, PyAny>, write: &Bound<'_, PyAny>) -> PyResult<Self> {
        let itype = itype.extract()?;
        let write = try_wire_iter_cloned(write)?;

        match base::Term::constant(itype, write) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[new]
    #[pyo3(signature = (itype, write, read = None))]
    fn new(
        itype: &Bound<'_, PyAny>,
        write: &Bound<'_, PyAny>,
        read: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        match read {
            Some(read) => Self::function(itype, write, read),
            None => Self::constant(itype, write),
        }
    }

    #[getter]
    fn write(slf: PyRef<'_, Self>) -> PyResult<TermInterface> {
        Self::interface(slf, TermInterfaceType::Write)
    }

    #[getter]
    fn read(slf: PyRef<'_, Self>) -> PyResult<TermInterface> {
        Self::interface(slf, TermInterfaceType::Read)
    }

    #[getter]
    fn itype(&self) -> theory::any::Any {
        self.base.itype().clone()
    }

    fn __str__(&self) -> String {
        self.base.to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base)
    }
}

impl Term {
    pub(crate) fn base(&self) -> &base::Term<theory::any::Any> {
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

impl From<base::Term<theory::any::Any>> for Term {
    fn from(base: base::Term<theory::any::Any>) -> Self {
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
    fn base(&self) -> &base::Interface<theory::any::Type> {
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
