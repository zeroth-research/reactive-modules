use crate::itype::IType;
use crate::wire::Wire;
use crate::*;
use pyo3::exceptions::{PyException, PyIndexError, PyTypeError};

#[pyclass(frozen)]
pub(crate) struct Term {
    base: base::Term<theory::python::IType>,
}

#[pymethods]
impl Term {
    #[staticmethod]
    fn function(itype: &Bound<'_, PyAny>, write: &Bound<'_, PyAny>, read: &Bound<'_, PyAny>) -> PyResult<Self> {
        let write: Vec<_> = try_wire_iter_cloned(write)?.collect();
        let resolved = extract_itype(itype, &write)?;
        let read = try_wire_iter_cloned(read)?;

        match base::Term::function(resolved, write.into_iter(), read) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn constant(itype: &Bound<'_, PyAny>, write: &Bound<'_, PyAny>) -> PyResult<Self> {
        let write: Vec<_> = try_wire_iter_cloned(write)?.collect();
        let resolved = extract_itype(itype, &write)?;

        match base::Term::constant(resolved, write.into_iter()) {
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
    fn itype(&self) -> IType {
        self.base.itype().clone().into()
    }

    fn __str__(&self) -> String {
        self.base.to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base)
    }
}

impl Term {
    pub(crate) fn base(&self) -> &base::Term<theory::python::IType> {
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

impl From<base::Term<theory::python::IType>> for Term {
    fn from(base: base::Term<theory::python::IType>) -> Self {
        Self { base }
    }
}

fn extract_itype(
    itype: &Bound<'_, PyAny>,
    _write: &[base::Wire<theory::python::Type>],
) -> PyResult<theory::python::IType> {
    itype
        .extract::<IType>()
        .map(|ops| ops.into())
        .map_err(|_| PyTypeError::new_err("itype must be an IType instance"))
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
    fn base(&self) -> &base::Interface<theory::python::Type> {
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
