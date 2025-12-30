use super::wire::Wire;
use super::{DType, IType, try_iter_borrow};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

#[pyclass]
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
enum TermInterfaceType {
    Read,
    Write,
}
#[pyclass]
struct TermInterface {
    term: Py<Term>,
    interface: TermInterfaceType,
}
#[pymethods]
impl TermInterface {
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Py<TermInterfaceIter>> {
        Py::new(
            py,
            TermInterfaceIter {
                module: self.term.clone_ref(py),
                interface: self.interface.clone(),
                index: 0,
            },
        )
    }

    fn __str__(slf: PyRef<'_, Self>) -> String {
        let base_term = &slf.term.borrow(slf.py()).base;
        match slf.interface {
            TermInterfaceType::Read => base_term.read().to_string(),
            TermInterfaceType::Write => base_term.write().to_string(),
        }
    }

    fn __eq__(&self, other: PyRef<'_, TermInterface>) -> bool {
        let py = other.py();
        let this_term = self.term.borrow(py);
        let other_term = other.term.borrow(py);

        let this_interface = match self.interface {
            TermInterfaceType::Read => this_term.base.read(),
            TermInterfaceType::Write => this_term.base.write(),
        };
        let other_interface = match other.interface {
            TermInterfaceType::Read => other_term.base.read(),
            TermInterfaceType::Write => other_term.base.write(),
        };

        this_interface == other_interface
    }
}

#[pyclass]
struct TermInterfaceIter {
    module: Py<Term>,
    interface: TermInterfaceType,
    index: usize,
}

#[pymethods]
impl TermInterfaceIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Wire> {
        let result: Option<Wire> = {
            let base_module = &slf.module.borrow(slf.py()).base;
            let base_interface = match slf.interface {
                TermInterfaceType::Read => base_module.read(),
                TermInterfaceType::Write => base_module.write(),
            };
            (slf.index < base_interface.len())
                .then(|| base_interface.entry(slf.index)[0].clone().into())
        };
        slf.index += 1;
        result
    }
}
