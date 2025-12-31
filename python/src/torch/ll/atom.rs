use super::wire::Wire;
use super::{DType, IType};
use pyo3::{IntoPyObject, Py, PyRef, PyRefMut, PyResult, Python, pyclass, pymethods};

#[pyclass]
pub(crate) struct Atom {
    base: base::Atom<DType, IType>,
}

impl From<base::Atom<DType, IType>> for Atom {
    fn from(base: base::Atom<DType, IType>) -> Self {
        Self { base }
    }
}

#[pymethods]
impl Atom {
    fn read(slf: PyRef<'_, Self>) -> PyResult<AtomInterface> {
        Self::interface(slf, AtomInterfaceType::Read)
    }

    fn ctrl(slf: PyRef<'_, Self>) -> PyResult<AtomInterface> {
        Self::interface(slf, AtomInterfaceType::Ctrl)
    }

    fn wait(slf: PyRef<'_, Self>) -> PyResult<AtomInterface> {
        Self::interface(slf, AtomInterfaceType::Await)
    }

    fn __str__(&self) -> String {
        self.base.to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base)
    }
}

impl Atom {
    fn interface(slf: PyRef<'_, Self>, interface: AtomInterfaceType) -> PyResult<AtomInterface> {
        let py = slf.py();
        let atom = slf.into_pyobject(py)?.unbind();
        Ok(AtomInterface { atom, interface })
    }
}

#[derive(Clone)]
enum AtomInterfaceType {
    Read,
    Ctrl,
    Await,
}
#[pyclass]
struct AtomInterface {
    atom: Py<Atom>,
    interface: AtomInterfaceType,
}
#[pymethods]
impl AtomInterface {
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Py<AtomInterfaceIter>> {
        let iter = AtomInterfaceIter {
            atom: self.atom.clone_ref(py),
            interface: self.interface.clone(),
            index: 0,
        };
        Py::new(py, iter)
    }

    fn __str__<'py>(&self, py: Python<'py>) -> String {
        let atom = &self.atom.borrow(py).base;
        match self.interface {
            AtomInterfaceType::Read => atom.read().to_string(),
            AtomInterfaceType::Await => atom.wait().to_string(),
            AtomInterfaceType::Ctrl => atom.ctrl().to_string(),
        }
    }
}

#[pyclass]
struct AtomInterfaceIter {
    atom: Py<Atom>,
    interface: AtomInterfaceType,
    index: usize,
}

#[pymethods]
impl AtomInterfaceIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Wire> {
        let result = {
            let atom = &slf.atom.borrow(slf.py()).base;
            let interface = match slf.interface {
                AtomInterfaceType::Read => atom.read(),
                AtomInterfaceType::Await => atom.wait(),
                AtomInterfaceType::Ctrl => atom.ctrl(),
            };
            if slf.index < interface.len() {
                Some(interface.wire(0, slf.index).clone().into())
            } else {
                None
            }
        };
        slf.index += 1;
        result
    }
}
