use crate::wire::Wire;
use crate::{DType, IType, try_iter_borrow};
use pyo3::exceptions::PyIndexError;
use pyo3::{Bound, IntoPyObject, Py, PyAny, PyRef, PyResult, Python, pyclass, pymethods};

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
#[pyclass(sequence)]
struct AtomInterface {
    atom: Py<Atom>,
    interface: AtomInterfaceType,
}
#[pymethods]
impl AtomInterface {
    fn __str__<'py>(&self, py: Python<'py>) -> String {
        let atom = &self.atom.borrow(py).base;
        match self.interface {
            AtomInterfaceType::Read => atom.read().to_string(),
            AtomInterfaceType::Await => atom.wait().to_string(),
            AtomInterfaceType::Ctrl => atom.ctrl().to_string(),
        }
    }

    fn __getitem__<'py>(&self, index: usize, py: Python<'py>) -> PyResult<Wire> {
        let atom = &self.atom.borrow(py).base;
        let interface = match self.interface {
            AtomInterfaceType::Read => atom.read(),
            AtomInterfaceType::Await => atom.wait(),
            AtomInterfaceType::Ctrl => atom.ctrl(),
        };
        if index < interface.len() {
            Ok(interface.wire(0, index).clone().into())
        } else {
            Err(PyIndexError::new_err("index out of bounds"))
        }
    }

    fn __len__<'py>(&self, py: Python<'py>) -> usize {
        let atom = &self.atom.borrow(py).base;
        match self.interface {
            AtomInterfaceType::Read => atom.read().len(),
            AtomInterfaceType::Await => atom.wait().len(),
            AtomInterfaceType::Ctrl => atom.ctrl().len(),
        }
    }

    fn __eq__<'py>(&self, other: &Bound<'py, PyAny>) -> bool {
        let py = other.py();
        let this = &self.atom.borrow(py).base;
        let other = match try_iter_borrow::<Wire>(other) {
            Ok(other) => other,
            Err(_) => return false,
        };

        let this = match self.interface {
            AtomInterfaceType::Read => this.read(),
            AtomInterfaceType::Await => this.wait(),
            AtomInterfaceType::Ctrl => this.ctrl(),
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
}
