use crate::term::{Term, TermInterfaceType};
use crate::wire::Wire;
use crate::{DType, IType, try_iter_borrow};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

#[pyclass(frozen)]
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
    #[getter]
    fn read(slf: PyRef<'_, Self>) -> PyResult<AtomInterface> {
        Self::interface(slf, AtomInterfaceType::Read)
    }

    #[getter]
    fn ctrl(slf: PyRef<'_, Self>) -> PyResult<AtomInterface> {
        Self::interface(slf, AtomInterfaceType::Ctrl)
    }

    #[getter]
    fn wait(slf: PyRef<'_, Self>) -> PyResult<AtomInterface> {
        Self::interface(slf, AtomInterfaceType::Await)
    }

    #[getter]
    fn param(slf: PyRef<'_, Self>) -> PyResult<AtomInterface> {
        Self::interface(slf, AtomInterfaceType::Param)
    }

    #[getter]
    fn init(slf: Bound<'_, Self>) -> AtomBlock {
        AtomBlock {
            atom: slf.unbind(),
            block: BlockType::Init,
        }
    }

    #[getter]
    fn update(slf: Bound<'_, Self>) -> AtomBlock {
        AtomBlock {
            atom: slf.unbind(),
            block: BlockType::Update,
        }
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
    Param,
}

#[pyclass(sequence)]
struct AtomInterface {
    atom: Py<Atom>,
    interface: AtomInterfaceType,
}

impl AtomInterface {
    fn base(&self) -> &base::Interface<DType> {
        let atom = &self.atom.get().base;
        match self.interface {
            AtomInterfaceType::Read => atom.read(),
            AtomInterfaceType::Await => atom.wait(),
            AtomInterfaceType::Ctrl => atom.ctrl(),
            AtomInterfaceType::Param => atom.param(),
        }
    }
}
#[pymethods]
impl AtomInterface {
    fn __str__(&self) -> String {
        self.base().to_string()
    }

    fn __getitem__(&self, index: usize) -> PyResult<Wire> {
        let item = self.base().wire(0, index);
        item.and_then(|w| Some(w.clone().into()))
            .ok_or(PyIndexError::new_err("index out of bounds"))
    }

    fn __len__(&self) -> usize {
        self.base().len()
    }

    fn __eq__<'py>(&self, other: &Bound<'py, PyAny>) -> bool {
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
}

#[derive(Clone)]
enum BlockType {
    Init,
    Update,
}

#[pyclass(sequence, frozen)]
pub(crate) struct AtomBlock {
    atom: Py<Atom>,
    block: BlockType,
}

impl AtomBlock {
    fn base(&self) -> &base::Block<DType, IType> {
        let atom = &self.atom.get().base;
        match self.block {
            BlockType::Init => atom.init(),
            BlockType::Update => atom.update(),
        }
    }
}

#[pymethods]
impl AtomBlock {
    // requires display in base - do you need that now?
    // fn __str__(&self) -> String {
    //     self.base().to_string()
    // }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base())
    }

    fn read(slf: Bound<'_, Self>) -> AtomBlockInterface {
        AtomBlockInterface {
            block: slf.unbind(),
            interface: TermInterfaceType::Read,
        }
    }

    fn write(slf: Bound<'_, Self>) -> AtomBlockInterface {
        AtomBlockInterface {
            block: slf.unbind(),
            interface: TermInterfaceType::Write,
        }
    }

    fn __getitem__(&self, index: usize) -> PyResult<Term> {
        let result = self.base().get(index).map(Clone::clone).map(Into::into);
        result.ok_or(PyIndexError::new_err("index out of bounds"))
    }

    fn __len__(&self) -> usize {
        self.base().len()
    }
}

#[pyclass(sequence)]
struct AtomBlockInterface {
    block: Py<AtomBlock>,
    interface: TermInterfaceType,
}

impl AtomBlockInterface {
    fn base(&self) -> &base::Interface<DType> {
        let term = self.block.get().base();
        match self.interface {
            TermInterfaceType::Read => term.read(),
            TermInterfaceType::Write => term.write(),
        }
    }
}

#[pymethods]
impl AtomBlockInterface {
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
