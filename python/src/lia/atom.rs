use std::fmt;
use theory::{self, bool, lia};

use super::Wire;
use crate::{IType, try_iter_borrow};
use pyo3::exceptions::PyIndexError;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;

#[pyclass(frozen)]
pub struct Atom {
    base: base::Atom<lia::LIA>,
}

impl From<base::Atom<lia::LIA>> for Atom {
    fn from(base: base::Atom<lia::LIA>) -> Self {
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
}

#[pyclass(sequence)]
struct AtomInterface {
    atom: Py<Atom>,
    interface: AtomInterfaceType,
}

impl AtomInterface {
    fn base(&self) -> &base::Interface<lia::Type> {
        let atom = &self.atom.get().base;
        match self.interface {
            AtomInterfaceType::Read => atom.read(),
            AtomInterfaceType::Await => atom.wait(),
            AtomInterfaceType::Ctrl => atom.ctrl(),
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
    fn base(&self) -> &base::Block<lia::LIA> {
        let atom = &self.atom.get().base;
        match self.block {
            BlockType::Init => atom.init(),
            BlockType::Update => atom.update(),
        }
    }
}

impl super::HasTermAt for AtomBlock {
    fn term_at(&self, idx: usize) -> Option<&base::Term<lia::LIA>> {
        self.base().get(idx)
    }
}

impl super::ReadWriteIntf for AtomBlock {
    fn interface(&self, is_read: bool) -> &base::Interface<lia::Type> {
        if is_read { self.base().read() } else { self.base().write() }
    }
}

#[pymethods]
impl AtomBlock {
    fn __repr__(&self) -> String {
        format!("{:?}", self.base())
    }

    fn read(slf: Bound<'_, Self>) -> AtomBlockInterface {
        AtomBlockInterface(super::ReadWriteInterface { owner: slf.unbind(), is_read: true })
    }

    fn write(slf: Bound<'_, Self>) -> AtomBlockInterface {
        AtomBlockInterface(super::ReadWriteInterface { owner: slf.unbind(), is_read: false })
    }

    fn __getitem__(slf: Bound<'_, Self>, index: usize) -> PyResult<TermRef> {
        if slf.get().base().get(index).is_none() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        Ok(TermRef(super::TermAt { owner: slf.unbind(), idx: index }))
    }

    fn __len__(&self) -> usize {
        self.base().len()
    }
}

#[pyclass(sequence)]
struct AtomBlockInterface(super::ReadWriteInterface<AtomBlock>);

#[pymethods]
impl AtomBlockInterface {
    fn __str__(&self) -> String {
        self.0.str()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        let other = match try_iter_borrow::<Wire>(other) {
            Ok(other) => other,
            Err(_) => return false,
        };

        let mut this = self.0.base().wires();
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
        self.0.get_item(index)
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }
}

/// Zero-copy accessor for a term in an atom
#[pyclass(frozen)]
pub struct TermRef(super::TermAt<AtomBlock>);

impl super::ReadWriteIntf for TermRef {
    fn interface(&self, is_read: bool) -> &base::Interface<lia::Type> {
        self.0.interface(is_read)
    }
}

#[pyclass(sequence)]
struct TermRefInterface(super::ReadWriteInterface<TermRef>);

#[pymethods]
impl TermRef {
    #[getter]
    fn read(slf: Bound<'_, Self>) -> TermRefInterface {
        TermRefInterface(super::ReadWriteInterface { owner: slf.unbind(), is_read: true })
    }

    #[getter]
    fn write(slf: Bound<'_, Self>) -> TermRefInterface {
        TermRefInterface(super::ReadWriteInterface { owner: slf.unbind(), is_read: false })
    }

    #[getter]
    fn itype(&self) -> IType {
        self.0.itype()
    }

    fn __repr__(&self) -> String {
        self.0.repr()
    }
}

#[pymethods]
impl TermRefInterface {
    fn __getitem__(&self, index: usize) -> PyResult<Wire> {
        self.0.get_item(index)
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __str__(&self) -> String {
        self.0.str()
    }
}