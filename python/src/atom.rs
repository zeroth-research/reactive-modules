use crate::term::TermInterfaceType;
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
    fn read(slf: PyRef<'_, Self>) -> PyResult<AtomInterface> {
        Self::interface(slf, AtomInterfaceType::Read)
    }

    fn ctrl(slf: PyRef<'_, Self>) -> PyResult<AtomInterface> {
        Self::interface(slf, AtomInterfaceType::Ctrl)
    }

    fn wait(slf: PyRef<'_, Self>) -> PyResult<AtomInterface> {
        Self::interface(slf, AtomInterfaceType::Await)
    }

    fn init(slf: Bound<'_, Self>) -> AtomTerms {
        AtomTerms {
            atom: slf.unbind(),
            typ: AtomTermType::Init,
        }
    }

    fn update(slf: Bound<'_, Self>) -> AtomTerms {
        AtomTerms {
            atom: slf.unbind(),
            typ: AtomTermType::Update,
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
    fn base(&self) -> &base::Interface<DType> {
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
enum AtomTermType {
    Init,
    Update,
}

/// An accessor for a Term in an [Atom]
#[pyclass(frozen)]
struct AtomTerm {
    atom: Py<Atom>,
    typ: AtomTermType,
    idx: usize,
}

impl AtomTerm {
    fn get(&self) -> &base::Term<DType, IType> {
        let atom = &self.atom.get().base;
        match self.typ {
            AtomTermType::Init => atom.init().get(self.idx).unwrap(),
            AtomTermType::Update => atom.update().get(self.idx).unwrap(),
        }
    }
}

#[pymethods]
impl AtomTerm {
    fn __str__(&self) -> String {
        self.get().to_string()
    }

    fn itype(&self) -> IType {
        self.get().itype().clone()
    }

    fn read(slf: Bound<'_, Self>) -> AtomTermInterface {
        AtomTermInterface {
            term: slf.unbind(),
            interface: TermInterfaceType::Read,
        }
    }

    fn write(slf: Bound<'_, Self>) -> AtomTermInterface {
        AtomTermInterface {
            term: slf.unbind(),
            interface: TermInterfaceType::Write,
        }
    }
}

#[pyclass(sequence)]
struct AtomTermInterface {
    term: Py<AtomTerm>,
    interface: TermInterfaceType,
}

impl AtomTermInterface {
    fn base(&self) -> &base::Interface<DType> {
        let term = self.term.get().get();
        match self.interface {
            TermInterfaceType::Read => term.read(),
            TermInterfaceType::Write => term.write(),
        }
    }
}

#[pymethods]
impl AtomTermInterface {
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

#[pyclass(sequence)]
struct AtomTerms {
    atom: Py<Atom>,
    typ: AtomTermType,
}

impl AtomTerms {
    fn atom(&self) -> &base::Atom<DType, IType> {
        &self.atom.get().base
    }
}

#[pymethods]
impl AtomTerms {
    fn __getitem__(slf: PyRef<'_, Self>, index: usize) -> PyResult<AtomTerm> {
        let atom = slf.atom();
        match slf.typ {
            AtomTermType::Init => {
                if index < atom.init().len() {
                    return Ok(AtomTerm {
                        atom: slf.atom.clone_ref(slf.py()),
                        typ: slf.typ.clone(),
                        idx: index,
                    });
                }
                Err(PyIndexError::new_err("index out of bounds"))
            }
            AtomTermType::Update => {
                if index < atom.update().len() {
                    return Ok(AtomTerm {
                        atom: slf.atom.clone_ref(slf.py()),
                        typ: slf.typ.clone(),
                        idx: index,
                    });
                }
                Err(PyIndexError::new_err("index out of bounds"))
            }
        }
    }

    fn __len__(&self) -> usize {
        match self.typ {
            AtomTermType::Init => self.atom().init().len(),
            AtomTermType::Update => self.atom().update().len(),
        }
    }
}
