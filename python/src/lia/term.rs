use super::{Type, Wire};
use crate::{IType, try_iter_borrow};
use pyo3::exceptions::{PyException, PyIndexError, PyValueError};
use pyo3::prelude::*;
use std::fmt;
use theory::{self, bool, lia};

#[pyclass(frozen)]
pub struct Term {
    base: base::Term<lia::LIA>,
}

impl super::ReadWriteIntf for Term {
    fn interface(&self, is_read: bool) -> &base::Interface<lia::Type> {
        if is_read { self.base.read() } else { self.base.write() }
    }
}

fn try_wire_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = base::Wire<lia::Type>>>
where
    for<'py> Wire: pyo3::FromPyObject<'py>,
    base::Wire<lia::Type>: From<Wire>,
{
    seq.iter()?
        .map(|item| item?.extract::<Wire>().map(Into::into))
        .collect::<PyResult<Vec<_>>>()
        .map(Vec::into_iter)
}

#[pymethods]
impl Term {
    #[staticmethod]
    fn function(itype: IType, write: &Bound<'_, PyAny>, read: &Bound<'_, PyAny>) -> PyResult<Self> {
        let write = try_wire_iter_cloned(write)?;
        let read = try_wire_iter_cloned(read)?;

        let op = match itype.try_into() {
            Ok(o) => o,
            Err(_) => {
                return Err(PyException::new_err("IType is not from LIA"));
            }
        };
        match base::Term::function(op, write, read) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn constant(itype: IType, write: &Bound<'_, PyAny>) -> PyResult<Self> {
        let write = try_wire_iter_cloned(write)?;

        let op = match itype.try_into() {
            Ok(o) => o,
            Err(_) => {
                return Err(PyException::new_err("IType is not from LIA"));
            }
        };

        match base::Term::constant(op, write) {
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

    #[getter]
    fn write(slf: PyRef<'_, Self>) -> PyResult<TermInterface> {
        Self::interface(slf, false)
    }

    #[getter]
    fn read(slf: PyRef<'_, Self>) -> PyResult<TermInterface> {
        Self::interface(slf, true)
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
    pub(crate) fn base(&self) -> &base::Term<lia::LIA> {
        &self.base
    }

    fn interface(slf: PyRef<'_, Self>, is_read: bool) -> PyResult<TermInterface> {
        let py = slf.py();
        let owner = slf.into_pyobject(py)?.unbind();
        Ok(TermInterface(super::ReadWriteInterface { owner, is_read }))
    }
}

impl From<base::Term<lia::LIA>> for Term {
    fn from(base: base::Term<lia::LIA>) -> Self {
        Self { base }
    }
}

impl From<Type> for lia::Type {
    fn from(dt: Type) -> Self {
        dt.0
    }
}

#[pyclass(sequence)]
struct TermInterface(super::ReadWriteInterface<Term>);

#[pymethods]
impl TermInterface {
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
