use crate::bv::BV;
use crate::lia::LIA;
use crate::lra::LRA;
use crate::{Combinatorial, Differential, Sequential, bv, lia, lra};
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::PyString;
#[cfg(feature = "pyo3")]
use pyo3::{Bound, FromPyObject, PyAny, PyResult, pyclass};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "pyo3", pyclass(frozen, eq))]
pub enum Sort {
    Bool([usize; 2]),
    Real([usize; 2]),
    Int([usize; 2]),
    BitVec(usize, [usize; 2]),
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Bool(s) => write!(f, "Bool({}, {})", s[0], s[1]),
            Sort::Real(s) => write!(f, "Real({}, {})", s[0], s[1]),
            Sort::Int(s) => write!(f, "Int({}, {})", s[0], s[1]),
            Sort::BitVec(bw, s) => write!(f, "BV<{}>({}, {})", bw, s[0], s[1]),
        }
    }
}

impl From<bv::Sort> for Sort {
    fn from(value: bv::Sort) -> Self {
        match value {
            bv::Sort::BV(bw, shape) => Sort::BitVec(bw, shape),
        }
    }
}

impl From<lia::Sort> for Sort {
    fn from(value: lia::Sort) -> Self {
        match value {
            lia::Sort::Int(shape) => Sort::Int(shape),
            lia::Sort::Bool(shape) => Sort::Bool(shape),
        }
    }
}

impl From<lra::Sort> for Sort {
    fn from(value: lra::Sort) -> Self {
        match value {
            lra::Sort::Bool(shape) => Sort::Bool(shape),
            lra::Sort::Real(shape) => Sort::Real(shape),
        }
    }
}

impl<E> TryFrom<Result<Sort, E>> for bv::Sort {
    type Error = String;
    fn try_from(value: Result<Sort, E>) -> Result<Self, Self::Error> {
        let sort = value.map_err(|_| "invalid cast")?;
        sort.try_into()
    }
}

impl<E> TryFrom<Result<Sort, E>> for lia::Sort {
    type Error = String;
    fn try_from(value: Result<Sort, E>) -> Result<Self, Self::Error> {
        let sort = value.map_err(|_| "invalid cast")?;
        sort.try_into()
    }
}

impl<E> TryFrom<Result<Sort, E>> for lra::Sort {
    type Error = String;
    fn try_from(value: Result<Sort, E>) -> Result<Self, Self::Error> {
        let sort = value.map_err(|_| "invalid cast")?;
        sort.try_into()
    }
}

impl TryFrom<Sort> for bv::Sort {
    type Error = String;

    fn try_from(value: Sort) -> Result<Self, Self::Error> {
        match value {
            Sort::BitVec(bw, shape) => Ok(bv::Sort::BV(bw, shape)),
            _ => Err("invalid cast".to_string()),
        }
    }
}

impl TryFrom<Sort> for lia::Sort {
    type Error = String;

    fn try_from(value: Sort) -> Result<Self, Self::Error> {
        match value {
            Sort::Bool(shape) => Ok(lia::Sort::Bool(shape)),
            Sort::Int(shape) => Ok(lia::Sort::Int(shape)),
            _ => Err("invalid cast".to_string()),
        }
    }
}

impl TryFrom<Sort> for lra::Sort {
    type Error = String;

    fn try_from(value: Sort) -> Result<Self, Self::Error> {
        match value {
            Sort::Bool(shape) => Ok(lra::Sort::Bool(shape)),
            Sort::Real(shape) => Ok(lra::Sort::Real(shape)),
            _ => Err("invalid cast".to_string()),
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub enum Any {
    HAVOC,
    SKIP,
    ZERO,
    LRA(LRA),
    LIA(LIA),
    BV(BV),
}

impl fmt::Display for Any {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Any::HAVOC => write!(f, "HAVOC"),
            Any::SKIP => write!(f, "SKIP"),
            Any::ZERO => write!(f, "ZERO"),
            Any::LRA(op) => write!(f, "{}", op),
            Any::LIA(op) => write!(f, "{}", op),
            Any::BV(op) => write!(f, "{}", op),
        }
    }
}

impl crate::Theory for Any {
    type Sort = Sort;
    const NAME: &'static str = "Any";

    fn check<R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<Sort>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        let mut read = read.into_iter().map(TryInto::try_into);
        let mut write = write.into_iter().map(TryInto::try_into);
        match &self {
            Any::HAVOC | Any::ZERO => match read.next() {
                None => Ok(()),
                _ => Err(format!("{} expects no read", self)),
            },
            Any::SKIP => loop {
                match (read.next(), write.next()) {
                    (Some(Ok(a)), Some(Ok(b))) if a == b => continue,
                    (None, None) => return Ok(()),
                    _ => return Err("SKIP expects matching read and write".to_string()),
                }
            },
            Any::LRA(itype) => itype.check(read, write),
            Any::LIA(itype) => itype.check(read, write),
            Any::BV(itype) => itype.check(read, write),
        }
    }
}

impl Combinatorial for Any {
    const HAVOC: Self = Any::HAVOC;
}

impl Sequential for Any {
    const SKIP: Self = Any::SKIP;
}

impl Differential for Any {
    const ZERO: Self = Any::ZERO;
}

#[cfg(feature = "pyo3")]
impl<'py> FromPyObject<'py> for Any {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(a) = obj.extract::<LRA>() {
            return Ok(Any::LRA(a));
        }
        if let Ok(a) = obj.extract::<LIA>() {
            return Ok(Any::LIA(a));
        }
        if let Ok(a) = obj.extract::<BV>() {
            return Ok(Any::BV(a));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected one of LRA, LIA, or BV",
        ))
    }
}

#[cfg(feature = "pyo3")]
impl<'py> IntoPyObject<'py> for Any {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            Any::HAVOC => Ok(PyString::new(py, "HAVOC").into_any()),
            Any::SKIP => Ok(PyString::new(py, "SKIP").into_any()),
            Any::ZERO => Ok(PyString::new(py, "ZERO").into_any()),
            Any::LRA(a) => a.into_pyobject(py).map(Bound::into_any),
            Any::LIA(a) => a.into_pyobject(py).map(Bound::into_any),
            Any::BV(a) => a.into_pyobject(py).map(Bound::into_any),
        }
    }
}
