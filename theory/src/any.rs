use crate::Combinatorial as _;
use crate::Differential as _;
use crate::Sequential as _;
use crate::bv::BV;
use crate::lia::LIA;
use crate::lra::LRA;
use crate::{Theory, bv, lia, lra};
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

#[derive(Debug, Clone)]
pub enum Sequential {
    SKIP,
    BV(BV),
    LRA(LRA),
    LIA(LIA),
}

fn check_skip<R, W, E>(read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    loop {
        match (read.next(), write.next()) {
            (Some(Ok(a)), Some(Ok(b))) if a == b => continue,
            (None, None) => return Ok(()),
            _ => return Err("SKIP expects matching read and write".to_string()),
        }
    }
}

impl Theory for Sequential {
    type Sort = Sort;
    const NAME: &'static str = "Sequential";
    fn check<R, W, S>(&self, read: R, write: W) -> Result<(), String>
    where
        S: TryInto<Sort>,
        W: IntoIterator<Item = S>,
        R: IntoIterator<Item = S>,
    {
        let read = read.into_iter().map(TryInto::try_into);
        let write = write.into_iter().map(TryInto::try_into);
        match &self {
            Sequential::SKIP => check_skip(read, write),
            Sequential::BV(itype) => itype.check(read, write),
            Sequential::LRA(itype) => itype.check(read, write),
            Sequential::LIA(itype) => itype.check(read, write),
        }
    }
}

impl crate::Sequential for Sequential {
    const SKIP: Self = Sequential::SKIP;
}

impl fmt::Display for Sequential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sequential::SKIP => write!(f, "SKIP"),
            Sequential::BV(bv) => write!(f, "{}", bv),
            Sequential::LRA(lra) => write!(f, "{}", lra),
            Sequential::LIA(lia) => write!(f, "{}", lia),
        }
    }
}

impl TryFrom<Any> for Sequential {
    type Error = String;
    fn try_from(any: Any) -> Result<Self, Self::Error> {
        match any {
            Any::SKIP => Ok(Sequential::SKIP),
            Any::BV(bv) => Ok(Sequential::BV(bv)),
            Any::LIA(lia) => Ok(Sequential::LIA(lia)),
            Any::LRA(lra) => Ok(Sequential::LRA(lra)),
            _ => Err(format!("{} is not in a sequential theory", any)),
        }
    }
}

impl From<Sequential> for Any {
    fn from(sequential: Sequential) -> Self {
        match sequential {
            Sequential::SKIP => Any::SKIP,
            Sequential::BV(bv) => Any::BV(bv),
            Sequential::LRA(lra) => Any::LRA(lra),
            Sequential::LIA(lia) => Any::LIA(lia),
        }
    }
}

impl TryFrom<Sequential> for BV {
    type Error = String;
    fn try_from(value: Sequential) -> Result<Self, Self::Error> {
        match value {
            Sequential::SKIP => Ok(BV::SKIP),
            Sequential::BV(bv) => Ok(bv),
            _ => Err("invalid cast".to_string()),
        }
    }
}

impl TryFrom<Sequential> for LIA {
    type Error = String;
    fn try_from(value: Sequential) -> Result<Self, Self::Error> {
        match value {
            Sequential::SKIP => Ok(LIA::SKIP),
            Sequential::LIA(lia) => Ok(lia),
            _ => Err("invalid cast".to_string()),
        }
    }
}

impl TryFrom<Sequential> for LRA {
    type Error = String;
    fn try_from(value: Sequential) -> Result<Self, Self::Error> {
        match value {
            Sequential::SKIP => Ok(LRA::SKIP),
            Sequential::LRA(lra) => Ok(lra),
            _ => Err("invalid cast".to_string()),
        }
    }
}

// We interpret Combinatorial as Sequential + SKIP, but this is specific to this implementation.
// This shall be generalised into a new Sequential + Combinatorial theory, if needed.
#[derive(Debug, Clone)]
pub enum Combinatorial {
    HAVOC,
    SKIP,
    BV(BV),
    LRA(LRA),
    LIA(LIA),
}

fn check_havoc<R, E>(read: R) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
{
    let mut read = read.into_iter();
    match read.next() {
        None => Ok(()),
        _ => Err("HAVOC expects no read wires".to_string()),
    }
}

impl Theory for Combinatorial {
    type Sort = Sort;
    const NAME: &'static str = "Combinatorial";
    fn check<R, W, S>(&self, read: R, write: W) -> Result<(), String>
    where
        S: TryInto<Sort>,
        W: IntoIterator<Item = S>,
        R: IntoIterator<Item = S>,
    {
        let read = read.into_iter().map(TryInto::try_into);
        let write = write.into_iter().map(TryInto::try_into);
        match &self {
            Combinatorial::HAVOC => check_havoc(read),
            Combinatorial::SKIP => check_skip(read, write),
            Combinatorial::BV(itype) => itype.check(read, write),
            Combinatorial::LRA(itype) => itype.check(read, write),
            Combinatorial::LIA(itype) => itype.check(read, write),
        }
    }
}

impl crate::Combinatorial for Combinatorial {
    const HAVOC: Self = Combinatorial::HAVOC;
}

impl fmt::Display for Combinatorial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Combinatorial::HAVOC => write!(f, "HAVOC"),
            Combinatorial::SKIP => write!(f, "SKIP"),
            Combinatorial::BV(bv) => write!(f, "{}", bv),
            Combinatorial::LRA(lra) => write!(f, "{}", lra),
            Combinatorial::LIA(lia) => write!(f, "{}", lia),
        }
    }
}

impl TryFrom<Any> for Combinatorial {
    type Error = String;
    fn try_from(any: Any) -> Result<Self, Self::Error> {
        match any {
            Any::HAVOC => Ok(Combinatorial::HAVOC),
            Any::SKIP => Ok(Combinatorial::SKIP),
            Any::BV(bv) => Ok(Combinatorial::BV(bv)),
            Any::LIA(lia) => Ok(Combinatorial::LIA(lia)),
            Any::LRA(lra) => Ok(Combinatorial::LRA(lra)),
            _ => Err(format!("{} is not in a combinatorial theory", any)),
        }
    }
}

impl From<Combinatorial> for Any {
    fn from(combinatorial: Combinatorial) -> Self {
        match combinatorial {
            Combinatorial::HAVOC => Any::HAVOC,
            Combinatorial::SKIP => Any::SKIP,
            Combinatorial::BV(bv) => Any::BV(bv),
            Combinatorial::LRA(lra) => Any::LRA(lra),
            Combinatorial::LIA(lia) => Any::LIA(lia),
        }
    }
}

impl From<Combinatorial> for Sequential {
    fn from(combinatorial: Combinatorial) -> Self {
        match combinatorial {
            Combinatorial::HAVOC => panic!("Attempted to cast HAVOC to Sequential"),
            Combinatorial::SKIP => Sequential::SKIP,
            Combinatorial::BV(bv) => Sequential::BV(bv),
            Combinatorial::LRA(lra) => Sequential::LRA(lra),
            Combinatorial::LIA(lia) => Sequential::LIA(lia),
        }
    }
}

impl TryFrom<Combinatorial> for BV {
    type Error = String;
    fn try_from(value: Combinatorial) -> Result<Self, Self::Error> {
        match value {
            Combinatorial::HAVOC => Ok(BV::HAVOC),
            Combinatorial::BV(bv) => Ok(bv),
            _ => Err("invalid cast".to_string()),
        }
    }
}

// impl TryFrom<Combinatorial> for LIA {
//     type Error = String;
//     fn try_from(value: Combinatorial) -> Result<Self, Self::Error> {
//         match value {
//             Combinatorial::HAVOC => Ok(LIA::HAVOC),
//             Combinatorial::LIA(lia) => Ok(lia),
//             _ => Err("invalid cast".to_string()),
//         }
//     }
// }
//
// impl TryFrom<Combinatorial> for LRA {
//     type Error = String;
//     fn try_from(value: Combinatorial) -> Result<Self, Self::Error> {
//         match value {
//             Combinatorial::HAVOC => Ok(LRA::HAVOC),
//             Combinatorial::LRA(lra) => Ok(lra),
//             _ => Err("invalid cast".to_string()),
//         }
//     }
// }

#[derive(Debug, Clone)]
pub enum Differential {
    ZERO,
    LRA(LRA),
}

pub(crate) fn check_zero<R>(read: R) -> Result<(), String>
where
    R: IntoIterator,
{
    let mut read = read.into_iter();
    match read.next() {
        None => Ok(()),
        _ => Err("ZERO expects no read wires".to_string()),
    }
}

impl Theory for Differential {
    type Sort = Sort;
    const NAME: &'static str = "Differential";
    fn check<R, W, S>(&self, read: R, write: W) -> Result<(), String>
    where
        S: TryInto<Sort>,
        W: IntoIterator<Item = S>,
        R: IntoIterator<Item = S>,
    {
        let read = read.into_iter().map(TryInto::try_into);
        let write = write.into_iter().map(TryInto::try_into);
        match &self {
            Differential::ZERO => check_zero(read),
            Differential::LRA(itype) => itype.check(read, write),
        }
    }
}

impl crate::Differential for Differential {
    const ZERO: Self = Differential::ZERO;
}

impl fmt::Display for Differential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Differential::ZERO => write!(f, "ZERO"),
            Differential::LRA(lra) => write!(f, "{}", lra),
        }
    }
}

impl TryFrom<Any> for Differential {
    type Error = String;
    fn try_from(any: Any) -> Result<Self, Self::Error> {
        match any {
            Any::ZERO => Ok(Differential::ZERO),
            Any::LRA(lra) => Ok(Differential::LRA(lra)),
            _ => Err(format!("{} is not in a differential theory", any)),
        }
    }
}

impl From<Differential> for Any {
    fn from(differential: Differential) -> Self {
        match differential {
            Differential::ZERO => Any::ZERO,
            Differential::LRA(lra) => Any::LRA(lra),
        }
    }
}

impl TryFrom<Differential> for LRA {
    type Error = String;
    fn try_from(value: Differential) -> Result<Self, Self::Error> {
        match value {
            Differential::ZERO => Ok(LRA::ZERO),
            Differential::LRA(lra) => Ok(lra),
        }
    }
}
