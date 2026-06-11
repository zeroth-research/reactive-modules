use pyo3::prelude::*;
use pyo3::{Bound, FromPyObject, PyAny, PyResult, pyclass};
use std::fmt;
use theory::bv::BV;
use theory::lia::LIA;
use theory::lra::LRA;
use theory::{bv, lia, lra};

#[derive(Debug, Clone, PartialEq, Eq)]
#[pyclass(frozen)]
pub enum Sort {
    Bool([usize; 2]),
    Real([usize; 2]),
    Int([usize; 2]),
    BV(usize, [usize; 2]),
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Bool(s) => write!(f, "Bool({}, {})", s[0], s[1]),
            Sort::Real(s) => write!(f, "Real({}, {})", s[0], s[1]),
            Sort::Int(s) => write!(f, "Int({}, {})", s[0], s[1]),
            Sort::BV(bw, s) => write!(f, "BV<{}>({}, {})", bw, s[0], s[1]),
        }
    }
}

impl From<bv::Sort> for Sort {
    fn from(value: bv::Sort) -> Self {
        match value {
            bv::Sort::BV(bw, shape) => Sort::BV(bw, shape),
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

impl TryFrom<Sort> for bv::Sort {
    type Error = String;

    fn try_from(value: Sort) -> Result<Self, Self::Error> {
        match value {
            Sort::BV(bw, shape) => Ok(bv::Sort::BV(bw, shape)),
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
    LRA(LRA),
    LIA(LIA),
    BV(BV),
}

impl fmt::Display for Any {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Any::LRA(op) => write!(f, "{}", op),
            Any::LIA(op) => write!(f, "{}", op),
            Any::BV(op) => write!(f, "{}", op),
        }
    }
}

struct TryFrom2<A>
where
    A: TryInto<Sort>,
{
    a: A,
}

impl<A: TryInto<Sort>> TryFrom2<A> {
    fn new(a: A) -> Self {
        Self { a }
    }
}

impl<A: TryInto<Sort> + fmt::Display> fmt::Display for TryFrom2<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.a, f)
    }
}

impl<A: TryInto<Sort>> TryFrom<TryFrom2<A>> for lra::Sort {
    type Error = String;

    fn try_from(value: TryFrom2<A>) -> Result<Self, Self::Error> {
        let b: Sort = value.a.try_into().map_err(|_| "invalid cast")?;
        let c: lra::Sort = b.try_into().map_err(|e: String| e.to_string())?;
        Ok(c)
    }
}

impl<A: TryInto<Sort>> TryFrom<TryFrom2<A>> for lia::Sort {
    type Error = String;

    fn try_from(value: TryFrom2<A>) -> Result<Self, Self::Error> {
        let b: Sort = value.a.try_into().map_err(|_| "invalid cast")?;
        let c: lia::Sort = b.try_into().map_err(|e: String| e.to_string())?;
        Ok(c)
    }
}

impl<A: TryInto<Sort>> TryFrom<TryFrom2<A>> for bv::Sort {
    type Error = String;

    fn try_from(value: TryFrom2<A>) -> Result<Self, Self::Error> {
        let b: Sort = value.a.try_into().map_err(|_| "invalid cast")?;
        let c: bv::Sort = b.try_into().map_err(|e: String| e.to_string())?;
        Ok(c)
    }
}

impl theory::Theory for Any {
    type Sort = Sort;
    const NAME: &'static str = "Any";

    fn check<R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<Sort> + fmt::Display,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        let read = read.into_iter().map(TryFrom2::new);
        let write = write.into_iter().map(TryFrom2::new);
        match &self {
            Any::LRA(itype) => itype.check(read, write),
            Any::LIA(itype) => itype.check(read, write),
            Any::BV(itype) => itype.check(read, write),
        }
    }
}

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

impl<'py> IntoPyObject<'py> for Any {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            Any::LRA(a) => a.into_pyobject(py).map(Bound::into_any),
            Any::LIA(a) => a.into_pyobject(py).map(Bound::into_any),
            Any::BV(a) => a.into_pyobject(py).map(Bound::into_any),
        }
    }
}
