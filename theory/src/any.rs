use crate::bv::BV;
use crate::lia::LIA;
use crate::lra::LRA;
use crate::{Theory, bv, lia, lra};
use crate::{check_havoc, check_skip, check_zero};
use derive_more::From;
#[cfg(feature = "pyo3")]
use pyo3::{prelude::*, types::PyString};
use std::fmt;
use subenum::subenum;

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

//============================================================
// The cast lattice
//============================================================
//
// Casts between theories follow the hierarchy
//
// ```text
//     LRA -> Combinatorial, Sequential, Differential
//     LIA -> Combinatorial, Sequential
//     BV  -> Combinatorial, Sequential
//     Combinatorial -> Sequential
//     Sequential    -> Any
//     Differential  -> Any
// ```
//
// with `From` going up. Downward, only the casts from `Any` to the
// catch-alls exist (they serve the Python boundary, which builds typed
// blocks from `Any` terms); casting down to the base theories is unused and
// intentionally not implemented.
//
// `subenum` defines the catch-alls as tagged subsets of `Any` -- generating
// the sub-enums themselves, the casts between each subset and `Any`, and
// (via the propagated derives) the `From` wraps of base-theory ops and the
// `Display` impls.

#[subenum(Combinatorial, Differential, Sequential)]
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, From, strum::Display)]
pub enum Any {
    #[subenum(Combinatorial, Sequential)]
    HAVOC,
    #[subenum(Sequential)]
    SKIP,
    #[subenum(Differential)]
    ZERO,
    #[subenum(Combinatorial, Differential, Sequential)]
    #[strum(to_string = "{0}")]
    LRA(LRA),
    #[subenum(Combinatorial, Sequential)]
    #[strum(to_string = "{0}")]
    LIA(LIA),
    #[subenum(Combinatorial, Sequential)]
    #[strum(to_string = "{0}")]
    BV(BV),
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

impl crate::Sequential for Sequential {
    const SKIP: Self = Sequential::SKIP;
}

impl crate::Combinatorial for Sequential {
    const HAVOC: Self = Sequential::HAVOC;
}

impl crate::Combinatorial for Combinatorial {
    const HAVOC: Self = Combinatorial::HAVOC;
}

impl crate::Differential for Differential {
    const ZERO: Self = Differential::ZERO;
}

impl Theory for Any {
    type Sort = Sort;
    const NAME: &'static str = "Any";

    fn check<R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<Sort>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        let read = read.into_iter().map(TryInto::try_into);
        let write = write.into_iter().map(TryInto::try_into);
        match self {
            Any::HAVOC => check_havoc(read, write),
            Any::SKIP => check_skip(read, write),
            Any::ZERO => check_zero(read, write),
            Any::LRA(itype) => itype.check(read, write),
            Any::LIA(itype) => itype.check(read, write),
            Any::BV(itype) => itype.check(read, write),
        }
    }
}

impl Theory for Sequential {
    type Sort = Sort;
    const NAME: &'static str = "Sequential";

    fn check<R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<Sort>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        let read = read.into_iter().map(TryInto::try_into);
        let write = write.into_iter().map(TryInto::try_into);
        match self {
            Sequential::HAVOC => check_havoc(read, write),
            Sequential::SKIP => check_skip(read, write),
            Sequential::LRA(itype) => itype.check(read, write),
            Sequential::LIA(itype) => itype.check(read, write),
            Sequential::BV(itype) => itype.check(read, write),
        }
    }
}

impl Theory for Combinatorial {
    type Sort = Sort;
    const NAME: &'static str = "Combinatorial";

    fn check<R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<Sort>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        let read = read.into_iter().map(TryInto::try_into);
        let write = write.into_iter().map(TryInto::try_into);
        match self {
            Combinatorial::HAVOC => check_havoc(read, write),
            Combinatorial::LRA(itype) => itype.check(read, write),
            Combinatorial::LIA(itype) => itype.check(read, write),
            Combinatorial::BV(itype) => itype.check(read, write),
        }
    }
}

impl Theory for Differential {
    type Sort = Sort;
    const NAME: &'static str = "Differential";

    fn check<R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<Sort>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        let read = read.into_iter().map(TryInto::try_into);
        let write = write.into_iter().map(TryInto::try_into);
        match self {
            Differential::ZERO => check_zero(read, write),
            Differential::LRA(itype) => itype.check(read, write),
        }
    }
}

// The one cast `subenum` cannot express: the embedding between two
// sub-enums. Its inverse is intentionally absent -- nothing needs to cast
// down from `Sequential` to `Combinatorial`.
impl From<Combinatorial> for Sequential {
    fn from(op: Combinatorial) -> Self {
        match op {
            Combinatorial::HAVOC => Sequential::HAVOC,
            Combinatorial::BV(bv) => Sequential::BV(bv),
            Combinatorial::LRA(lra) => Sequential::LRA(lra),
            Combinatorial::LIA(lia) => Sequential::LIA(lia),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Casts up the hierarchy (Into) wrap the op in the right variant.

    #[test]
    fn base_theories_cast_up() {
        assert!(matches!(LRA::Add().into(), Combinatorial::LRA(LRA::Add())));
        assert!(matches!(LRA::Add().into(), Sequential::LRA(LRA::Add())));
        assert!(matches!(LRA::Add().into(), Differential::LRA(LRA::Add())));
        assert!(matches!(LIA::Add().into(), Combinatorial::LIA(LIA::Add())));
        assert!(matches!(LIA::Add().into(), Sequential::LIA(LIA::Add())));
        assert!(matches!(BV::And().into(), Combinatorial::BV(BV::And())));
        assert!(matches!(BV::And().into(), Sequential::BV(BV::And())));
    }

    #[test]
    fn catch_alls_cast_up() {
        let comb: Sequential = Combinatorial::LRA(LRA::Add()).into();
        assert!(matches!(comb, Sequential::LRA(LRA::Add())));
        assert!(matches!(Combinatorial::HAVOC.into(), Sequential::HAVOC));
        assert!(matches!(Sequential::SKIP.into(), Any::SKIP));
        assert!(matches!(Differential::ZERO.into(), Any::ZERO));
    }

    // Casts down the hierarchy (TryInto) recover the op or reject it.

    #[test]
    fn display_is_stable() {
        // Elements print their name, member ops delegate to the base theory.
        assert_eq!(Any::HAVOC.to_string(), "HAVOC");
        assert_eq!(Sequential::SKIP.to_string(), "SKIP");
        assert_eq!(Differential::ZERO.to_string(), "ZERO");
        assert_eq!(Any::LRA(LRA::Add()).to_string(), "Add");
        assert_eq!(Sequential::LIA(LIA::ReLU()).to_string(), "ReLU");
        assert_eq!(Combinatorial::BV(BV::MatMul()).to_string(), "MatMul");
        // Base-theory special cases keep their formats.
        assert_eq!(
            LRA::Uninterpreted("f".to_string()).to_string(),
            "Uninterpreted(f)"
        );
        assert_eq!(
            BV::BitSelect { high: 7, low: 0 }.to_string(),
            "BitSelect[7:0]"
        );
        assert_eq!(BV::Extend { extra: 8 }.to_string(), "Extend(+8)");
    }

    #[test]
    fn any_casts_down_or_rejects() {
        assert!(matches!(Any::SKIP.try_into(), Ok(Sequential::SKIP)));
        assert!(matches!(Any::HAVOC.try_into(), Ok(Combinatorial::HAVOC)));
        assert!(matches!(Any::ZERO.try_into(), Ok(Differential::ZERO)));
        // ZERO is differential-only; SKIP is not combinatorial or differential.
        assert!(Sequential::try_from(Any::ZERO).is_err());
        assert!(Combinatorial::try_from(Any::SKIP).is_err());
        assert!(Differential::try_from(Any::SKIP).is_err());
    }

    // The distinguished elements are unary at every level of the hierarchy,
    // matching the base theories' `Id`/`Havoc`/`Zero`.

    #[test]
    fn skip_is_unary() {
        let t = || Sort::Real([1, 1]);
        let u = || Sort::Real([2, 1]);
        for skip in [
            Any::SKIP.check([t()], [t()]),
            Sequential::SKIP.check([t()], [t()]),
        ] {
            assert!(skip.is_ok());
        }
        // arity and sort mismatches
        assert!(Any::SKIP.check([t(), u()], [t(), u()]).is_err());
        assert!(Any::SKIP.check([t()], [u()]).is_err());
        assert!(Any::SKIP.check([] as [Sort; 0], [] as [Sort; 0]).is_err());
        assert!(Sequential::SKIP.check([t(), u()], [t(), u()]).is_err());
    }

    #[test]
    fn havoc_is_unary() {
        let t = || Sort::Real([1, 1]);
        let u = || Sort::Real([2, 1]);
        for havoc in [
            Any::HAVOC.check([] as [Sort; 0], [t()]),
            Sequential::HAVOC.check([] as [Sort; 0], [t()]),
            Combinatorial::HAVOC.check([] as [Sort; 0], [t()]),
        ] {
            assert!(havoc.is_ok());
        }
        // no reads, exactly one write
        assert!(Any::HAVOC.check([t()], [t()]).is_err());
        assert!(Any::HAVOC.check([] as [Sort; 0], [t(), u()]).is_err());
        assert!(Any::HAVOC.check([] as [Sort; 0], [] as [Sort; 0]).is_err());
        assert!(
            Combinatorial::HAVOC
                .check([] as [Sort; 0], [t(), u()])
                .is_err()
        );
    }

    #[test]
    fn zero_is_unary() {
        let t = || Sort::Real([1, 1]);
        let u = || Sort::Real([2, 1]);
        for zero in [
            Any::ZERO.check([] as [Sort; 0], [t()]),
            Differential::ZERO.check([] as [Sort; 0], [t()]),
        ] {
            assert!(zero.is_ok());
        }
        // no reads, exactly one write
        assert!(Any::ZERO.check([t()], [t()]).is_err());
        assert!(Any::ZERO.check([] as [Sort; 0], [t(), u()]).is_err());
        assert!(Any::ZERO.check([] as [Sort; 0], [] as [Sort; 0]).is_err());
        assert!(
            Differential::ZERO
                .check([] as [Sort; 0], [t(), u()])
                .is_err()
        );
    }
}
