use crate::bv::BV;
use crate::lia::LIA;
use crate::lra::LRA;
use crate::{Signature, Tangent, bv, lia, lra};
use crate::{check_havoc, check_skip, check_zero};
use derive_more::From;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;
use subenum::subenum;

// The enum is defined twice, gated on the `pyo3` feature: the variant-level
// `#[pyo3(constructor = ...)]` attribute cannot be wrapped in `cfg_attr`
// (rustc does not expand `cfg_attr` inside a proc macro's input, so
// `pyclass` would never see it). Keep the two definitions in sync.
#[cfg(feature = "pyo3")]
#[pyclass(frozen, eq, str)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sort {
    Bool([usize; 2]),
    /// A real tensor; `rank` is the differential grade: 0 = value,
    /// 1 = first derivative, ...
    #[pyo3(constructor = (shape, rank = 0))]
    Real {
        shape: [usize; 2],
        rank: u8,
    },
    Int([usize; 2]),
    BitVec(usize, [usize; 2]),
    /// The trivial tangent of the constant sorts (Bool, Int, BitVec): a
    /// singleton, inhabited by exactly the zero value. Terminal, not empty.
    Zero(),
}

#[cfg(not(feature = "pyo3"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sort {
    Bool([usize; 2]),
    /// A real tensor; `rank` is the differential grade: 0 = value,
    /// 1 = first derivative, ...
    Real {
        shape: [usize; 2],
        rank: u8,
    },
    Int([usize; 2]),
    BitVec(usize, [usize; 2]),
    /// The trivial tangent of the constant sorts (Bool, Int, BitVec): a
    /// singleton, inhabited by exactly the zero value. Terminal, not empty.
    Zero(),
}

impl Sort {
    /// A real value sort (rank 0).
    pub fn real(shape: [usize; 2]) -> Self {
        Sort::Real { shape, rank: 0 }
    }
}

/// The tangent former: reals grade up (`rank + 1`, same carrier); the
/// constant sorts (Bool, Int, BitVec) collapse to the trivial tangent
/// `Zero`, which is a fixed point.
impl Tangent for Sort {
    #[allow(non_snake_case)]
    fn T(&self) -> Self {
        match *self {
            Sort::Real { shape, rank } => Sort::Real {
                shape,
                rank: rank + 1,
            },
            Sort::Bool(_) | Sort::Int(_) | Sort::BitVec(..) => Sort::Zero(),
            Sort::Zero() => Sort::Zero(),
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Bool(shape) => {
                write!(f, "Bool([{},{}])", shape[0], shape[1])
            }
            Sort::Real { shape, rank: 0 } => {
                write!(f, "Real([{},{}])", shape[0], shape[1])
            }
            Sort::Real { shape, rank } => {
                write!(f, "T{} Real([{},{}])", rank, shape[0], shape[1])
            }
            Sort::Int(shape) => {
                write!(f, "Int([{},{}])", shape[0], shape[1])
            }
            Sort::BitVec(bw, shape) => {
                write!(f, "Bv{}([{},{}])", bw, shape[0], shape[1])
            }
            Sort::Zero() => {
                write!(f, "Zero")
            }
        }
    }
}

impl From<bv::Sort> for Sort {
    fn from(value: bv::Sort) -> Self {
        match value {
            bv::Sort::BV(bw, shape) => Sort::BitVec(bw, shape),
            bv::Sort::Zero => Sort::Zero(),
        }
    }
}

impl From<lia::Sort> for Sort {
    fn from(value: lia::Sort) -> Self {
        match value {
            lia::Sort::Int(shape) => Sort::Int(shape),
            lia::Sort::Bool(shape) => Sort::Bool(shape),
            lia::Sort::Zero => Sort::Zero(),
        }
    }
}

impl From<lra::Sort> for Sort {
    fn from(value: lra::Sort) -> Self {
        match value {
            lra::Sort::Bool(shape) => Sort::Bool(shape),
            lra::Sort::Real { shape, rank } => Sort::Real { shape, rank },
            lra::Sort::Zero => Sort::Zero(),
        }
    }
}

impl TryFrom<Sort> for bv::Sort {
    type Error = String;

    fn try_from(value: Sort) -> Result<Self, Self::Error> {
        match value {
            Sort::BitVec(bw, shape) => Ok(bv::Sort::BV(bw, shape)),
            Sort::Zero() => Ok(bv::Sort::Zero),
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
            Sort::Zero() => Ok(lia::Sort::Zero),
            _ => Err("invalid cast".to_string()),
        }
    }
}

impl TryFrom<Sort> for lra::Sort {
    type Error = String;

    fn try_from(value: Sort) -> Result<Self, Self::Error> {
        match value {
            Sort::Bool(shape) => Ok(lra::Sort::Bool(shape)),
            Sort::Real { shape, rank } => Ok(lra::Sort::Real { shape, rank }),
            Sort::Zero() => Ok(lra::Sort::Zero),
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

#[subenum(
    Combinatorial(cfg_attr(feature = "pyo3", pyclass)),
    Differential(cfg_attr(feature = "pyo3", pyclass)),
    Sequential(cfg_attr(feature = "pyo3", pyclass))
)]
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, From, strum::Display)]
pub enum Any {
    #[subenum(Combinatorial, Sequential)]
    #[from(skip)]
    HAVOC(Sort),
    #[subenum(Sequential)]
    #[from(skip)]
    SKIP(Sort),
    #[subenum(Differential)]
    #[from(skip)]
    ZERO(Sort),
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
            Any::HAVOC(a) => Combinatorial::HAVOC(a)
                .into_pyobject(py)
                .map(Bound::into_any),
            Any::SKIP(a) => Sequential::SKIP(a).into_pyobject(py).map(Bound::into_any),
            Any::ZERO(a) => Differential::ZERO(a).into_pyobject(py).map(Bound::into_any),
            Any::LRA(a) => a.into_pyobject(py).map(Bound::into_any),
            Any::LIA(a) => a.into_pyobject(py).map(Bound::into_any),
            Any::BV(a) => a.into_pyobject(py).map(Bound::into_any),
        }
    }
}

impl crate::Sequential for Sequential {
    fn skip(range: &Sort) -> Self {
        Sequential::SKIP(*range)
    }
}

impl crate::Combinatorial for Sequential {
    fn havoc(range: &Sort) -> Self {
        Sequential::HAVOC(*range)
    }
}

impl crate::Combinatorial for Combinatorial {
    fn havoc(range: &Sort) -> Self {
        Combinatorial::HAVOC(*range)
    }
}

impl crate::Differential for Differential {
    fn zero(range: &Sort) -> Self {
        Differential::ZERO(*range)
    }
}

impl Signature for Any {
    type Sort = Sort;
    const NAME: &'static str = "Any";

    fn check<R, W, E: fmt::Display>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<Sort, E>>,
        W: IntoIterator<Item = Result<Sort, E>>,
    {
        // let read = read.into_iter().map(TryInto::try_into);
        // let write = write.into_iter().map(TryInto::try_into);
        match self {
            Any::HAVOC(range) => check_havoc(range, read, write),
            Any::SKIP(range) => check_skip(range, read, write),
            Any::ZERO(range) => check_zero(range, read, write),
            Any::LRA(itype) => itype.check(try_into(read), try_into(write)),
            Any::LIA(itype) => itype.check(try_into(read), try_into(write)),
            Any::BV(itype) => itype.check(try_into(read), try_into(write)),
        }
    }
}

fn try_into<R, S, T, E: fmt::Display, F: fmt::Display>(
    iter: R,
) -> impl Iterator<Item = Result<T, String>>
where
    S: TryInto<T, Error = F>,
    R: IntoIterator<Item = Result<S, E>>,
{
    iter.into_iter().map(|r| {
        r.map_err(|e| e.to_string())
            .and_then(|s| s.try_into().map_err(|f| f.to_string()))
    })
}

impl Signature for Sequential {
    type Sort = Sort;
    const NAME: &'static str = "Sequential";

    fn check<R, W, E: fmt::Display>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<Sort, E>>,
        W: IntoIterator<Item = Result<Sort, E>>,
    {
        match self {
            Sequential::HAVOC(range) => check_havoc(range, read, write),
            Sequential::SKIP(range) => check_skip(range, read, write),
            Sequential::LRA(itype) => itype.check(try_into(read), try_into(write)),
            Sequential::LIA(itype) => itype.check(try_into(read), try_into(write)),
            Sequential::BV(itype) => itype.check(try_into(read), try_into(write)),
        }
    }
}

impl Signature for Combinatorial {
    type Sort = Sort;
    const NAME: &'static str = "Combinatorial";

    fn check<R, W, E: fmt::Display>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<Sort, E>>,
        W: IntoIterator<Item = Result<Sort, E>>,
    {
        // let read = read.into_iter().map(TryInto::try_into);
        // let write = write.into_iter().map(TryInto::try_into);
        match self {
            Combinatorial::HAVOC(range) => check_havoc(range, read, write),
            Combinatorial::LRA(itype) => itype.check(try_into(read), try_into(write)),
            Combinatorial::LIA(itype) => itype.check(try_into(read), try_into(write)),
            Combinatorial::BV(itype) => itype.check(try_into(read), try_into(write)),
        }
    }
}

impl Signature for Differential {
    type Sort = Sort;
    const NAME: &'static str = "Differential";

    fn check<R, W, E: fmt::Display>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<Sort, E>>,
        W: IntoIterator<Item = Result<Sort, E>>,
    {
        // let read = read.into_iter().map(TryInto::try_into);
        // let write = write.into_iter().map(TryInto::try_into);
        match self {
            Differential::ZERO(range) => check_zero(range, read, write),
            Differential::LRA(itype) => itype.check(try_into(read), try_into(write)),
        }
    }
}

// The one cast `subenum` cannot express: the embedding between two
// sub-enums. Its inverse is intentionally absent -- nothing needs to cast
// down from `Sequential` to `Combinatorial`.
impl From<Combinatorial> for Sequential {
    fn from(op: Combinatorial) -> Self {
        match op {
            Combinatorial::HAVOC(range) => Sequential::HAVOC(range),
            Combinatorial::BV(bv) => Sequential::BV(bv),
            Combinatorial::LRA(lra) => Sequential::LRA(lra),
            Combinatorial::LIA(lia) => Sequential::LIA(lia),
        }
    }
}
//
#[cfg(test)]
mod tests {
    use super::*;

    fn ok(s: Sort) -> Result<Sort, String> {
        Ok(s)
    }

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
        let t = Sort::real([1, 1]);
        assert!(matches!(
            Combinatorial::HAVOC(t).into(),
            Sequential::HAVOC(_)
        ));
        assert!(matches!(Sequential::SKIP(t).into(), Any::SKIP(_)));
        assert!(matches!(Differential::ZERO(t).into(), Any::ZERO(_)));
    }

    // Casts down the hierarchy (TryInto) recover the op or reject it.

    #[test]
    fn display_is_stable() {
        // Elements print their name, member ops delegate to the base theory.
        let t = Sort::real([1, 1]);
        assert_eq!(Any::HAVOC(t).to_string(), "HAVOC");
        assert_eq!(Sequential::SKIP(t).to_string(), "SKIP");
        assert_eq!(Differential::ZERO(t).to_string(), "ZERO");
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
        let t = Sort::real([1, 1]);
        assert!(matches!(Any::SKIP(t).try_into(), Ok(Sequential::SKIP(_))));
        assert!(matches!(
            Any::HAVOC(t).try_into(),
            Ok(Combinatorial::HAVOC(_))
        ));
        assert!(matches!(Any::ZERO(t).try_into(), Ok(Differential::ZERO(_))));
        // ZERO is differential-only; SKIP is not combinatorial or differential.
        assert!(Sequential::try_from(Any::ZERO(t)).is_err());
        assert!(Combinatorial::try_from(Any::SKIP(t)).is_err());
        assert!(Differential::try_from(Any::SKIP(t)).is_err());
    }

    // The distinguished elements are unary at every level of the hierarchy,
    // matching the base theories' `Id`/`Havoc`/`Zero`.

    #[test]
    fn skip_is_unary() {
        let t = || Sort::real([1, 1]);
        let u = || Sort::real([2, 1]);
        for skip in [
            Any::SKIP(t()).check([t()].map(ok), [t()].map(ok)),
            Sequential::SKIP(t()).check([t()].map(ok), [t()].map(ok)),
        ] {
            assert!(skip.is_ok());
        }
        // arity and sort mismatches
        assert!(
            Any::SKIP(t())
                .check([t(), u()].map(ok), [t(), u()].map(ok))
                .is_err()
        );
        assert!(Any::SKIP(t()).check([t()].map(ok), [u()].map(ok)).is_err());
        assert!(Any::SKIP(t()).check([].map(ok), [].map(ok)).is_err());
        assert!(
            Sequential::SKIP(t())
                .check([t(), u()].map(ok), [t(), u()].map(ok))
                .is_err()
        );
    }

    #[test]
    fn havoc_is_unary() {
        let t = || Sort::real([1, 1]);
        let u = || Sort::real([2, 1]);
        for havoc in [
            Any::HAVOC(t()).check([].map(ok), [t()].map(ok)),
            Sequential::HAVOC(t()).check([].map(ok), [t()].map(ok)),
            Combinatorial::HAVOC(t()).check([].map(ok), [t()].map(ok)),
        ] {
            assert!(havoc.is_ok());
        }
        // no reads, exactly one write
        assert!(Any::HAVOC(t()).check([t()].map(ok), [t()].map(ok)).is_err());
        assert!(
            Any::HAVOC(t())
                .check([].map(ok), [t(), u()].map(ok))
                .is_err()
        );
        assert!(Any::HAVOC(t()).check([].map(ok), [].map(ok)).is_err());
        assert!(
            Combinatorial::HAVOC(t())
                .check([].map(ok), [t(), u()].map(ok))
                .is_err()
        );
    }

    #[test]
    fn zero_is_unary() {
        let t = || Sort::real([1, 1]);
        let u = || Sort::real([2, 1]);
        for zero in [
            Any::ZERO(t()).check([].map(ok), [t()].map(ok)),
            Differential::ZERO(t()).check([].map(ok), [t()].map(ok)),
        ] {
            assert!(zero.is_ok());
        }
        // no reads, exactly one write
        assert!(Any::ZERO(t()).check([t()].map(ok), [t()].map(ok)).is_err());
        assert!(
            Any::ZERO(t())
                .check([].map(ok), [t(), u()].map(ok))
                .is_err()
        );
        assert!(Any::ZERO(t()).check([].map(ok), [].map(ok)).is_err());
        assert!(
            Differential::ZERO(t())
                .check([].map(ok), [t(), u()].map(ok))
                .is_err()
        );
    }
}
