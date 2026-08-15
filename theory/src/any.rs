use crate::Combinatorial as _;
use crate::Differential as _;
use crate::Sequential as _;
use crate::bv::BV;
use crate::lia::LIA;
use crate::lra::LRA;
use crate::{Theory, bv, lia, lra};
#[cfg(feature = "pyo3")]
use pyo3::{prelude::*, types::PyString};
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
        match &self {
            Any::HAVOC => check_havoc(read, write),
            Any::SKIP => check_skip(read, write),
            Any::ZERO => check_zero(read, write),
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

impl From<BV> for Any {
    fn from(bv: BV) -> Self {
        Any::BV(bv)
    }
}

impl From<LIA> for Any {
    fn from(lia: LIA) -> Self {
        Any::LIA(lia)
    }
}

impl From<LRA> for Any {
    fn from(lra: LRA) -> Self {
        Any::LRA(lra)
    }
}

impl TryFrom<Any> for BV {
    type Error = String;
    fn try_from(value: Any) -> Result<Self, Self::Error> {
        match value {
            Any::SKIP => Ok(BV::SKIP),
            Any::HAVOC => Ok(BV::HAVOC),
            Any::BV(bv) => Ok(bv),
            _ => Err("invalid cast".to_string()),
        }
    }
}

impl TryFrom<Any> for LIA {
    type Error = String;
    fn try_from(value: Any) -> Result<Self, Self::Error> {
        match value {
            Any::SKIP => Ok(LIA::SKIP),
            Any::HAVOC => Ok(LIA::HAVOC),
            Any::LIA(lia) => Ok(lia),
            _ => Err("invalid cast".to_string()),
        }
    }
}

impl TryFrom<Any> for LRA {
    type Error = String;
    fn try_from(value: Any) -> Result<Self, Self::Error> {
        match value {
            Any::SKIP => Ok(LRA::SKIP),
            Any::HAVOC => Ok(LRA::HAVOC),
            Any::ZERO => Ok(LRA::ZERO),
            Any::LRA(lra) => Ok(lra),
            _ => Err("invalid cast".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Sequential {
    HAVOC,
    SKIP,
    BV(BV),
    LRA(LRA),
    LIA(LIA),
}

impl crate::Sequential for Sequential {
    const SKIP: Self = Sequential::SKIP;
}

impl crate::Combinatorial for Sequential {
    const HAVOC: Self = Sequential::HAVOC;
}

// SKIP is unary: it copies exactly one read wire to one write wire of the
// same sort, matching the arity of the base theories' `Id`.
fn check_skip<R, W, E>(read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match (read.next(), write.next()) {
        (Some(Ok(r)), Some(Ok(w))) if r == w => {}
        _ => {
            return Err(
                "SKIP expects exactly one read and one write of the same sort".to_string(),
            );
        }
    }
    if read.next().is_some() || write.next().is_some() {
        return Err("SKIP expects exactly one read and one write".to_string());
    }
    Ok(())
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
            Sequential::HAVOC => check_havoc(read, write),
            Sequential::SKIP => check_skip(read, write),
            Sequential::BV(itype) => itype.check(read, write),
            Sequential::LRA(itype) => itype.check(read, write),
            Sequential::LIA(itype) => itype.check(read, write),
        }
    }
}

impl fmt::Display for Sequential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sequential::HAVOC => write!(f, "HAVOC"),
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
            Any::HAVOC => Ok(Sequential::HAVOC),
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
            Sequential::HAVOC => Any::HAVOC,
            Sequential::SKIP => Any::SKIP,
            Sequential::BV(bv) => Any::BV(bv),
            Sequential::LRA(lra) => Any::LRA(lra),
            Sequential::LIA(lia) => Any::LIA(lia),
        }
    }
}

impl From<BV> for Sequential {
    fn from(bv: BV) -> Self {
        Sequential::BV(bv)
    }
}

impl From<LIA> for Sequential {
    fn from(lia: LIA) -> Self {
        Sequential::LIA(lia)
    }
}

impl From<LRA> for Sequential {
    fn from(lra: LRA) -> Self {
        Sequential::LRA(lra)
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

#[derive(Debug, Clone)]
pub enum Combinatorial {
    HAVOC,
    BV(BV),
    LRA(LRA),
    LIA(LIA),
}

impl crate::Combinatorial for Combinatorial {
    const HAVOC: Self = Combinatorial::HAVOC;
}

// HAVOC is unary: it writes exactly one wire and reads none.
pub(crate) fn check_havoc<R, W>(read: R, write: W) -> Result<(), String>
where
    R: IntoIterator,
    W: IntoIterator,
{
    if read.into_iter().next().is_some() {
        return Err("HAVOC expects no read wires".to_string());
    }
    let mut write = write.into_iter();
    if write.next().is_none() {
        return Err("HAVOC expects exactly one write wire, got none".to_string());
    }
    if write.next().is_some() {
        return Err("HAVOC expects exactly one write wire, got more".to_string());
    }
    Ok(())
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
            Combinatorial::HAVOC => check_havoc(read, write),
            Combinatorial::BV(itype) => itype.check(read, write),
            Combinatorial::LRA(itype) => itype.check(read, write),
            Combinatorial::LIA(itype) => itype.check(read, write),
        }
    }
}

impl fmt::Display for Combinatorial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Combinatorial::HAVOC => write!(f, "HAVOC"),
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
            Combinatorial::BV(bv) => Any::BV(bv),
            Combinatorial::LRA(lra) => Any::LRA(lra),
            Combinatorial::LIA(lia) => Any::LIA(lia),
        }
    }
}

impl From<Combinatorial> for Sequential {
    fn from(combinatorial: Combinatorial) -> Self {
        match combinatorial {
            Combinatorial::HAVOC => Sequential::HAVOC,
            Combinatorial::BV(bv) => Sequential::BV(bv),
            Combinatorial::LRA(lra) => Sequential::LRA(lra),
            Combinatorial::LIA(lia) => Sequential::LIA(lia),
        }
    }
}

impl TryFrom<Sequential> for Combinatorial {
    type Error = String;
    fn try_from(value: Sequential) -> Result<Self, Self::Error> {
        match value {
            Sequential::HAVOC => Ok(Combinatorial::HAVOC),
            Sequential::BV(bv) => Ok(Combinatorial::BV(bv)),
            Sequential::LRA(lra) => Ok(Combinatorial::LRA(lra)),
            Sequential::LIA(lia) => Ok(Combinatorial::LIA(lia)),
            Sequential::SKIP => Err("SKIP is not in a combinatorial theory".to_string()),
        }
    }
}

impl From<BV> for Combinatorial {
    fn from(bv: BV) -> Self {
        Combinatorial::BV(bv)
    }
}

impl From<LIA> for Combinatorial {
    fn from(lia: LIA) -> Self {
        Combinatorial::LIA(lia)
    }
}

impl From<LRA> for Combinatorial {
    fn from(lra: LRA) -> Self {
        Combinatorial::LRA(lra)
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

impl TryFrom<Combinatorial> for LIA {
    type Error = String;
    fn try_from(value: Combinatorial) -> Result<Self, Self::Error> {
        match value {
            Combinatorial::HAVOC => Ok(LIA::HAVOC),
            Combinatorial::LIA(lia) => Ok(lia),
            _ => Err("invalid cast".to_string()),
        }
    }
}

impl TryFrom<Combinatorial> for LRA {
    type Error = String;
    fn try_from(value: Combinatorial) -> Result<Self, Self::Error> {
        match value {
            Combinatorial::HAVOC => Ok(LRA::HAVOC),
            Combinatorial::LRA(lra) => Ok(lra),
            _ => Err("invalid cast".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Differential {
    ZERO,
    LRA(LRA),
}

// ZERO is unary: it writes exactly one wire and reads none.
pub(crate) fn check_zero<R, W>(read: R, write: W) -> Result<(), String>
where
    R: IntoIterator,
    W: IntoIterator,
{
    if read.into_iter().next().is_some() {
        return Err("ZERO expects no read wires".to_string());
    }
    let mut write = write.into_iter();
    if write.next().is_none() {
        return Err("ZERO expects exactly one write wire, got none".to_string());
    }
    if write.next().is_some() {
        return Err("ZERO expects exactly one write wire, got more".to_string());
    }
    Ok(())
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
            Differential::ZERO => check_zero(read, write),
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

impl From<LRA> for Differential {
    fn from(lra: LRA) -> Self {
        Differential::LRA(lra)
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
    fn round_trip_through_sequential() {
        let up: Sequential = LRA::Add().into();
        assert!(matches!(up.try_into(), Ok(LRA::Add())));

        let up: Sequential = LIA::Add().into();
        assert!(matches!(up.try_into(), Ok(LIA::Add())));

        let up: Sequential = BV::And().into();
        assert!(matches!(up.try_into(), Ok(BV::And())));
    }

    #[test]
    fn round_trip_through_combinatorial() {
        let up: Combinatorial = LRA::Add().into();
        assert!(matches!(up.try_into(), Ok(LRA::Add())));

        let up: Combinatorial = LIA::Add().into();
        assert!(matches!(up.try_into(), Ok(LIA::Add())));

        let up: Combinatorial = BV::And().into();
        assert!(matches!(up.try_into(), Ok(BV::And())));
    }

    #[test]
    fn round_trip_through_differential() {
        let up: Differential = LRA::Add().into();
        assert!(matches!(up.try_into(), Ok(LRA::Add())));
    }

    #[test]
    fn distinguished_elements_cast_down_to_base() {
        // The catch-alls' SKIP/HAVOC/ZERO land on the base theories' elements.
        assert!(matches!(Sequential::SKIP.try_into(), Ok(LRA::Id())));
        assert!(matches!(Sequential::SKIP.try_into(), Ok(LIA::Id())));
        assert!(matches!(Combinatorial::HAVOC.try_into(), Ok(LRA::Havoc())));
        assert!(matches!(Combinatorial::HAVOC.try_into(), Ok(LIA::Havoc())));
        assert!(matches!(Differential::ZERO.try_into(), Ok(LRA::Zero())));
    }

    #[test]
    fn sequential_casts_down_to_combinatorial() {
        let seq: Sequential = LRA::Add().into();
        assert!(matches!(
            seq.try_into(),
            Ok(Combinatorial::LRA(LRA::Add()))
        ));
        assert!(matches!(
            Sequential::HAVOC.try_into(),
            Ok(Combinatorial::HAVOC)
        ));
        // SKIP is sequential-only.
        assert!(Combinatorial::try_from(Sequential::SKIP).is_err());
    }

    #[test]
    fn foreign_ops_fail_to_cast_down() {
        // An LRA op does not cast down to LIA or BV, and vice versa.
        let seq: Sequential = LRA::Add().into();
        assert!(LIA::try_from(seq).is_err());
        let seq: Sequential = LIA::Add().into();
        assert!(BV::try_from(seq).is_err());
        let comb: Combinatorial = BV::And().into();
        assert!(LRA::try_from(comb).is_err());
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

    #[test]
    fn round_trip_through_any() {
        // Transitive shortcuts: base theories cast straight up to Any and back.
        let up: Any = LRA::Add().into();
        assert!(matches!(up, Any::LRA(LRA::Add())));
        assert!(matches!(Any::LRA(LRA::Add()).try_into(), Ok(LRA::Add())));

        let up: Any = LIA::Add().into();
        assert!(matches!(up, Any::LIA(LIA::Add())));
        assert!(matches!(Any::LIA(LIA::Add()).try_into(), Ok(LIA::Add())));

        let up: Any = BV::And().into();
        assert!(matches!(up, Any::BV(BV::And())));
        assert!(matches!(Any::BV(BV::And()).try_into(), Ok(BV::And())));
    }

    #[test]
    fn any_distinguished_elements_cast_down_to_base() {
        assert!(matches!(Any::SKIP.try_into(), Ok(LRA::Id())));
        assert!(matches!(Any::SKIP.try_into(), Ok(LIA::Id())));
        assert!(matches!(Any::SKIP.try_into(), Ok(BV::Id())));
        assert!(matches!(Any::HAVOC.try_into(), Ok(LRA::Havoc())));
        assert!(matches!(Any::HAVOC.try_into(), Ok(LIA::Havoc())));
        assert!(matches!(Any::HAVOC.try_into(), Ok(BV::Havoc())));
        // ZERO reaches only the differential base theory, LRA.
        assert!(matches!(Any::ZERO.try_into(), Ok(LRA::Zero())));
        assert!(LIA::try_from(Any::ZERO).is_err());
        assert!(BV::try_from(Any::ZERO).is_err());
        // Foreign ops reject.
        assert!(LIA::try_from(Any::LRA(LRA::Add())).is_err());
        assert!(BV::try_from(Any::LIA(LIA::Add())).is_err());
        assert!(LRA::try_from(Any::BV(BV::And())).is_err());
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
