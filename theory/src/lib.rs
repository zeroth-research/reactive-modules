pub mod any;
pub mod bv;
pub mod lia;
pub mod lra;
pub mod sort;
pub mod tensor;

use crate::sort::{MultiSort, TangentMultiSort};
use std::fmt;
use std::fmt::Debug;
//use std::ops::Mul;
pub use tensor::PyTensor;

type Name = u8;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatName {
    // symbolic name, e.g., `N` in `Mat(Int, N, 5)`
    Sym(Name),
    // concrete constant, e.g., `8` in `Mat(Int, N, 5)`
    Const(usize),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SignatureElem<MS: MultiSort> {
    // sort
    S(MS),
    // tangent space of a sort
    Tan(MS),
    // type variable
    TypeVar(Name),
    // dimension variable (natural number)
    // NatVar(Name),
    // matrix of element type `MS` and dimensions of type variables
    Mat(MS, MatName, MatName),
}

#[allow(non_snake_case)]
impl<MS: MultiSort> SignatureElem<MS> {
    pub fn mk_S(srt: MS) -> SignatureElem<MS> {
        Self::S(srt)
    }

    pub fn mk_TypeVar(name: Name) -> SignatureElem<MS> {
        Self::TypeVar(name)
    }

    //pub fn mk_NatVar(name: Name) -> SignatureElem<MS> {
    //    todo!()
    //}

    pub fn mk_Mat(srt: MS, m: MatName, n: MatName) -> SignatureElem<MS> {
        Self::Mat(srt, m, n)
    }
}

#[allow(non_snake_case)]
impl<MS: TangentMultiSort> SignatureElem<MS> {
    pub fn mk_Tan(srt: MS) -> SignatureElem<MS> {
        Self::Tan(srt)
    }
}

// Variables are printed as `a`, `b`, ..., `z`, `a1`, `b1`, ..., `z1`, `a2`, ...
fn fmt_var_name(f: &mut fmt::Formatter<'_>, name: Name) -> fmt::Result {
    let letter = char::from(b'a' + name % 26);
    match name / 26 {
        0 => write!(f, "{letter}"),
        round => write!(f, "{letter}{round}"),
    }
}

impl fmt::Display for MatName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatName::Sym(name) => fmt_var_name(f, *name),
            MatName::Const(n) => write!(f, "{n}"),
        }
    }
}

impl<MS: MultiSort + fmt::Display> fmt::Display for SignatureElem<MS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignatureElem::S(srt) => write!(f, "{srt}"),
            SignatureElem::Tan(srt) => write!(f, "Tan({srt})"),
            SignatureElem::Mat(srt, m, n) => write!(f, "Mat({srt}, {m}, {n})"),
            SignatureElem::TypeVar(name) => {
                f.write_str("'")?;
                fmt_var_name(f, *name)
            }
        }
    }
}

pub trait Signature<MS: MultiSort> {
    fn dom(&self) -> impl Iterator<Item = SignatureElem<MS>>;
    fn cod(&self) -> impl Iterator<Item = SignatureElem<MS>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Copy, Clone)]
    enum TestSort {
        A(),
        B(usize),
        TB(usize, usize),
        TanZero(),
    }

    struct TestSignature {
        dom: Vec<SignatureElem<TestSort>>,
        cod: Vec<SignatureElem<TestSort>>,
    }

    impl Signature<TestSort> for TestSignature {
        fn dom(&self) -> impl Iterator<Item = SignatureElem<TestSort>> {
            self.dom.iter().cloned()
        }

        fn cod(&self) -> impl Iterator<Item = SignatureElem<TestSort>> {
            self.cod.iter().cloned()
        }
    }

    // Prints, e.g., `(A, B(8)) -> B(8)`; a domain or codomain that is not
    // a single element is printed as a tuple, e.g., `() -> A`.
    impl fmt::Display for TestSignature {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let fmt_elems = |f: &mut fmt::Formatter<'_>, elems: &[SignatureElem<TestSort>]| {
                if let [elem] = elems {
                    return write!(f, "{elem}");
                }
                f.write_str("(")?;
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{elem}")?;
                }
                f.write_str(")")
            };
            fmt_elems(f, &self.dom)?;
            f.write_str(" -> ")?;
            fmt_elems(f, &self.cod)
        }
    }

    impl MultiSort for TestSort {
        const NAME: &'static str = "TestSort";
    }

    impl fmt::Display for TestSort {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                TestSort::A() => write!(f, "A"),
                TestSort::B(n) => write!(f, "B({n})"),
                TestSort::TB(n, d) => write!(f, "TB({n}, {d})"),
                TestSort::TanZero() => write!(f, "TanZero"),
            }
        }
    }

    impl TangentMultiSort for TestSort {
        //const NAME = "TestSort";
        fn T(&self) -> TestSort {
            match self {
                TestSort::A() => TestSort::TanZero(),
                TestSort::B(n) => TestSort::TB(*n, 1),
                TestSort::TB(n, d) => TestSort::TB(*n, d + 1),
                TestSort::TanZero() => TestSort::TanZero(),
            }
        }
    }

    #[test]
    fn test_1() {
        let t = SignatureElem::<TestSort>::mk_TypeVar(0);
        let A = SignatureElem::<TestSort>::mk_S(TestSort::A());
        let B = SignatureElem::<TestSort>::mk_S(TestSort::B(8));
        let T = SignatureElem::<TestSort>::mk_Tan(TestSort::B(4).T());
        let M1 =
            SignatureElem::<TestSort>::mk_Mat(TestSort::B(3), MatName::Sym(0), MatName::Sym(1));
        let M2 =
            SignatureElem::<TestSort>::mk_Mat(TestSort::B(3), MatName::Sym(1), MatName::Sym(2));
        let M3 =
            SignatureElem::<TestSort>::mk_Mat(TestSort::B(3), MatName::Sym(0), MatName::Sym(2));

        let sig: TestSignature = TestSignature {
            dom: vec![A, B, M1, M2, t],
            cod: vec![M3],
        };
        assert_eq!(
            sig.to_string(),
            "(A, B(8), Mat(B(3), a, b), Mat(B(3), b, c), 'a) -> Mat(B(3), a, c)"
        );
    }

    #[test]
    fn display_signature_elems() {
        type E = SignatureElem<TestSort>;
        assert_eq!(E::mk_S(TestSort::B(8)).to_string(), "B(8)");
        assert_eq!(E::mk_Tan(TestSort::B(4)).to_string(), "Tan(B(4))");
        assert_eq!(E::mk_TypeVar(1).to_string(), "'b");
        assert_eq!(
            E::mk_Mat(TestSort::A(), MatName::Sym(13), MatName::Const(5)).to_string(),
            "Mat(A, n, 5)"
        );
    }

    #[test]
    fn display_var_names_past_z() {
        assert_eq!(MatName::Sym(25).to_string(), "z");
        assert_eq!(MatName::Sym(26).to_string(), "a1");
        assert_eq!(MatName::Sym(27).to_string(), "b1");
        assert_eq!(MatName::Sym(255).to_string(), "v9");
        assert_eq!(SignatureElem::<TestSort>::mk_TypeVar(52).to_string(), "'a2");
    }

    #[test]
    fn display_tuples() {
        let a = SignatureElem::mk_S(TestSort::A());
        let b = SignatureElem::mk_S(TestSort::B(8));
        let sig = |dom: Vec<_>, cod: Vec<_>| TestSignature { dom, cod }.to_string();
        assert_eq!(sig(vec![a], vec![b]), "A -> B(8)");
        assert_eq!(sig(vec![a, a], vec![a]), "(A, A) -> A");
        assert_eq!(sig(vec![], vec![b]), "() -> B(8)");
        assert_eq!(sig(vec![a], vec![]), "A -> ()");
        assert_eq!(sig(vec![], vec![]), "() -> ()");
        assert_eq!(sig(vec![a], vec![a, b]), "A -> (A, B(8))");
    }
}

pub trait Theory {
    type Sort;

    const NAME: &'static str;

    fn check<R, W, E: fmt::Display>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<(Self::Sort, u8), E>>,
        W: IntoIterator<Item = Result<(Self::Sort, u8), E>>;
}

pub trait Combinatorial: Theory {
    // Returns the unary itype that chooses an element within the range
    fn havoc(range: &Self::Sort) -> Self;
}

pub trait Sequential: Theory {
    // Returns the unary itype that copies a value within the range
    fn skip(range: &Self::Sort) -> Self;
}

pub trait Differential: Theory {
    // Returns the unary itype that indicates zero derivative
    fn zero(range: &Self::Sort) -> Self;
}

// Helpers for type-checking procedures

// SKIP is unary: it copies exactly one read wire to one write wire of the
// same sort, matching the arity of the base theories' `Id`.
// This is a helper routine; concrete theories are free to implement their own
fn check_skip<R, W, S, E>(range: &S, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<(S, u8), E>>,
    W: IntoIterator<Item = Result<(S, u8), E>>,
    S: Eq + Debug,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match (read.next(), write.next()) {
        (Some(Ok((r, rd))), Some(Ok((w, wd)))) if r != w || &w != range || wd != rd => {
            return Err(format!(
                "SKIP expects exactly one read and one write of sort {:?} and equal degree",
                range
            ));
        }
        (Some(_), Some(_)) => {}
        _ => {
            return Err("SKIP expects exactly one read and one write".to_string());
        }
    }
    if read.next().is_some() || write.next().is_some() {
        return Err("SKIP expects exactly one read and one write".to_string());
    }
    Ok(())
}

// HAVOC is unary: it writes exactly one wire and reads none.
// This is a helper routine; concrete theories are free to implement their own
pub(crate) fn check_havoc<S, R, W, E>(range: &S, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<(S, u8), E>>,
    W: IntoIterator<Item = Result<(S, u8), E>>,
    S: Eq + Debug,
    E: fmt::Display,
{
    if read.into_iter().next().is_some() {
        return Err("HAVOC expects no read wires".to_string());
    }
    let mut write = write.into_iter();
    match write.next() {
        Some(Ok((sort, _))) if &sort != range => {
            return Err(format!("HAVOC expects write of dtype {:?}", range));
        }
        Some(Err(e)) => return Err(e.to_string()),
        None => return Err("HAVOC expects exactly one write wire, got none".to_string()),
        _ => {}
    }
    if write.next().is_some() {
        return Err("HAVOC expects exactly one write wire, got more".to_string());
    }
    Ok(())
}

// ZERO is unary: it writes exactly one wire and reads none.
// This is a helper routine; concrete theories are free to implement their own
pub(crate) fn check_zero<S, R, W, E>(range: &S, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<(S, u8), E>>,
    W: IntoIterator<Item = Result<(S, u8), E>>,
    S: Eq + Debug,
    E: fmt::Display,
{
    if read.into_iter().next().is_some() {
        return Err("ZERO expects no read wires".to_string());
    }
    let mut write = write.into_iter();
    match write.next() {
        Some(Ok((sort, _))) if &sort != range => {
            return Err(format!("ZERO expects write of dtype {:?}", range));
        }
        Some(Err(e)) => return Err(e.to_string()),
        None => return Err("ZERO expects exactly one write wire, got none".to_string()),
        _ => {}
    }
    if write.next().is_some() {
        return Err("ZERO expects exactly one write wire, got more".to_string());
    }
    Ok(())
}

fn next_expect_degree<R, S, E: fmt::Display>(
    iter: &mut R,
    pos: usize,
    degree: u8,
) -> Result<S, String>
where
    R: Iterator<Item = Result<(S, u8), E>>,
{
    let (s, d) = next_with_degree(iter, pos)?;
    if d != degree {
        Err(format!("Arg {pos} expected to be {degree}, got {d}"))
    } else {
        Ok(s)
    }
}

fn next_with_degree<R, S, E: fmt::Display>(iter: &mut R, pos: usize) -> Result<(S, u8), String>
where
    R: Iterator<Item = Result<(S, u8), E>>,
{
    if let Some(item) = iter.next() {
        item.map_err(|e| e.to_string())
    } else {
        Err(format!("Arg {pos} expected, but got none"))
    }
}
