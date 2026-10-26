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

#[derive(Copy, Clone)]
pub enum MatName {
    // symbolic name, e.g., `N` in `Mat(Int, N, 5)`
    Sym(Name),
    // concrete constant, e.g., `8` in `Mat(Int, N, 5)`
    Const(usize),
}

#[derive(Copy, Clone)]
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

    impl MultiSort for TestSort {
        const NAME: &'static str = "TestSort";
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
