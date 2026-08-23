pub mod any;
pub mod bv;
pub mod lia;
pub mod lra;
pub mod tensor;

use std::fmt;
use std::fmt::Debug;
pub use tensor::PyTensor;
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
