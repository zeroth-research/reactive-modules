pub mod any;
pub mod bv;
pub mod lia;
pub mod lra;
pub mod tensor;

use crate::any::Sort;
use std::fmt;
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
    const HAVOC: Self;
}

pub trait Sequential: Theory {
    const SKIP: Self;
}

pub trait Differential: Theory {
    const ZERO: Self;
}

// Helpers for type-checking procedures

// SKIP is unary: it copies exactly one read wire to one write wire of the
// same sort, matching the arity of the base theories' `Id`.
fn check_skip<R, W, E>(read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<(Sort, u8), E>>,
    W: IntoIterator<Item = Result<(Sort, u8), E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match (read.next(), write.next()) {
        (Some(Ok(r)), Some(Ok(w))) if r == w => {}
        _ => {
            return Err("SKIP expects exactly one read and one write of the same sort".to_string());
        }
    }
    if read.next().is_some() || write.next().is_some() {
        return Err("SKIP expects exactly one read and one write".to_string());
    }
    Ok(())
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
