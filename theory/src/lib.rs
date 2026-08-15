pub mod any;
pub mod bv;
pub mod lia;
pub mod lra;
pub mod tensor;

use crate::any::Sort;
pub use tensor::PyTensor;

pub trait Theory {
    type Sort;

    const NAME: &'static str;

    fn check<R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<Self::Sort>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>;
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
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
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

fn read_nxt<R, D, T>(read: &mut R, i: usize, theory: &'static str) -> Result<T, String>
where
    R: Iterator<Item = D>,
    D: TryInto<T>,
{
    if let Some(d) = read.next() {
        d.try_into()
            .map_err(|_| format!("Read arg {i} not compatible with {theory}"))
    } else {
        Err(format!("Read arg {i} expected, but got none"))
    }
}

fn write_nxt<R, D, T>(write: &mut R, i: usize, theory: &'static str) -> Result<T, String>
where
    R: Iterator<Item = D>,
    D: TryInto<T>,
{
    if let Some(d) = write.next() {
        d.try_into()
            .map_err(|_| format!("Write arg {i} not compatible with {theory}"))
    } else {
        Err(format!("Write arg {i} expected, but got none"))
    }
}
