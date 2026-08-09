use std::fmt;

pub mod bv;
pub mod lia;
pub mod lra;
pub mod tensor;

pub use tensor::PyTensor;

pub trait Theory {
    type Sort: fmt::Display; // this global requirement is temporary, until we have error enums.
    // eventually display should never be required by default, neither here not by check

    const NAME: &'static str;

    fn check<R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<Self::Sort> + fmt::Display + Eq,
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

fn read_nxt<R, D, T>(read: &mut R, i: usize, theory: &'static str) -> Result<T, String>
where
    R: Iterator<Item = D>,
    D: TryInto<T> + fmt::Display,
{
    if let Some(d) = read.next() {
        let repr = format!("{d}");
        d.try_into()
            .map_err(|_| format!("Read arg {i} (`{repr}`) not compatible with {theory}"))
    } else {
        Err(format!("Read arg {i} expected, but got none"))
    }
}

fn write_nxt<R, D, T>(write: &mut R, i: usize, theory: &'static str) -> Result<T, String>
where
    R: Iterator<Item = D>,
    D: TryInto<T> + fmt::Display,
{
    if let Some(d) = write.next() {
        let repr = format!("{d}");
        d.try_into()
            .map_err(|_| format!("Write arg {i} (`{repr}`) not compatible with {theory}"))
    } else {
        Err(format!("Write arg {i} expected, but got none"))
    }
}
