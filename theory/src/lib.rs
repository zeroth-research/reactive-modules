use std::fmt;
use std::ops::Deref;

pub mod any;
pub mod bv;
pub mod lia;
pub mod lra;

/// Wrapper around [`tch::Tensor`] that implements [`Clone`] via [`tch::Tensor::shallow_clone`].
#[derive(Debug)]
pub struct Tensor(pub tch::Tensor);

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor(self.0.shallow_clone())
    }
}

impl Deref for Tensor {
    type Target = tch::Tensor;
    fn deref(&self) -> &tch::Tensor {
        &self.0
    }
}

impl From<tch::Tensor> for Tensor {
    fn from(t: tch::Tensor) -> Self {
        Tensor(t)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// tch::Tensor's C++ API is safe for concurrent reads; consistent with PyTensor in python crate.
unsafe impl Sync for Tensor {}

pub trait Theory {
    // TODO: in torch, from where we took this name (I think), dtype refers to
    // the type of the element in the tensor (*d*ata type). Maybe we should
    // consider renaming this to "Types" or something, to avoid confusion.
    type DType;

    fn check<R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<Self::DType> + fmt::Display,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>;
}

// Helpers for type-checking procedures

fn read_nxt<R, D, T>(read: &mut R, i: usize) -> Result<T, String>
where
    R: Iterator<Item = D>,
    D: TryInto<T> + fmt::Display,
{
    if let Some(d) = read.next() {
        let repr = format!("{d}");
        d.try_into()
            .map_err(|_| format!("Read arg {i} (`{repr}`) not compatible with Theory"))
    } else {
        Err(format!("Read arg {i} expected, but got none"))
    }
}

fn write_nxt<R, D, T>(write: &mut R, i: usize) -> Result<T, String>
where
    R: Iterator<Item = D>,
    D: TryInto<T> + fmt::Display,
{
    if let Some(d) = write.next() {
        let repr = format!("{d}");
        d.try_into()
            .map_err(|_| format!("Write arg {i} (`{repr}`) not compatible with Theory"))
    } else {
        Err(format!("Write arg {i} expected, but got none"))
    }
}
