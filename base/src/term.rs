use crate::wire::{Interface, Wire};
use std::collections::HashMap;
use std::fmt;

/// A single term corresponds to a single instruction
/// and has an input (`read`) and output (`write`).
///
/// Terms can be over different instruction sets (e.g., pytorch/linear guarded commands).
///
/// A list of terms represents a compute graph. A term is a node in the graph,
/// and it references the input/output edges (read/write wires).
/// [Interface]s are essentially single static assignments.
#[derive(Debug, Clone)]
pub struct Term<D, I> {
    /// The instruction to be executed by this node.
    itype: I,
    /// The outputs of this term.
    write: Interface<D>,
    /// The inputs to this term.
    read: Interface<D>,
}

impl<D, I> Term<D, I> {
    pub fn new(itype: I, write: Interface<D>, read: Interface<D>) -> Self {
        Self { itype, write, read }
    }

    pub fn itype(&self) -> &I {
        &self.itype
    }

    pub fn write(&self) -> &Interface<D> {
        &self.write
    }

    pub fn read(&self) -> &Interface<D> {
        &self.read
    }
}

impl<D: Eq, I> Term<D, I> {
    pub fn function<T, U, W, R>(itype: I, write: W, read: R) -> Result<Self, &'static str>
    where
        T: Into<Wire<D>>,
        U: Into<Wire<D>>,
        W: IntoIterator<Item = T>,
        R: IntoIterator<Item = U>,
    {
        Ok(Self::new(
            itype,
            Interface::unique(write)?,
            Interface::sequence(read)?,
        ))
    }

    pub fn constant<T, W>(itype: I, write: W) -> Result<Self, &'static str>
    where
        T: Into<Wire<D>>,
        W: IntoIterator<Item = T>,
    {
        Ok(Self::new(
            itype,
            Interface::unique(write)?,
            Interface::none(),
        ))
    }
}

#[macro_export]
macro_rules! term {
    ($itype:tt, $write:expr) => {
        Term::constant($itype, $write)
    };

    ($itype:tt, $write:expr, $read:expr) => {
        Term::function($itype, $write, $read)
    };
}

impl<D: fmt::Display, I: fmt::Display> fmt::Display for Term<D, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const BOLD: &str = "\x1b[1m";
        const RESET: &str = "\x1b[0m";
        write!(f, "{} ", self.itype,)?;
        write!(
            f,
            "{}",
            self.write
                .ids()
                .map(|a| format!("w{a}"))
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        write!(
            f,
            "; {}",
            self.read
                .ids()
                .map(|a| format!("w{a}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

pub(crate) struct Block<D, I> {
    pub(crate) terms: Vec<Term<D, I>>,
    pub(crate) read: Interface<D>,
    pub(crate) write: Interface<D>,
}

impl<D: Eq + Clone, I> Block<D, I> {
    pub fn try_from_iter<V: IntoIterator<Item = Term<D, I>>>(
        iter: V,
    ) -> Result<Self, &'static str> {
        let mut read: HashMap<usize, D> = HashMap::new();
        let mut write: HashMap<usize, D> = HashMap::new();
        let vec: Vec<Term<D, I>> = Vec::from_iter(iter);

        for term in vec.iter() {
            for (rd, dtype) in term.read().wires().map(Into::into) {
                let expected_dtype = write.get(&rd);
                // if it hasn't been written before in the block, then it's read
                if expected_dtype.is_none() {
                    read.insert(rd, dtype.clone());
                } else if expected_dtype.is_some_and(|d| d != dtype) {
                    return Err("dtype mismatch");
                }
            }
            for (wt, dtype) in term.write().wires().map(Into::into) {
                if read.contains_key(&wt) {
                    return Err("read before write");
                }
                if write.insert(wt, dtype.clone()).is_some() {
                    return Err("write after write");
                }
            }
        }

        debug_assert!(read.keys().all(|k| !write.contains_key(k)));

        Ok(Block {
            terms: vec,
            read: Interface::from_wires_unchecked(read),
            write: Interface::from_wires_unchecked(write),
        })
    }
}
