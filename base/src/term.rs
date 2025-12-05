use crate::wire::{Interface, Wire};
use std::collections::{HashMap, HashSet};
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
    pub fn new_unchecked(itype: I, write: Interface<D>, read: Interface<D>) -> Self {
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
        Ok(Self::new_unchecked(
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
        Ok(Self::new_unchecked(
            itype,
            Interface::unique(write)?,
            Interface::empty(),
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

#[derive(Debug, Clone)]
pub struct Block<D, I> {
    terms: Vec<Term<D, I>>,
    read: Interface<D>,
    write: Interface<D>,
}

impl<D, I> Block<D, I> {
    pub fn iter(&self) -> impl Iterator<Item = &Term<D, I>> {
        self.terms.iter()
    }

    pub fn read(&self) -> &Interface<D> {
        &self.read
    }

    pub fn write(&self) -> &Interface<D> {
        &self.write
    }
}

impl<'a, D, I> IntoIterator for &'a Block<D, I> {
    type Item = &'a Term<D, I>;
    type IntoIter = std::slice::Iter<'a, Term<D, I>>;

    fn into_iter(self) -> Self::IntoIter {
        self.terms.iter()
    }
}

impl<D: Eq + Clone, I> Block<D, I> {
    pub(crate) fn try_from_iter<V: IntoIterator<Item = Term<D, I>>>(
        iter: V,
    ) -> Result<Self, &'static str> {
        let mut read_set: HashSet<usize> = HashSet::new();
        let mut write_to_dtype: HashMap<usize, &D> = HashMap::new();

        let mut read: Vec<Wire<D>> = Vec::new();
        let mut write: Vec<Wire<D>> = Vec::new();

        let terms: Vec<Term<D, I>> = Vec::from_iter(iter);

        for term in terms.iter() {
            for (rd, dtype) in term.read().wires().map(Into::into) {
                let expected_dtype = write_to_dtype.get(&rd);
                // if it hasn't been written before in the block, then it's read
                if expected_dtype.is_none() {
                    read_set.insert(rd);
                    read.push((rd, dtype.clone()).into());
                } else if expected_dtype.is_some_and(|&d| d != dtype) {
                    return Err("dtype mismatch");
                }
            }

            for (wt, dtype) in term.write().wires().map(Into::into) {
                if read_set.contains(&wt) {
                    return Err("read before write");
                }
                write.push((wt, dtype.clone()).into());
                if write_to_dtype.insert(wt, dtype).is_some() {
                    return Err("write after write");
                }
            }
            write.extend(term.write().wires().cloned().map(Into::into));
        }

        debug_assert!(read_set.iter().all(|k| !write_to_dtype.contains_key(k)));

        Ok(Block {
            terms,
            read: Interface::from_wires_unchecked(read),
            write: Interface::from_wires_unchecked(write),
        })
    }
}
