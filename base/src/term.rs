use crate::wire::{Interface, Wire};
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug};
use theory::Theory;

/// A single term corresponds to a single instruction
/// and has an input (`read`) and output (`write`).
///
/// Terms can be over different instruction sets (e.g., pytorch/linear guarded commands).
///
/// A list of terms represents a compute graph. A term is a node in the graph,
/// and it references the input/output edges (read/write wires).
/// [Interface]s are essentially single static assignments.
#[derive(Debug, Clone)]
pub struct Term<T: Theory> {
    /// The instruction to be executed by this node.
    itype: T,
    /// The outputs of this term.
    write: Interface<T::DType>,
    /// The inputs to this term.
    read: Interface<T::DType>,
}

impl<T: Theory> Term<T> {
    pub fn new_unchecked(itype: T, write: Interface<T::DType>, read: Interface<T::DType>) -> Self {
        Self { itype, write, read }
    }

    pub fn itype(&self) -> &T {
        &self.itype
    }

    pub fn write(&self) -> &Interface<T::DType> {
        &self.write
    }

    pub fn read(&self) -> &Interface<T::DType> {
        &self.read
    }
}

impl<T> Term<T>
where
    T: Theory,
    T::DType: Eq,
{
    pub fn function<D, U, W, R>(itype: T, write: W, read: R) -> Result<Self, &'static str>
    where
        D: Into<Wire<T::DType>>,
        U: Into<Wire<T::DType>>,
        W: IntoIterator<Item = D>,
        R: IntoIterator<Item = U>,
    {
        Ok(Self::new_unchecked(
            itype,
            Interface::unique(write)?,
            Interface::sequence(read)?,
        ))
    }

    pub fn constant<D, W>(itype: T, write: W) -> Result<Self, &'static str>
    where
        D: Into<Wire<T::DType>>,
        W: IntoIterator<Item = D>,
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
    ($itype:expr, $write:expr) => {
        Term::constant($itype, $write)
    };

    ($itype:expr, $write:expr, $read:expr) => {
        Term::function($itype, $write, $read)
    };
}

impl<TH: Theory> fmt::Display for Term<TH>
where
    TH: fmt::Display,
    TH::DType: fmt::Display,
{
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
pub struct Block<T: Theory> {
    terms: Vec<Term<T>>,
    read: Interface<T::DType>,
    write: Interface<T::DType>,
}

impl<T: Theory> Block<T> {
    pub fn iter(&self) -> impl Iterator<Item = &Term<T>> {
        self.terms.iter()
    }

    /// Returns a reference to the *read interface* of the block.
    ///
    /// The read interface lists all wires that must be provided externally
    /// for the block to operate, and are not written internally by the block.
    /// These wires are inputs required by the block as a whole.
    pub fn read(&self) -> &Interface<T::DType> {
        &self.read
    }

    /// Returns a reference to the *write interface* of the block.
    ///
    /// The write interface lists all wires that the block writes. These wires represent
    /// the outputs of the block as a whole; they can all be read outside the block.
    pub fn write(&self) -> &Interface<T::DType> {
        &self.write
    }

    /// Return a reference to the n-th term in the block
    pub fn get(&self, n: usize) -> Option<&Term<T>> {
        self.terms.get(n)
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub(crate) fn empty() -> Self {
        Self {
            terms: Vec::new(),
            read: Interface::empty(),
            write: Interface::empty(),
        }
    }
}

impl<'a, T: Theory> IntoIterator for &'a Block<T> {
    type Item = &'a Term<T>;
    type IntoIter = std::slice::Iter<'a, Term<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.terms.iter()
    }
}

impl<T: Theory> Block<T>
where
    T::DType: Eq + Clone,
{
    pub(crate) fn try_from_iter<V: IntoIterator<Item = Term<T>>>(
        iter: V,
    ) -> Result<Self, &'static str> {
        let mut read_set: HashSet<usize> = HashSet::new();
        let mut write_to_dtype: HashMap<usize, &T::DType> = HashMap::new();

        let mut read: Vec<Wire<T::DType>> = Vec::new();
        let mut write: Vec<Wire<T::DType>> = Vec::new();

        let terms: Vec<Term<T>> = Vec::from_iter(iter);

        for term in terms.iter() {
            for rd in term.read().wires() {
                let expected_dtype = write_to_dtype.get(&rd.id());
                // if it hasn't been written before in the block, then it's read
                if expected_dtype.is_none() {
                    read_set.insert(rd.id());
                    read.push(rd.clone());
                } else if expected_dtype.is_some_and(|&d| d != rd.dtype()) {
                    return Err("dtype mismatch");
                }
            }

            for wt in term.write().wires() {
                if read_set.contains(&wt.id()) {
                    return Err("read before write");
                }
                write.push(wt.clone());
                if write_to_dtype.insert(wt.id(), wt.dtype()).is_some() {
                    return Err("write after write");
                }
            }
            write.extend(term.write().wires().cloned());
        }

        debug_assert!(read_set.iter().all(|k| !write_to_dtype.contains_key(k)));

        Ok(Block {
            terms,
            read: Interface::from_wires_unchecked(read),
            write: Interface::from_wires_unchecked(write),
        })
    }
}
