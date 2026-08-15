use crate::wire::{Interface, Wire};
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug};
use std::vec;
use theory::{Differential, Sequential, Theory, any};

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
    write: Interface<T::Sort>,
    /// The inputs to this term.
    read: Interface<T::Sort>,
}

impl<T: Theory> Term<T> {
    pub fn new_unchecked(itype: T, write: Interface<T::Sort>, read: Interface<T::Sort>) -> Self {
        Self { itype, write, read }
    }

    pub fn itype(&self) -> &T {
        &self.itype
    }

    pub fn write(&self) -> &Interface<T::Sort> {
        &self.write
    }

    pub fn read(&self) -> &Interface<T::Sort> {
        &self.read
    }
}

impl<T> Term<T>
where
    T: Theory,
    T::Sort: Eq + Clone,
{
    pub fn function<D, U, W, R>(itype: T, write: W, read: R) -> Result<Self, String>
    where
        D: Into<Wire<T::Sort>>,
        U: Into<Wire<T::Sort>>,
        W: IntoIterator<Item = D>,
        R: IntoIterator<Item = U>,
    {
        let term =
            Self::new_unchecked(itype, Interface::unique(write)?, Interface::sequence(read)?);

        if term.read().ids().any(|i| term.write.ids().any(|j| i == j)) {
            return Err("Term reads and writes the same wire".into());
        }

        // type-check the term. We do it only after contruction of the term, because type-checking
        // would consume the values of `write` and `read` otherwise
        let r = term.read.as_slice().iter().map(|w| w.dtype().clone());
        let w = term.write.as_slice().iter().map(|w| w.dtype().clone());
        match term.itype.check(r, w) {
            Ok(_) => Ok(term),
            Err(e) => Err(e),
        }
    }

    pub fn constant<D, W>(itype: T, write: W) -> Result<Self, String>
    where
        D: Into<Wire<T::Sort>>,
        W: IntoIterator<Item = D>,
    {
        let term = Self::new_unchecked(itype, Interface::unique(write)?, Interface::empty());

        if term.read().ids().any(|i| term.write.ids().any(|j| i == j)) {
            return Err("Term reads and writes the same wire".into());
        }

        // type-check the term. We do it only after contruction of the term, because type-checking
        // would consume the values of `write` and `read` otherwise
        let r = term.read.as_slice().iter().map(|w| w.dtype().clone());
        let w = term.write.as_slice().iter().map(|w| w.dtype().clone());
        match term.itype.check(r, w) {
            Ok(_) => Ok(term),
            Err(e) => Err(e),
        }
    }
}

impl<T: Theory> Term<T> {
    /// Re-types a term along an infallible theory embedding (`U: Into<T>`).
    ///
    /// Only the instruction is converted; the wires are untouched, so a term
    /// that was well-formed over `U` stays well-formed over `T` and no
    /// re-validation is needed.
    fn convert<U>(term: Term<U>) -> Self
    where
        U: Theory<Sort = T::Sort> + Into<T>,
    {
        Self::new_unchecked(term.itype.into(), term.write, term.read)
    }

    fn try_convert<U, E>(term: Term<U>) -> Result<Self, E>
    where
        U: Theory<Sort = T::Sort> + TryInto<T, Error = E>,
    {
        let itype = term.itype.try_into()?;
        Ok(Self::new_unchecked(itype, term.write, term.read))
    }
}

impl From<Term<any::Sequential>> for Term<any::Any> {
    fn from(term: Term<any::Sequential>) -> Self {
        Self::convert(term)
    }
}

impl From<Term<any::Combinatorial>> for Term<any::Any> {
    fn from(term: Term<any::Combinatorial>) -> Self {
        Self::convert(term)
    }
}

impl From<Term<any::Differential>> for Term<any::Any> {
    fn from(term: Term<any::Differential>) -> Self {
        Self::convert(term)
    }
}

impl TryFrom<Term<any::Any>> for Term<any::Sequential> {
    type Error = String;
    fn try_from(term: Term<any::Any>) -> Result<Self, Self::Error> {
        Self::try_convert(term).map_err(|e| e.to_string())
    }
}

impl TryFrom<Term<any::Any>> for Term<any::Combinatorial> {
    type Error = String;
    fn try_from(term: Term<any::Any>) -> Result<Self, Self::Error> {
        Self::try_convert(term).map_err(|e| e.to_string())
    }
}

impl TryFrom<Term<any::Any>> for Term<any::Differential> {
    type Error = String;
    fn try_from(term: Term<any::Any>) -> Result<Self, Self::Error> {
        Self::try_convert(term).map_err(|e| e.to_string())
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
    TH::Sort: fmt::Display,
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
    read: Interface<T::Sort>,
    write: Interface<T::Sort>,
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
    pub fn read(&self) -> &Interface<T::Sort> {
        &self.read
    }

    /// Returns a reference to the *write interface* of the block.
    ///
    /// The write interface lists all wires that the block writes. These wires represent
    /// the outputs of the block as a whole; they can all be read outside the block.
    pub fn write(&self) -> &Interface<T::Sort> {
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

impl<T: Theory> IntoIterator for Block<T> {
    type Item = Term<T>;
    type IntoIter = vec::IntoIter<Term<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.terms.into_iter()
    }
}

impl<T: Theory> Block<T>
where
    T::Sort: Eq + Clone,
{
    /// Builds a block from an iterator of *fallible* elements, mirroring the
    /// standard library's `FromIterator<Result<A, E>> for Result<V, E>`: the
    /// first `Err` element short-circuits and is returned as the block's
    /// error. Elements over a convertible theory (`U: Into<T>`) are embedded
    /// into `T`. Infallible sources wrap their terms with `.map(Ok)`.
    ///
    /// The error type is fixed to `String` — the crate's uniform error
    /// model — so that `Ok` needs no annotation at infallible call sites.
    pub(crate) fn try_from_iter<U, V>(iter: V) -> Result<Self, String>
    where
        U: Theory<Sort = T::Sort> + Into<T>,
        V: IntoIterator<Item = Result<Term<U>, String>>,
    {
        let mut read_set: HashSet<usize> = HashSet::new();
        let mut write_to_dtype: HashMap<usize, &T::Sort> = HashMap::new();

        let mut read: Vec<Wire<T::Sort>> = Vec::new();
        let mut write: Vec<Wire<T::Sort>> = Vec::new();

        let terms: Vec<Term<T>> = iter
            .into_iter()
            .map(|t| t.map(Term::convert))
            .collect::<Result<_, _>>()?;

        for term in terms.iter() {
            for rd in term.read().wires() {
                let expected_dtype = write_to_dtype.get(&rd.id());
                // if it hasn't been written before in the block, then it's read
                if expected_dtype.is_none() {
                    read_set.insert(rd.id());
                    read.push(rd.clone());
                } else if expected_dtype.is_some_and(|&d| d != rd.dtype()) {
                    return Err(format!(
                        "Wire {} seen multiple times with different dtype",
                        rd.id()
                    ));
                }
            }

            for wt in term.write().wires() {
                if read_set.contains(&wt.id()) {
                    return Err(format!(
                        "Wire {} is read by a term preceding the term that writes into this wire",
                        wt.id()
                    ));
                }
                write.push(wt.clone());
                if write_to_dtype.insert(wt.id(), wt.dtype()).is_some() {
                    return Err(format!("Wire {} is written more than once", wt.id()));
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

impl<D: Differential> Block<D>
where
    D::Sort: Clone + Eq,
{
    /// Builds a block of one `ZERO` term per write wire, so that theories can
    /// keep `ZERO` at a fixed arity and each term stays a single-wire node in
    /// the compute graph.
    pub(crate) fn zero<W: IntoIterator<Item = Wire<D::Sort>>>(write: W) -> Result<Self, String> {
        Block::try_from_iter(write.into_iter().map(|w| Term::constant(D::ZERO, [w])))
    }
}

impl<J: Sequential> Block<J>
where
    J::Sort: Clone + Eq,
{
    /// Builds a block of one `SKIP` term per `(write, read)` wire pair, so
    /// that theories can keep `SKIP` at a fixed arity and each pair is an
    /// independent edge in the compute graph (no spurious dependency of one
    /// write on all the other reads).
    pub(crate) fn skip<
        W: IntoIterator<Item = Wire<J::Sort>>,
        R: IntoIterator<Item = Wire<J::Sort>>,
    >(
        write: W,
        read: R,
    ) -> Result<Self, String> {
        let mut write = write.into_iter();
        let mut read = read.into_iter();
        Block::try_from_iter(std::iter::from_fn(move || {
            match (write.next(), read.next()) {
                (Some(w), Some(r)) => Some(Term::function(J::SKIP, [w], [r])),
                (None, None) => None,
                _ => Some(Err("skip requires as many write as read wires".to_string())),
            }
        }))
    }
}

impl<T: Theory> fmt::Display for Block<T>
where
    T: fmt::Display,
    T::Sort: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            self.terms
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}
