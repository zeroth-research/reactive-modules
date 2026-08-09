use std::array::from_fn;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone)]
pub struct Wire<S> {
    id: usize,
    dtype: S,
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

impl<S> Wire<S> {
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn dtype(&self) -> &S {
        &self.dtype
    }

    pub fn new(dtype: S) -> Self {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        if id == usize::MAX {
            panic!("wire id overflow");
        }
        Self { id, dtype }
    }
}

impl<S> PartialEq for Wire<S> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<S> Eq for Wire<S> {}

impl<S> Hash for Wire<S> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<S> From<Wire<S>> for (usize, S) {
    fn from(w: Wire<S>) -> Self {
        (w.id, w.dtype)
    }
}

impl<'a, S> From<&'a Wire<S>> for (usize, &'a S) {
    fn from(w: &'a Wire<S>) -> Self {
        (w.id, &w.dtype)
    }
}

/// An interface consisting of `N`-tuples of wires of data type `D`.
///
/// # Overview
/// `Interface<D, N>` represents a *local bundle* of wires.
/// Conceptually, it behaves like a sequence of tuples, where each tuple
/// contains exactly `N` elements of type `Wire<D>`. Each tuple can be seen
/// an element of type [Wire<D>; N], where all wires within each tuple are
/// guaranteed to have the same dtype.
///
/// # Type Parameters
/// - `D`: the data type carried by each wire.
/// - `N`: the arity of the interface (number of wires in each tuple).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interface<S, const N: usize = 1> {
    wires: [Vec<Wire<S>>; N],
}

impl<S, const N: usize> Interface<S, N> {
    pub fn empty() -> Interface<S, N> {
        Self {
            wires: [(); N].map(|_| Vec::new()),
        }
    }
}

impl<S, const N: usize> Default for Interface<S, N> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<S> Interface<S, 1> {
    pub fn as_slice(&self) -> &[Wire<S>] {
        self.wires[0].as_slice()
    }
}

impl<S> Interface<S, 2> {
    pub fn latched(&self) -> &[Wire<S>] {
        self.wires[0].as_slice()
    }

    pub fn next(&self) -> &[Wire<S>] {
        self.wires[1].as_slice()
    }
}

impl<S, const N: usize> Interface<S, N> {
    pub fn wire(&self, time: usize, index: usize) -> Option<&Wire<S>> {
        self.wires.get(time).and_then(|v| v.get(index))
    }

    pub fn entry(&self, index: usize) -> Option<[&Wire<S>; N]> {
        (index < self.len()).then(|| from_fn(|i| &self.wires[i][index]))
    }
}

impl<S, const N: usize> Interface<S, N> {
    pub fn iter(&self) -> impl Iterator<Item = [&Wire<S>; N]> {
        IterRef {
            iters: std::array::from_fn(|i| self.wires[i].iter()),
        }
    }

    pub fn wires(&self) -> impl Iterator<Item = &Wire<S>> {
        self.wires.iter().flatten()
    }

    pub fn ids(&self) -> impl Iterator<Item = usize> {
        self.wires().map(Wire::id)
    }
}

impl<S, const N: usize> Interface<S, N> {
    /// Returns true if the wire indices of self are also indices of other, regardless of their type.
    /// This function runs in place, in O(self.len() * other.len()) time
    pub fn is_subset<const M: usize>(&self, other: &Interface<S, M>) -> bool {
        for a in self.ids() {
            if other.ids().all(|b| a != b) {
                return false;
            }
        }
        true
    }

    /// Returns true if the wire indices of self are disjoint from the indices of other, regardless of their type.
    /// This function runs in place, in O(self.len() * other.len()) time
    pub fn is_disjoint(&self, other: &Interface<S>) -> bool {
        for a in self.ids() {
            if other.ids().any(|b| a == b) {
                return false;
            }
        }
        true
    }

    pub fn len(&self) -> usize {
        if N > 0 { self.wires[0].len() } else { 0 }
    }

    pub fn is_empty(&self) -> bool {
        N == 0 || self.wires[0].is_empty()
    }
}

pub struct IterOwned<S, const N: usize> {
    iters: [std::vec::IntoIter<Wire<S>>; N],
}

impl<S, const N: usize> Iterator for IterOwned<S, N> {
    type Item = [Wire<S>; N];

    fn next(&mut self) -> Option<Self::Item> {
        let out: [Option<Wire<S>>; N] = from_fn(|i| self.iters[i].next());
        debug_assert!(out.iter().all(Option::is_some) || out.iter().all(Option::is_none));
        (N != 0 && out[0].is_some()).then(|| out.map(Option::unwrap))
    }
}

pub struct IterRef<'a, S, const N: usize> {
    iters: [std::slice::Iter<'a, Wire<S>>; N],
}

impl<'a, S, const N: usize> Iterator for IterRef<'a, S, N> {
    type Item = [&'a Wire<S>; N];

    fn next(&mut self) -> Option<Self::Item> {
        let out: [Option<&Wire<S>>; N] = from_fn(|i| self.iters[i].next());
        debug_assert!(out.iter().all(Option::is_some) || out.iter().all(Option::is_none));
        (N != 0 && out[0].is_some()).then(|| out.map(Option::unwrap))
    }
}

impl<D, const N: usize> IntoIterator for Interface<D, N> {
    type Item = [Wire<D>; N];
    type IntoIter = IterOwned<D, N>;

    fn into_iter(self) -> Self::IntoIter {
        IterOwned {
            iters: self.wires.map(|c| c.into_iter()),
        }
    }
}

impl<'a, S, const N: usize> IntoIterator for &'a Interface<S, N> {
    type Item = [&'a Wire<S>; N];
    type IntoIter = IterRef<'a, S, N>;

    fn into_iter(self) -> Self::IntoIter {
        IterRef {
            iters: from_fn(|i| self.wires[i].iter()),
        }
    }
}

/// from_iter is unchecked. Use at your own risk
impl<S: Eq, T: Into<[Wire<S>; N]>, const N: usize> FromIterator<T> for Interface<S, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::try_from_iter(iter).unwrap()
    }
}

impl<S: Eq, T: Into<Wire<S>>> From<T> for Interface<S> {
    fn from(t: T) -> Self {
        Self::from_wires_unchecked([t])
    }
}

// returns the wire at position (0,0) and throws away the rest
impl<S: Eq> TryFrom<Interface<S>> for Wire<S> {
    type Error = String;

    fn try_from(x: Interface<S>) -> Result<Self, Self::Error> {
        let mut it = x.wires.into_iter().flatten();
        it.next()
            .ok_or("There is no wire at position (0, 0)".into())
    }
}

// returns the wire at position (0,0) and throws away the rest
impl<S: Eq> TryFrom<Interface<S>> for (usize, S) {
    type Error = String;

    fn try_from(x: Interface<S>) -> Result<Self, Self::Error> {
        let mut it = x.wires.into_iter().flatten().map(Into::into);
        it.next()
            .ok_or("There is no wire at position (0, 0)".into())
    }
}

impl<D: Eq, const N: usize> Interface<D, N> {
    fn collect<T: Into<[Wire<D>; N]>, I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut wires: [Vec<Wire<D>>; N] = match iter.size_hint() {
            (_, Some(upper)) => [(); N].map(|_| Vec::with_capacity(upper)),
            _ => [(); N].map(|_| Vec::new()),
        };

        for indexed_wire in iter.map(Into::into) {
            for (to, from) in wires.iter_mut().zip(indexed_wire) {
                to.push(from)
            }
        }

        Self { wires }
    }

    pub fn try_from_iter<T: Into<[Wire<D>; N]>, I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<Self, String> {
        let interface = Self::collect(iter);

        let mut w_to_dtype: HashMap<usize, &D> = HashMap::new();
        for wires in interface.iter() {
            for (id, dtype) in wires.map(Into::into) {
                if dtype != wires[0].dtype() {
                    return Err(format!(
                        "Wire {} has wrong dtype, should match dtype of wire {}",
                        id,
                        wires[0].id()
                    ));
                }
                if w_to_dtype.insert(id, dtype).is_some_and(|o| o != dtype) {
                    return Err(format!("Wire {} seen twice with different dtypes", id));
                }
            }
        }

        Ok(interface)
    }

    pub(crate) fn from_iter_unchecked<T: Into<[Wire<D>; N]>, I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Self {
        let interface = Self::collect(iter);

        #[cfg(debug_assertions)]
        {
            // wires dtype must be consistent
            let mut w_to_dtype: HashMap<usize, &D> = HashMap::new();
            for wires in interface.iter() {
                for (id, dtype) in wires.map(Into::into) {
                    debug_assert!(dtype == wires[0].dtype());
                    debug_assert!(w_to_dtype.insert(id, dtype).is_none_or(|o| o == dtype));
                }
            }
        }

        interface
    }
}

impl<S: Eq> Interface<S> {
    pub(crate) fn sequence<T: Into<Wire<S>>, I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<Self, String> {
        Self::try_from_iter(iter.into_iter().map(|w| [w.into()]))
    }

    pub(crate) fn unique<T: Into<Wire<S>>, I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<Self, String> {
        let interface = Self::collect(iter.into_iter().map(|w| [w.into()]));

        let mut ids: HashSet<usize> = HashSet::new();
        for id in interface.wires().map(Wire::id) {
            if !ids.insert(id) {
                return Err(format!("duplicate id {}", id));
            }
        }

        Ok(interface)
    }

    pub(crate) fn from_wires_unchecked<T: Into<Wire<S>>, I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Self {
        Self::from_iter_unchecked(iter.into_iter().map(|w| [w.into()]))
    }
}

impl<S: fmt::Display, const N: usize> fmt::Display for Interface<S, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for wires in self {
            if !first {
                write!(f, "; ")?;
            }
            write!(f, "w{} ", wires[0].id)?;
            for w in wires.iter().take(N).skip(1) {
                write!(f, ", w{} ", w.id)?;
            }
            write!(f, ": {}", wires[0].dtype)?;
            first = false;
        }
        Ok(())
    }
}

impl<S: fmt::Display> fmt::Display for Wire<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} : {}", self.id, self.dtype)?;
        Ok(())
    }
}
