use std::array::from_fn;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Debug;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wire<D> {
    id: usize,
    dtype: D,
}

impl<D> Wire<D> {
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn dtype(&self) -> &D {
        &self.dtype
    }

    pub fn new(id: usize, dtype: D) -> Self {
        Self { id, dtype }
    }
}

impl<D> From<Wire<D>> for (usize, D) {
    fn from(w: Wire<D>) -> Self {
        (w.id, w.dtype)
    }
}

impl<'a, D> From<&'a Wire<D>> for (usize, &'a D) {
    fn from(w: &'a Wire<D>) -> Self {
        (w.id, &w.dtype)
    }
}

impl<D> From<(usize, D)> for Wire<D> {
    fn from(w: (usize, D)) -> Self {
        Self::new(w.0, w.1)
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
pub struct Interface<D, const N: usize = 1> {
    wires: [Vec<Wire<D>>; N],
}

impl<D, const N: usize> Interface<D, N> {
    pub fn empty() -> Interface<D, N> {
        Self {
            wires: [(); N].map(|_| Vec::new()),
        }
    }
}

impl<D, const N: usize> Default for Interface<D, N> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<D: Eq> Interface<D> {
    pub fn single(id: usize, dtype: D) -> Interface<D> {
        Self::from_iter_unchecked([[Wire::new(id, dtype)]])
    }
}

impl<D> Interface<D, 1> {
    pub fn as_slice(&self) -> &[Wire<D>] {
        self.wires[0].as_slice()
    }
}

impl<D> Interface<D, 2> {
    pub fn latched(&self) -> &[Wire<D>] {
        self.wires[0].as_slice()
    }

    pub fn next(&self) -> &[Wire<D>] {
        self.wires[1].as_slice()
    }
}

impl<D, const N: usize> Interface<D, N> {
    pub fn wire(&self, time: usize, index: usize) -> Option<&Wire<D>> {
        self.wires.get(time).and_then(|v| v.get(index))
    }

    pub fn entry(&self, index: usize) -> Option<[&Wire<D>; N]> {
        (index < self.len()).then(|| from_fn(|i| &self.wires[i][index]))
    }
}

impl<D, const N: usize> Interface<D, N> {
    pub fn iter(&self) -> impl Iterator<Item = [&Wire<D>; N]> {
        IterRef {
            iters: std::array::from_fn(|i| self.wires[i].iter()),
        }
    }

    pub fn wires(&self) -> impl Iterator<Item = &Wire<D>> {
        self.wires.iter().flatten()
    }

    pub fn ids(&self) -> impl Iterator<Item = usize> {
        self.wires().map(Wire::id)
    }
}

impl<D, const N: usize> Interface<D, N> {
    /// Returns true if the wire indices of self are also indices of other, regardless of their type.
    /// This function runs in place, in O(self.len() * other.len()) time
    pub fn is_subset<const M: usize>(&self, other: &Interface<D, M>) -> bool {
        for a in self.ids() {
            if other.ids().all(|b| a != b) {
                return false;
            }
        }
        true
    }

    /// Returns true if the wire indices of self are disjoint from the indices of other, regardless of their type.
    /// This function runs in place, in O(self.len() * other.len()) time
    pub fn is_disjoint(&self, other: &Interface<D>) -> bool {
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

pub struct IterOwned<D, const N: usize> {
    iters: [std::vec::IntoIter<Wire<D>>; N],
}

impl<D, const N: usize> Iterator for IterOwned<D, N> {
    type Item = [Wire<D>; N];

    fn next(&mut self) -> Option<Self::Item> {
        let out: [Option<Wire<D>>; N] = from_fn(|i| self.iters[i].next());
        debug_assert!(out.iter().all(Option::is_some) || out.iter().all(Option::is_none));
        (N != 0 && out[0].is_some()).then(|| out.map(Option::unwrap))
    }
}

pub struct IterRef<'a, D, const N: usize> {
    iters: [std::slice::Iter<'a, Wire<D>>; N],
}

impl<'a, D, const N: usize> Iterator for IterRef<'a, D, N> {
    type Item = [&'a Wire<D>; N];

    fn next(&mut self) -> Option<Self::Item> {
        let out: [Option<&Wire<D>>; N] = from_fn(|i| self.iters[i].next());
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

impl<'a, D, const N: usize> IntoIterator for &'a Interface<D, N> {
    type Item = [&'a Wire<D>; N];
    type IntoIter = IterRef<'a, D, N>;

    fn into_iter(self) -> Self::IntoIter {
        IterRef {
            iters: from_fn(|i| self.wires[i].iter()),
        }
    }
}

/// from_iter is unchecked. Use at your own risk
impl<D: Eq, T: Into<[Wire<D>; N]>, const N: usize> FromIterator<T> for Interface<D, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::try_from_iter(iter).unwrap()
    }
}

impl<D: Eq, T: Into<Wire<D>>> From<T> for Interface<D> {
    fn from(t: T) -> Self {
        Self::from_wires_unchecked([t])
    }
}

// returns the wire at position (0,0) and throws away the rest
impl<D: Eq> TryFrom<Interface<D>> for Wire<D> {
    type Error = ();

    fn try_from(x: Interface<D>) -> Result<Self, Self::Error> {
        let mut it = x.wires.into_iter().flatten();
        it.next().ok_or(())
    }
}

// returns the wire at position (0,0) and throws away the rest
impl<D: Eq> TryFrom<Interface<D>> for (usize, D) {
    type Error = ();

    fn try_from(x: Interface<D>) -> Result<Self, Self::Error> {
        let mut it = x.wires.into_iter().flatten().map(Into::into);
        it.next().ok_or(())
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
    ) -> Result<Self, &'static str> {
        let interface = Self::collect(iter);

        let mut w_to_dtype: HashMap<usize, &D> = HashMap::new();
        for wires in interface.iter() {
            for (id, dtype) in wires.map(Into::into) {
                if dtype != wires[0].dtype() {
                    return Err("dtype mismatch");
                }
                if w_to_dtype.insert(id, dtype).is_some_and(|o| o != dtype) {
                    return Err("dtype mismatch");
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

impl<D: Eq> Interface<D> {
    pub fn sequence<T: Into<Wire<D>>, I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<Self, &'static str> {
        Self::try_from_iter(iter.into_iter().map(|w| [w.into()]))
    }

    pub fn unique<T: Into<Wire<D>>, I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<Self, &'static str> {
        let interface = Self::collect(iter.into_iter().map(|w| [w.into()]));

        let mut ids: HashSet<usize> = HashSet::new();
        for id in interface.wires().map(Wire::id) {
            if !ids.insert(id) {
                return Err("duplicate id");
            }
        }

        Ok(interface)
    }

    pub fn from_wires_unchecked<T: Into<Wire<D>>, I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_iter_unchecked(iter.into_iter().map(|w| [w.into()]))
    }
}

impl<D: fmt::Display, const N: usize> fmt::Display for Interface<D, N> {
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

impl<D: fmt::Display> fmt::Display for Wire<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} : {}", self.id, self.dtype)?;
        Ok(())
    }
}
