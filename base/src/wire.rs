use std::collections::HashMap;
use std::fmt;
use std::iter::Map;

/// A wiring represents a view over a sequence of indices, each of which is associated
/// with a type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wire<D> {
    pub(crate) vec: Vec<(usize, D)>,
}

impl<D> Wire<D> {
    pub fn none() -> Wire<D> {
        Self { vec: vec![] }
    }

    pub fn one(offset: usize, dtype: D) -> Wire<D> {
        Self {
            vec: vec![(offset, dtype)],
        }
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }
}

type Iter<'a, D> = Map<std::slice::Iter<'a, (usize, D)>, fn(&'a (usize, D)) -> (usize, &'a D)>;

impl<'a, D> Wire<D> {
    pub fn iter(&'a self) -> Iter<'a, D> {
        self.vec.iter().map(|(w, t)| (*w, t))
    }
}

impl<D> IntoIterator for Wire<D> {
    type Item = (usize, D);
    type IntoIter = std::vec::IntoIter<(usize, D)>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.into_iter()
    }
}

impl<'a, D> IntoIterator for &'a Wire<D> {
    type Item = (usize, &'a D);
    type IntoIter = Iter<'a, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[macro_export]
macro_rules! wire {
    // case 1: wires are passed are index-dtype pairs
    ( $( ($x:expr, $y:expr) ),* $(,)? ) => {{
        let tmp = [ $( ($x, $y) ),* ];
        tmp.into_iter().collect::<Wire<_>>()
    }};

    // case 2: wires are passed as references, and are automatically cloned
    ( $( &$x:expr ),* $(,)? ) => {{
        let tmp = [ $( &$x ),* ];
        tmp.into_iter().flatten().map(|(w, d)| (w, d.clone())).collect::<Wire<_>>()
    }};

    // case 3: wires are passed as single elements
    ( $( $x:expr ),* $(,)? ) => {{
        let tmp = [ $( $x ),* ];
        tmp.into_iter().flatten().collect::<Wire<_>>()
    }};
}

/// from_iter can panic. Use at your own risk
impl<D: Eq> FromIterator<(usize, D)> for Wire<D> {
    fn from_iter<I: IntoIterator<Item = (usize, D)>>(iter: I) -> Self {
        Self::try_from_iter(iter).unwrap()
    }
}

impl<D: Eq> Wire<D> {
    pub(crate) fn new_unchecked(vec: Vec<(usize, D)>) -> Wire<D> {
        #[cfg(debug_assertions)]
        {
            // wires may be duplicate but their type should be consistent
            let mut w_to_dtype: HashMap<usize, &D> = HashMap::new();
            for (a, b) in vec.iter() {
                debug_assert!(w_to_dtype.insert(*a, b).is_none_or(|c| c == b));
            }
        }
        Self { vec }
    }

    pub fn try_from_iter<I>(iter: I) -> Result<Self, &'static str>
    where
        I: IntoIterator<Item = (usize, D)> + Sized,
    {
        let iter = iter.into_iter();
        let mut vec: Vec<(usize, D)> = Vec::with_capacity(iter.size_hint().0);

        let mut w_to_dtype: HashMap<usize, usize> = HashMap::new();
        for (a, b) in iter {
            if w_to_dtype
                .insert(a, vec.len())
                .is_some_and(|i| b != vec[i].1)
            {
                return Err("Inconsistent wire dtype");
            }
            vec.push((a, b));
        }

        Ok(Self::new_unchecked(vec))
    }

    /// Returns true if the wire indices of self are also indices of other, regardless of their type.
    /// This function runs in place, in O(self.len() * other.len()) time
    pub fn is_subset(&self, other: &Wire<D>) -> bool {
        for (a, _) in self.vec.iter() {
            if other.vec.iter().all(|(b, _)| a != b) {
                return false;
            }
        }
        true
    }
}

impl<D: Eq + Clone> Wire<D> {
    pub fn many(offset: usize, dtype: D, n: usize) -> Self {
        let mut vec = Vec::with_capacity(n);
        vec.extend((offset..offset + n - 1).map(|i| (i, dtype.clone())));
        vec.push((offset + n - 1, dtype));
        Self::new_unchecked(vec)
    }
}

impl<D: fmt::Display> fmt::Display for Wire<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (i, d) in self.iter() {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "x{} : {}", i, d)?;
            first = false;
        }
        Ok(())
    }
}
