use crate::primitives::action::Action;
use crate::primitives::variable::{VarIterExt, Variable};
use std::collections::HashSet;

pub struct Atom<V: Variable, S, I: Action<V, S>, U: Action<V, S>>
where
    for<'a> &'a S: IntoIterator<Item = &'a V>,
    for<'a> <&'a S as IntoIterator>::IntoIter: ExactSizeIterator,
{
    /// control
    pub ctr: S,
    /// read
    pub read: S,
    /// wait
    pub wait: S,
    /// init
    pub init: I,
    /// update
    pub update: U,
}

impl<V: Variable, S, I: Action<V, S>, U: Action<V, S>> Atom<V, S, I, U>
where
    for<'a> &'a S: IntoIterator<Item = &'a V>,
    for<'a> <&'a S as IntoIterator>::IntoIter: ExactSizeIterator,
{
    pub fn new(ctr: S, read: S, wait: S, init: I, update: U) -> Self {
        #[cfg(debug_assertions)]
        {
            debug_assert!(!ctr.has_duplicates() && ctr.is_latched());
            debug_assert!(!read.has_duplicates() && read.is_latched());
            debug_assert!(!wait.has_duplicates() && wait.is_latched());
            debug_assert!(wait.as_set().is_disjoint(&ctr.as_set()));
            debug_assert_eq!(wait.next_set(), init.read().into_set());
            debug_assert_eq!(ctr.next_set(), init.write().into_set());
            let mut read_union_wait_next = HashSet::<V>::new();
            read_union_wait_next.extend(read.into_set());
            read_union_wait_next.extend(wait.next_set());
            debug_assert_eq!(read_union_wait_next, update.read().into_set());
            debug_assert_eq!(ctr.next_set(), update.write().into_set());
        }
        Self {
            ctr,
            read,
            wait,
            init,
            update,
        }
    }

    /// `b.awaits(a)` if
    /// `b.wait` and `a.ctr` have a common element
    pub fn awaits(&self, other: &Atom<V, S, I, U>) -> bool {
        // TODO: this is inefficient in the context of the constructor of module
        !self.wait.as_set().is_disjoint(&other.ctr.as_set())
    }
}
