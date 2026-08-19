use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{cmp, fmt};

#[derive(Debug, Clone, Copy)]
pub struct Wire<S> {
    id: usize,
    dtype: S,
    degree: u8,
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

impl<S> Wire<S> {
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn dtype(&self) -> &S {
        &self.dtype
    }

    pub fn degree(&self) -> u8 {
        self.degree
    }

    pub fn new(dtype: S, degree: u8) -> Self {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        if id == usize::MAX {
            panic!("wire id overflow");
        }
        Self { id, dtype, degree }
    }

    pub fn zero(dtype: S) -> Self {
        Self::new(dtype, 0)
    }

    pub fn one(dtype: S) -> Self {
        Self::new(dtype, 1)
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

impl<S: fmt::Display> fmt::Display for Wire<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} : {}", self.id, self.dtype)?;
        Ok(())
    }
}

impl<S: Clone> From<Wire<S>> for (S, u8) {
    fn from(wire: Wire<S>) -> Self {
        (wire.dtype, wire.degree)
    }
}

impl<S> Ord for Wire<S> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl<S> PartialOrd for Wire<S> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}
