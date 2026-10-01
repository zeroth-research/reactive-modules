use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{cmp, fmt};

/// A typed signal carrying one value per round.
///
/// A wire is the unit of data flow: terms read and write wires, and blocks,
/// atoms, and modules are wired together by sharing them. Two wires are the
/// *same* wire exactly when they share their `id` — a process-unique number
/// minted at creation — so equality, hashing, and ordering all go through it,
/// and ordering doubles as creation order. The sort `S` types the value on
/// the wire; it does not take part in identity.
///
/// A wire is *just* an identity with a sort: the differential form is the
/// sort's responsibility. A derivative wire is one whose sort is in the
/// image of the tangent former [`Tangent::T`](theory::Tangent::T) —
/// [`Var::new`](crate::var::Var::new) mints it as `Wire::new(s.T())` — and
/// the theories check plain sorts, with the grade riding inside them.
#[derive(Debug, Clone, Copy)]
pub struct Wire<S> {
    id: usize,
    dtype: S,
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
pub(crate) const PREFIX: &str = "#";

impl<S> Wire<S> {
    /// The wire's process-unique identity, assigned at creation; wires are
    /// equal, hash alike, and order exactly by it.
    pub fn id(&self) -> usize {
        self.id
    }

    /// The sort of the value the wire carries.
    pub fn dtype(&self) -> &S {
        &self.dtype
    }

    /// Creates a fresh wire with a globally unique id.
    ///
    /// The id counter is not guarded against overflow: exhausting `usize` on
    /// a 64-bit target would take centuries of continuous allocation. A wrap
    /// would silently break the id-based equality, hashing, and ordering.
    ///
    /// # Parameters
    /// - `dtype`: The sort of the value the wire carries.
    ///
    /// # Returns
    /// The fresh wire.
    pub fn new(dtype: S) -> Self {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
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

pub struct Display<'a, S> {
    wire: &'a Wire<S>,
    typed: bool,
}

impl<S: fmt::Display> fmt::Display for Display<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{PREFIX}{}", self.wire.id)?;
        if self.typed {
            write!(f, " : {}", self.wire.dtype)?;
        }
        Ok(())
    }
}

impl<S: Debug> Debug for Display<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{{PREFIX}{}", self.wire.id)?;
        if self.typed {
            write!(f, " : {:?}", self.wire.dtype)?;
        }
        write!(f, "}}")
    }
}

impl<S> Wire<S> {
    /// Displays the wire with its sort: `#id : sort`.
    ///
    /// # Returns
    /// A borrowing adapter implementing [`fmt::Display`] (and [`Debug`],
    /// braced).
    pub fn typed(&self) -> Display<'_, S> {
        Display {
            wire: self,
            typed: true,
        }
    }

    /// Displays the wire by its id alone: `#id`.
    ///
    /// # Returns
    /// A borrowing adapter implementing [`fmt::Display`] (and [`Debug`],
    /// braced).
    pub fn untyped(&self) -> Display<'_, S> {
        Display {
            wire: self,
            typed: false,
        }
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
