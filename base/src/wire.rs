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
/// The `degree` says which differential form the wire carries: a `0`-form
/// is a value, a `1`-form is a derivative (the rate of change of a value
/// with respect to time, as written by delay blocks). The degree is checked
/// by the theories — an op reads and writes wires of the forms it expects —
/// but, like the sort, it does not take part in identity.
#[derive(Debug, Clone, Copy)]
pub struct Wire<S> {
    id: usize,
    dtype: S,
    degree: u8, // indicates the differential form
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

    /// The differential form of the wire: a `0`-form carries a value, a
    /// `1`-form carries a derivative.
    pub fn degree(&self) -> u8 {
        self.degree
    }

    /// Creates a fresh wire with a globally unique id.
    ///
    /// The id counter is not guarded against overflow: exhausting `usize` on
    /// a 64-bit target would take centuries of continuous allocation. A wrap
    /// would silently break the id-based equality, hashing, and ordering.
    ///
    /// # Parameters
    /// - `dtype`: The sort of the value the wire carries.
    /// - `degree`: The differential form of the wire: `0`-form for a value,
    ///   `1`-form for a derivative.
    ///
    /// # Returns
    /// The fresh wire.
    ///
    /// # See Also
    /// - [`Wire::scalar`] and [`Wire::covector`], fixing the form by name.
    pub fn new(dtype: S, degree: u8) -> Self {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        Self { id, dtype, degree }
    }

    /// Creates a fresh `0`-form wire: it carries a value.
    ///
    /// # Parameters
    /// - `dtype`: The sort of the value the wire carries.
    ///
    /// # Returns
    /// The fresh wire.
    pub fn scalar(dtype: S) -> Self {
        Self::new(dtype, 0)
    }

    /// Creates a fresh `1`-form wire: it carries a derivative.
    ///
    /// # Parameters
    /// - `dtype`: The sort of the value whose derivative the wire carries.
    ///
    /// # Returns
    /// The fresh wire.
    pub fn covector(dtype: S) -> Self {
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
