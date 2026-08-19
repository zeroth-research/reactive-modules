use crate::Wire;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

#[derive(Clone, Copy)]
pub struct Var<S> {
    ltc: Wire<S>,
    nxt: Wire<S>,
    der: Wire<S>,
}

impl<S: Clone> Var<S> {
    pub fn new(dtype: S) -> Self {
        Self {
            ltc: Wire::zero(dtype.clone()),
            nxt: Wire::zero(dtype.clone()),
            der: Wire::one(dtype),
        }
    }
}

impl<S> Var<S> {
    pub(crate) fn ltc(&self) -> &Wire<S> {
        &self.ltc
    }

    pub(crate) fn nxt(&self) -> &Wire<S> {
        &self.nxt
    }

    pub(crate) fn der(&self) -> &Wire<S> {
        &self.der
    }

    pub(crate) fn wires(&self) -> [&Wire<S>; 3] {
        [&self.ltc, &self.nxt, &self.der]
    }
}

/// Equality goes through the latched wire — the variable's bearing element,
/// consistent with its [`Deref`] view: two variables are the same variable
/// iff they share the same latched wire.
impl<S> PartialEq for Var<S> {
    fn eq(&self, other: &Self) -> bool {
        self.ltc == other.ltc
    }
}

impl<S> Eq for Var<S> {}

/// Hashing matches equality: only the latched wire is hashed.
impl<S> Hash for Var<S> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ltc.hash(state);
    }
}

/// Variables are totally ordered by their latched wire — the
/// bearing element of equality — which globally orders them by creation.
impl<S> Ord for Var<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.ltc.cmp(&other.ltc)
    }
}

impl<S> PartialOrd for Var<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A variable casts into its latched wire: `let a: Wire<_> = *x;`, and
/// implicitly at any call boundary bounded on `Into<Wire<S>>` (e.g. the term
/// constructors).
impl<S> From<Var<S>> for Wire<S> {
    fn from(var: Var<S>) -> Self {
        var.ltc
    }
}

/// A variable dereferences to its latched wire: `&x` coerces to `&Wire<S>`
/// wherever one is expected, and `Wire`'s methods (`x.id()`, `x.dtype()`,
/// `x.degree()`) apply to the variable directly.
///
/// `Deref` for a non-pointer type is deliberate here: the latched wire is the
/// variable's primary view, the way `String` derefs to `str`.
impl<S> Deref for Var<S> {
    type Target = Wire<S>;

    fn deref(&self) -> &Self::Target {
        &self.ltc
    }
}

/// The variable's derived (derivative) wire: `let w: Wire<_> = d(x);`.
///
/// Bounded on `Borrow`, so it accepts the variable by value (`d(x)`, natural
/// for `Copy` sorts) or by reference (`d(&x)`); either way only the wire is
/// cloned and the variable stays usable.
pub fn d<S: Clone>(var: impl Borrow<Var<S>>) -> Wire<S> {
    var.borrow().der.clone()
}

/// The borrowing alternative to [`d`]: a reference to the variable's derived
/// (derivative) wire, `let w: &Wire<_> = d_ref(&x);`.
///
/// Reference in, reference out — no clone
pub fn d_ref<S>(var: &Var<S>) -> &Wire<S> {
    &var.der
}

/// The variable's next wire: `let w: Wire<_> = X(x);`.
///
/// Bounded on `Borrow`, so it accepts the variable by value (`X(x)`, natural
/// for `Copy` sorts) or by reference (`X(&x)`); either way only the wire is
/// cloned and the variable stays usable.
#[allow(non_snake_case)]
pub fn X<S: Clone>(var: impl Borrow<Var<S>>) -> Wire<S> {
    var.borrow().nxt.clone()
}

/// The borrowing alternative to [`X`]: a reference to the variable's next
/// wire, `let w: &Wire<_> = X_ref(&x);`.
///
/// Reference in, reference out — no clone
#[allow(non_snake_case)]
pub fn X_ref<S>(var: &Var<S>) -> &Wire<S> {
    &var.nxt
}

impl<S: fmt::Display> fmt::Display for Var<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "v({},{},{}) : {}",
            self.ltc.id(),
            self.nxt.id(),
            self.der.id(),
            self.dtype()
        )
    }
}

impl<S: Debug> Debug for Var<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Var {{ ltc: {:?} nxt: {:?}, der: {:?} }} : {:?}",
            self.ltc.id(),
            self.nxt.id(),
            self.der.id(),
            self.dtype()
        )
    }
}

#[derive(Debug, Clone)]
pub struct Interface<S> {
    vars: Vec<Var<S>>,
}

impl<S> Interface<S> {
    pub fn empty() -> Self {
        Self { vars: Vec::new() }
    }

    /// Returns true if the variables of self are disjoint from the variables of other
    /// This function runs in place, in O(self.len() * log(other.len())) time
    pub fn is_disjoint(&self, other: &Interface<S>) -> bool {
        for a in self.vars.iter() {
            if other.vars.binary_search(a).is_ok() {
                return false;
            }
        }
        true
    }

    pub fn len(&self) -> usize {
        self.vars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Var<S>> {
        self.vars.iter()
    }
}

impl<S: Debug> Interface<S> {
    pub(crate) fn from_exact_iter_unchecked<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Var<S>>,
        I::IntoIter: ExactSizeIterator,
    {
        let iter = iter.into_iter();
        let mut vars = Vec::with_capacity(iter.len());
        vars.extend(iter.map(Into::into));

        // variables must be unique and sorted
        debug_assert!(
            vars.is_sorted_by(|a, b| a < b),
            "duplicate or unsorted {:?}",
            vars
        );

        Self { vars }
    }
}

impl<S> IntoIterator for Interface<S> {
    type Item = Var<S>;
    type IntoIter = std::vec::IntoIter<Var<S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.vars.into_iter()
    }
}

impl<'a, S> IntoIterator for &'a Interface<S> {
    type Item = &'a Var<S>;
    type IntoIter = std::slice::Iter<'a, Var<S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.vars.iter()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Term;
    use theory::lra::{LRA, Sort};

    #[test]
    fn variables_are_globally_ordered_by_creation() {
        let x = Var::new("t");
        let y = Var::new("t");
        let z = Var::new("t");

        // the latched wire's id increases with creation
        assert!(x < y && y < z);
        assert!(x <= x && x == x);

        // sorting recovers the creation order
        let mut vars = [z, x, y];
        vars.sort();
        assert_eq!(vars, [x, y, z]);
    }

    #[test]
    fn variable_casts_to_its_wires() {
        // variables are Copy: owned bindings, ampersand-free use everywhere
        let x = Var::new("type");

        let a: Wire<_> = *x; // the latched wire
        let b: Wire<_> = d(x); //     its derivative
        let c: Wire<_> = X(x); //   the next wire

        // d/nxt copy, never clone and never consume: the variable stays
        // usable and the views are stable
        assert_eq!(d(x), b);
        assert_eq!(X(x), c);

        // three distinct wires; the latched and next are values (degree 0),
        // the derivative is one degree up
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
        assert_eq!(a.degree(), 0);
        assert_eq!(b.degree(), 1);

        // the variable dereferences to its latched wire: `Wire`'s methods
        // apply to it directly, and `&x` coerces to `&Wire<_>`
        assert_eq!(x.degree(), 0);
        assert_eq!(x.id(), a.id());
        let borrowed: &Wire<_> = &x;
        assert_eq!(*borrowed, a);

        // X and d take any borrow of the variable: by reference as well
        assert_eq!(d(&x), b);
        assert_eq!(X(&x), c);

        // the _ref alternatives borrow all the way through: reference in,
        // reference out, no clone
        let da: &Wire<_> = &x;
        let db: &Wire<_> = d_ref(&x);
        let cb: &Wire<_> = X_ref(&x);
        assert_eq!(da, &a);
        assert_eq!(db, &b);
        assert_eq!(cb, &c);
    }

    #[test]
    fn variables_pass_into_term_constructors() {
        use crate::Term;
        use theory::lra::{LRA, Sort};

        let x = Var::new(Sort::Real([1, 1]));

        // a variable stands for its latched wire in a term position
        let t = Term::constant(LRA::Havoc(), [x]).unwrap();
        assert_eq!(t.write().iter().next().map(Wire::id), Some(x.id()));

        // and its views pass as well: x' = x
        let t = Term::function(LRA::Id(), [X(x)], [x]).unwrap();
        assert_eq!(t.write().iter().next().map(Wire::id), Some(X(x).id()));
        assert_eq!(t.read().iter().next().map(Wire::id), Some(x.id()));
    }

    #[test]
    fn variable_fail_into_term_constructors() {
        let x = Var::new(Sort::Real([1, 1]));
        // dx = x: the derivative is degree 1, ordinary operands must be degree 0
        assert!(Term::function(LRA::Id(), [d(x)], [x]).is_err());
    }
}
