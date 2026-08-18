use crate::Wire;
use std::borrow::Borrow;
use std::collections::HashSet;
use std::fmt;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

#[derive(Debug, Clone, Copy)]
pub struct Variable<S> {
    ltc: Wire<S>,
    nxt: Wire<S>,
    der: Wire<S>,
}

impl<S: Clone> Variable<S> {
    pub fn new(dtype: S) -> Self {
        Self {
            ltc: Wire::zero(dtype.clone()),
            nxt: Wire::zero(dtype.clone()),
            der: Wire::one(dtype),
        }
    }
}

impl<S> Variable<S> {
    pub(crate) fn ltc(&self) -> &Wire<S> {
        &self.ltc
    }

    pub(crate) fn nxt(&self) -> &Wire<S> {
        &self.nxt
    }

    pub(crate) fn der(&self) -> &Wire<S> {
        &self.der
    }
}

/// Equality goes through the latched wire — the variable's bearing element,
/// consistent with its [`Deref`] view: two variables are the same variable
/// iff they share the same latched wire.
impl<S> PartialEq for Variable<S> {
    fn eq(&self, other: &Self) -> bool {
        self.ltc == other.ltc
    }
}

impl<S> Eq for Variable<S> {}

/// Hashing matches equality: only the latched wire is hashed.
impl<S> Hash for Variable<S> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ltc.hash(state);
    }
}

/// A variable casts into its latched wire: `let a: Wire<_> = *x;`, and
/// implicitly at any call boundary bounded on `Into<Wire<S>>` (e.g. the term
/// constructors).
impl<S> From<Variable<S>> for Wire<S> {
    fn from(var: Variable<S>) -> Self {
        var.ltc
    }
}

/// A variable dereferences to its latched wire: `&x` coerces to `&Wire<S>`
/// wherever one is expected, and `Wire`'s methods (`x.id()`, `x.dtype()`,
/// `x.degree()`) apply to the variable directly.
///
/// `Deref` for a non-pointer type is deliberate here: the latched wire is the
/// variable's primary view, the way `String` derefs to `str`.
impl<S> Deref for Variable<S> {
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
pub fn d<S: Clone>(var: impl Borrow<Variable<S>>) -> Wire<S> {
    var.borrow().der.clone()
}

/// The borrowing alternative to [`d`]: a reference to the variable's derived
/// (derivative) wire, `let w: &Wire<_> = d_ref(&x);`.
///
/// Reference in, reference out — no clone
pub fn d_ref<S>(var: &Variable<S>) -> &Wire<S> {
    &var.der
}

/// The variable's next wire: `let w: Wire<_> = X(x);`.
///
/// Bounded on `Borrow`, so it accepts the variable by value (`X(x)`, natural
/// for `Copy` sorts) or by reference (`X(&x)`); either way only the wire is
/// cloned and the variable stays usable.
#[allow(non_snake_case)]
pub fn X<S: Clone>(var: impl Borrow<Variable<S>>) -> Wire<S> {
    var.borrow().nxt.clone()
}

/// The borrowing alternative to [`X`]: a reference to the variable's next
/// wire, `let w: &Wire<_> = X_ref(&x);`.
///
/// Reference in, reference out — no clone
#[allow(non_snake_case)]
pub fn X_ref<S>(var: &Variable<S>) -> &Wire<S> {
    &var.nxt
}

impl<S: fmt::Display> fmt::Display for Variable<S> {
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

#[derive(Debug, Clone)]
pub struct Interface<S> {
    vars: Vec<Variable<S>>,
}

impl<S> Interface<S> {
    pub fn empty() -> Self {
        Self { vars: Vec::new() }
    }

    /// Returns true if the wire indices of self are disjoint from the indices of other, regardless of their type.
    /// This function runs in place, in O(self.len() * other.len()) time
    pub fn is_disjoint(&self, other: &Interface<S>) -> bool {
        for a in self.vars.iter() {
            if other.vars.iter().any(|b| a == b) {
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

    pub fn iter(&self) -> impl Iterator<Item = &Variable<S>> {
        self.vars.iter()
    }

    pub fn try_from_iter<I: IntoIterator<Item = Variable<S>>>(iter: I) -> Result<Self, String> {
        let vars: Vec<_> = iter.into_iter().collect();

        // variables must have no repetition
        let mut decl: HashSet<usize> = HashSet::new();
        for var in vars.iter() {
            if decl.insert(var.id()) {
                return Err(format!("Duplicate variable {:?}", var.id()));
            }
        }

        Ok(Self { vars })
    }
}

impl<S: Debug> Interface<S> {
    pub(crate) fn from_iter_unchecked<T: Into<Variable<S>>, I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Self {
        let vars: Vec<_> = iter.into_iter().map(Into::into).collect();

        #[cfg(debug_assertions)]
        {
            // variables must have no repetition
            let mut decl: HashSet<usize> = HashSet::new();
            for var in vars.iter() {
                debug_assert!(decl.insert(var.id()), "duplicate var {:?}", var);
            }
        }

        Self { vars }
    }
}

impl<S> IntoIterator for Interface<S> {
    type Item = Variable<S>;
    type IntoIter = std::vec::IntoIter<Variable<S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.vars.into_iter()
    }
}

impl<'a, S> IntoIterator for &'a Interface<S> {
    type Item = &'a Variable<S>;
    type IntoIter = std::slice::Iter<'a, Variable<S>>;

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
    fn variable_casts_to_its_wires() {
        // variables are Copy: owned bindings, ampersand-free use everywhere
        let x = Variable::new("type");

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

        let x = Variable::new(Sort::Real([1, 1]));

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
        let x = Variable::new(Sort::Real([1, 1]));
        // dx = x: the derivative is degree 1, ordinary operands must be degree 0
        assert!(Term::function(LRA::Id(), [d(x)], [x]).is_err());
    }
}
