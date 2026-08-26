use crate::Wire;
use crate::wire::PREFIX;
use std::borrow::{Borrow, Cow};
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
            ltc: Wire::scalar(dtype.clone()),
            nxt: Wire::scalar(dtype.clone()),
            der: Wire::covector(dtype),
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

/// A variable borrows as its latched wire, consistently with `Eq`, `Ord`,
/// and `Hash`, which all delegate to it: a `Var` stands for its latched wire
/// wherever a `Borrow<Wire<S>>` is accepted (e.g. keyed lookups, printing).
impl<S> Borrow<Wire<S>> for Var<S> {
    fn borrow(&self) -> &Wire<S> {
        self
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

pub struct Display<'a, S> {
    var: &'a Var<S>,
    name: Cow<'a, str>,
}

impl<S: fmt::Display> fmt::Display for Display<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} : {}", self.name, self.var.dtype())
    }
}

impl<S: Debug> Debug for Display<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} : {:?}", self.name, self.var.dtype())
    }
}

impl<'a, S> Var<S> {
    pub fn with_name(&'a self, name: Cow<'a, str>) -> Display<'a, S> {
        Display { var: self, name }
    }
}

impl<S: Debug> Debug for Var<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Var {{ ltc: {PREFIX}{:?}, nxt: {PREFIX}{:?}, der: {PREFIX}{:?}, dtype: {:?} }} ",
            self.ltc.id(),
            self.nxt.id(),
            self.der.id(),
            self.dtype()
        )
    }
}

#[derive(Clone)]
pub struct Interface<S> {
    vars: Vec<Var<S>>,
    wires: Vec<(Wire<S>, Var<S>)>,
}

impl<S> Interface<S> {
    pub fn empty() -> Self {
        Self {
            vars: Vec::new(),
            wires: Vec::new(),
        }
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

    pub fn var(&self, wire: &Wire<S>) -> Option<&Var<S>> {
        let res = self.wires.binary_search_by_key(&wire, |(w, _)| w);
        res.map(|idx| &self.wires[idx].1).ok()
    }

    pub fn nth(&self, index: usize) -> Option<&Var<S>> {
        self.vars.get(index)
    }

    pub fn contains(&self, var: &Var<S>) -> bool {
        self.vars.binary_search(var).is_ok()
    }
}

impl<S: Debug + Clone> Interface<S> {
    pub(crate) fn from_iter_unchecked<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Var<S>>,
    {
        let vars: Vec<Var<S>> = iter.into_iter().collect();
        Self::from_exact_iter_unchecked(vars)
    }

    pub(crate) fn from_exact_iter_unchecked<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Var<S>>,
        I::IntoIter: ExactSizeIterator,
    {
        let iter = iter.into_iter();
        let mut vars = Vec::with_capacity(iter.len());
        vars.extend(iter);

        let mut wires: Vec<(Wire<S>, Var<S>)> = Vec::with_capacity(3 * vars.len());
        wires.extend(vars.iter().map(|v: &Var<S>| (v.ltc.clone(), v.clone())));
        wires.extend(vars.iter().map(|v: &Var<S>| (v.nxt.clone(), v.clone())));
        wires.extend(vars.iter().map(|v: &Var<S>| (v.der.clone(), v.clone())));
        wires.sort_unstable();

        // variables must be unique and sorted
        debug_assert!(
            vars.is_sorted_by(|a, b| a < b),
            "duplicate or unsorted {:?}",
            vars
        );

        Self { vars, wires }
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

impl<S: Debug> Debug for Interface<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Interface {{ vars: {:?} }}", self.vars)
    }
}
