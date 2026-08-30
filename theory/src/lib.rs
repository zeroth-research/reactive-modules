pub mod any;
pub mod bv;
pub mod lia;
pub mod lra;
pub mod tensor;

use std::fmt;
use std::fmt::Debug;
pub use tensor::PyTensor;

/// Sorts closed under the tangent former: `T(s)` is the sort of rates of
/// change of values of sort `s`.
///
/// This trait captures only the *object part* of a tangent structure — the
/// action of `T` on sorts — which is all the reactive layer needs to mint
/// derivative wires. The structure maps of a tangent category (the zero
/// section, bundle addition, projection, lift, flip) are signature
/// *generators* (`zero` is the zero section; addition is the base theory's
/// `Add` at tangent sorts), and their axioms (Rosický; Cockett–Cruttwell)
/// are equations between operations — they belong to the upcoming
/// `Theory: Signature` subtrait, not here. Implementing `Tangent` does not
/// certify tangent-category structure.
///
/// Discrete sorts have a *trivial* tangent, not a missing one: `T` maps
/// them to an inhabited singleton sort (`Zero`), whose only writer is the
/// `zero` generator.
pub trait Tangent {
    /// The tangent sort: `T` as in the tangent bundle `TM`.
    #[allow(non_snake_case)]
    fn T(&self) -> Self;
}

/// Identity tangent for opaque string sorts (test scaffolding: theories
/// whose checks are vacuous do not distinguish values from rates).
impl Tangent for &str {
    fn T(&self) -> Self {
        self
    }
}

pub trait Signature {
    type Sort;

    const NAME: &'static str;

    fn check<R, W, E: fmt::Display>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<Self::Sort, E>>,
        W: IntoIterator<Item = Result<Self::Sort, E>>;
}

pub trait Combinatorial: Signature {
    // Returns the unary itype that chooses an element within the range
    fn havoc(range: &Self::Sort) -> Self;
}

pub trait Sequential: Signature {
    // Returns the unary itype that copies a value within the range
    fn skip(range: &Self::Sort) -> Self;
}

pub trait Differential: Signature {
    // Returns the unary itype that indicates zero rate of change; `range`
    // is the *tangent* sort the generator writes (the derivative wire's sort)
    fn zero(range: &Self::Sort) -> Self;
}

// Helpers for type-checking procedures

// SKIP is unary: it copies exactly one read wire to one write wire of the
// same sort, matching the arity of the base theories' `Id`.
// This is a helper routine; concrete theories are free to implement their own
fn check_skip<R, W, S, E>(range: &S, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<S, E>>,
    W: IntoIterator<Item = Result<S, E>>,
    S: Eq + Debug,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match (read.next(), write.next()) {
        (Some(Ok(r)), Some(Ok(w))) if r != w || &w != range => {
            return Err(format!(
                "SKIP expects exactly one read and one write of sort {:?}",
                range
            ));
        }
        (Some(_), Some(_)) => {}
        _ => {
            return Err("SKIP expects exactly one read and one write".to_string());
        }
    }
    if read.next().is_some() || write.next().is_some() {
        return Err("SKIP expects exactly one read and one write".to_string());
    }
    Ok(())
}

// HAVOC is unary: it writes exactly one wire and reads none.
// This is a helper routine; concrete theories are free to implement their own
pub(crate) fn check_havoc<S, R, W, E>(range: &S, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<S, E>>,
    W: IntoIterator<Item = Result<S, E>>,
    S: Eq + Debug,
    E: fmt::Display,
{
    if read.into_iter().next().is_some() {
        return Err("HAVOC expects no read wires".to_string());
    }
    let mut write = write.into_iter();
    match write.next() {
        Some(Ok(sort)) if &sort != range => {
            return Err(format!("HAVOC expects write of dtype {:?}", range));
        }
        Some(Err(e)) => return Err(e.to_string()),
        None => return Err("HAVOC expects exactly one write wire, got none".to_string()),
        _ => {}
    }
    if write.next().is_some() {
        return Err("HAVOC expects exactly one write wire, got more".to_string());
    }
    Ok(())
}

// ZERO is unary: it writes exactly one wire and reads none. Its range is a
// tangent sort — the derivative wire it silences.
// This is a helper routine; concrete theories are free to implement their own
pub(crate) fn check_zero<S, R, W, E>(range: &S, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<S, E>>,
    W: IntoIterator<Item = Result<S, E>>,
    S: Eq + Debug,
    E: fmt::Display,
{
    if read.into_iter().next().is_some() {
        return Err("ZERO expects no read wires".to_string());
    }
    let mut write = write.into_iter();
    match write.next() {
        Some(Ok(sort)) if &sort != range => {
            return Err(format!("ZERO expects write of dtype {:?}", range));
        }
        Some(Err(e)) => return Err(e.to_string()),
        None => return Err("ZERO expects exactly one write wire, got none".to_string()),
        _ => {}
    }
    if write.next().is_some() {
        return Err("ZERO expects exactly one write wire, got more".to_string());
    }
    Ok(())
}

fn next_sort<R, S, E: fmt::Display>(iter: &mut R, pos: usize) -> Result<S, String>
where
    R: Iterator<Item = Result<S, E>>,
{
    if let Some(item) = iter.next() {
        item.map_err(|e| e.to_string())
    } else {
        Err(format!("Arg {pos} expected, but got none"))
    }
}
