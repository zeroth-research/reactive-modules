/*!
# Stochastic Petri nets

Defines the minimalistic theory [`SPN`] for modelling stochastic Petri nets:
places holding tokens, transitions moving tokens around, and Poisson clocks
deciding when a transition fires.

Unlike [`crate::lia`], the theory this one is derived from, the sorts are not
matrix sorts. A [`Sort`] value is one of three scalars, plus the trivial
tangent:

- `Nat` — a token count (the marking of a single place),
- `Bool` — a truth value (e.g. whether a transition is enabled),
- `Clock { rank }` — the time left until a Poisson clock expires; `rank` is the
  differential grade: 0 is a value, 1 a first derivative, and so on — the sort
  former [`Tangent`] raises it,
- `Zero` — the trivial tangent of the constant sorts, the inhabited singleton
  whose only writer is the `zero` generator.

The clock is the only sort that can move: a marking changes by firing a
transition, not by flowing, so `Nat` and `Bool` are constant sorts and their
tangent is `Zero`. The clock tangent is a sort of its own, but it too is
inhabited by zero alone — the theory has no generator for a non-zero rate of
change.

The operations in [`SPN`] are:

- [`SPN::Nat`], [`SPN::Bool`], [`SPN::Clock`] — literals; each reads nothing and
  writes a single wire of the corresponding value sort.
- [`SPN::And`], [`SPN::Or`], [`SPN::Not`] — boolean operations on `Bool`.
- [`SPN::IsZero`], [`SPN::ClkIsZero`] — tests for an empty place (`Nat`) and for
  an expired clock (`Clock`); both produce a `Bool`.
- [`SPN::Inc`], [`SPN::Dec`] — produce and consume a token: `Nat -> Nat`.
- [`SPN::Id`] — copies its single read wire to its single write wire; it is
  defined on every sort and acts rank-generically.
- [`SPN::Nondet`]`(s)` — nondeterministic choice of a value of the value sort
  `s`; reads nothing and writes a single wire of that sort.
- [`SPN::Pos`]`(λ)` — arms a fresh Poisson clock with rate `λ`; reads nothing and
  writes a single `Clock` value wire.
- [`SPN::ClkZerograd`] — the zero flow of a clock: reads nothing and writes a
  single clock *tangent* wire (rank at least 1), saying that the clock does not
  run.
- [`SPN::Zero`] — the unique inhabitant of `Zero`, the trivial tangent of the
  constant sorts.

There are no ordering comparisons and no arithmetic beyond `Inc`/`Dec`: the
guards of a Petri net only ask whether a place is empty or whether a clock has
expired.

`SPN` implements [`Signature`], [`Sequential`], [`Combinatorial`] and
[`Differential`]; [`Signature::check`] validates the sorts of the read/write
wires against the selected operation. [`Differential::zero`] resolves to
[`SPN::ClkZerograd`] on a clock tangent and to [`SPN::Zero`] on the trivial
one; those two are the only generators that write a tangent wire.

## Examples

```
use theory::{Signature, Tangent};
use theory::spn::{SPN, Sort};

let ok = Ok::<_, String>;
let clk = Sort::clock();

// Emptiness of a place: Nat -> Bool. Clocks have their own test.
assert!(SPN::IsZero().check([Sort::Nat()].map(ok), [Sort::Bool()].map(ok)).is_ok());
assert!(SPN::IsZero().check([clk].map(ok), [Sort::Bool()].map(ok)).is_err());
assert!(SPN::ClkIsZero().check([clk].map(ok), [Sort::Bool()].map(ok)).is_ok());

// Firing a transition consumes a token: Nat -> Nat.
assert!(SPN::Dec().check([Sort::Nat()].map(ok), [Sort::Nat()].map(ok)).is_ok());

// A fresh Poisson clock with rate 2.5 reads nothing and writes a clock value.
assert!(SPN::Pos(2.5).check([].map(ok), [clk].map(ok)).is_ok());
assert!(SPN::Pos(2.5).check([].map(ok), [Sort::Nat()].map(ok)).is_err());

// Tangent wires are written by the zero generators alone.
assert!(SPN::ClkZerograd().check([].map(ok), [clk.T()].map(ok)).is_ok());
assert!(SPN::Clock(1.0).check([].map(ok), [clk.T()].map(ok)).is_err());
assert!(SPN::Zero().check([].map(ok), [Sort::Nat().T()].map(ok)).is_ok());
```
*/

use crate::*;
#[cfg(feature = "pyo3")]
use pyo3::pyclass;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyclass(frozen, eq, str))]
pub enum Sort {
    /// The time left until a Poisson clock expires; `rank` is the
    /// differential grade: 0 = value, 1 = first derivative, ...
    Clock { rank: u8 },
    // the remaining variants are empty tuple variants, not unit ones: a
    // `pyclass` complex enum (one with a field-carrying variant) takes no unit
    // variants
    /// A token count (the marking of a place): a constant sort — a marking
    /// changes by firing a transition, not by flowing.
    Nat(),
    /// A truth value: a constant sort.
    Bool(),
    /// The trivial tangent of the constant sorts: a singleton, inhabited by
    /// exactly the zero value. Terminal, not empty.
    Zero(),
}

impl Sort {
    /// A clock value sort (rank 0).
    pub fn clock() -> Self {
        Sort::Clock { rank: 0 }
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, Sort::Bool())
    }

    pub fn is_nat(&self) -> bool {
        matches!(self, Sort::Nat())
    }

    /// True of a clock of any grade, value or tangent.
    pub fn is_clock(&self) -> bool {
        matches!(self, Sort::Clock { .. })
    }

    /// The differential grade, for the sorts that carry one.
    pub fn rank(&self) -> Option<u8> {
        match self {
            Sort::Clock { rank } => Some(*rank),
            Sort::Nat() | Sort::Bool() | Sort::Zero() => None,
        }
    }
}

/// The tangent former: a clock grades up (`rank + 1`); the constant sorts
/// `Nat` and `Bool` collapse to the trivial tangent `Zero`, which is a fixed
/// point.
impl Tangent for Sort {
    #[allow(non_snake_case)]
    fn T(&self) -> Self {
        match *self {
            Sort::Clock { rank } => Sort::Clock { rank: rank + 1 },
            Sort::Nat() | Sort::Bool() | Sort::Zero() => Sort::Zero(),
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Clock { rank: 0 } => write!(f, "Clock"),
            Sort::Clock { rank } => write!(f, "T{rank} Clock"),
            Sort::Nat() => write!(f, "Nat"),
            Sort::Bool() => write!(f, "Bool"),
            Sort::Zero() => write!(f, "Zero"),
        }
    }
}

/// Theory of stochastic Petri nets over the scalar sorts [`Sort::Nat`],
/// [`Sort::Bool`] and [`Sort::Clock`].
#[derive(Clone, Debug, strum::Display)]
#[cfg_attr(feature = "pyo3", pyclass(frozen))]
pub enum SPN {
    // constants
    /// A token-count literal
    #[strum(to_string = "({0} : nat)")]
    Nat(u64),
    /// A truth-value literal
    #[strum(to_string = "({0} : bool)")]
    Bool(bool),
    /// A clock literal: the time left until the clock expires
    #[strum(to_string = "({0} : clk)")]
    Clock(f64),
    // boolean operations
    And(),
    Or(),
    Not(),
    // tests (the only guards the theory provides)
    /// Has the clock expired? `Clock -> Bool`
    ClkIsZero(),
    /// Is the place empty? `Nat -> Bool`
    IsZero(),
    // token arithmetic
    /// Produce a token: `Nat -> Nat`. Overflow is left to the semantics.
    Inc(),
    /// Consume a token: `Nat -> Nat`. Guard with [`SPN::IsZero`]; underflow is
    /// left to the semantics.
    Dec(),
    /// Copy a value of any sort
    Id(),
    /// Nondeterministic choice of a value of the given value sort
    #[strum(to_string = "(* : {0})")]
    Nondet(Sort),
    /// Arm a fresh Poisson clock with the given rate
    #[strum(to_string = "Pos({0})")]
    Pos(f64),
    /// Zero flow of a clock: the clock does not run. It writes a clock
    /// tangent (rank at least 1) and is what [`Differential::zero`] resolves
    /// to on the clock fragment.
    ClkZerograd(),
    /// The unique inhabitant of the [`Sort::Zero`] sort: the only generator
    /// writing a `Zero` wire (the trivial tangent of the constant sorts).
    Zero(),
}

impl Sequential for SPN {
    fn skip(_range: &Self::Sort) -> Self {
        SPN::Id()
    }
}

impl Combinatorial for SPN {
    fn havoc(range: &Sort) -> Self {
        match range {
            // a tangent sort is inhabited by zero alone, so havoc over it is
            // that single inhabitant
            Sort::Zero() => SPN::Zero(),
            Sort::Clock { rank } if *rank > 0 => SPN::ClkZerograd(),
            _ => SPN::Nondet(*range),
        }
    }
}

impl Differential for SPN {
    fn zero(range: &Sort) -> Self {
        // `range` is the tangent sort the generator writes
        match range {
            Sort::Clock { .. } => SPN::ClkZerograd(),
            Sort::Nat() | Sort::Bool() | Sort::Zero() => SPN::Zero(),
        }
    }
}

impl Signature for SPN {
    type Sort = Sort;
    const NAME: &'static str = "SPN";

    fn check<R, W, E: fmt::Display>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<Sort, E>>,
        W: IntoIterator<Item = Result<Sort, E>>,
    {
        match self {
            SPN::Nat(_) | SPN::Bool(_) | SPN::Clock(_) => check_const(self, read, write),
            SPN::And() | SPN::Or() | SPN::Not() => check_bool(self, read, write),
            SPN::IsZero() | SPN::ClkIsZero() => check_is_zero(self, read, write),
            SPN::Inc() | SPN::Dec() => check_nat_ops(self, read, write),
            SPN::Id() => check_flow(self, read, write),
            SPN::Nondet(sort) => check_nondet(sort, read, write),
            SPN::Pos(_) => check_sample(self, read, write),
            SPN::ClkZerograd() => check_clk_zerograd(read, write),
            SPN::Zero() => check_zero(&Sort::Zero(), read, write),
        }
    }
}

/// The sort a literal writes, i.e. the sort of the value it carries.
fn const_sort(op: &SPN) -> Sort {
    match op {
        SPN::Nat(_) => Sort::Nat(),
        SPN::Bool(_) => Sort::Bool(),
        SPN::Clock(_) => Sort::clock(),
        _ => unreachable!(),
    }
}

// A literal reads nothing and writes exactly one wire, whose sort must be the
// value sort of the literal; a tangent wire takes no constant but zero.
fn check_const<R, W, E: fmt::Display>(op: &SPN, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    if read.next().is_some() {
        return Err(format!("{op:?}: a constant cannot read values"));
    }
    let (w1, None) = (next_sort(&mut write, 0)?, write.next()) else {
        return Err(format!(
            "{op:?}: must write exactly one value (writes more)"
        ));
    };
    let expected = const_sort(op);
    match w1 {
        Sort::Zero() => Err(format!(
            "{op:?}: cannot write a Zero wire; use ZERO to say that the value does not change"
        )),
        Sort::Clock { rank } if rank > 0 => Err(format!(
            "{op:?}: cannot write a constant to the clock tangent {w1}; \
             use ZERO to say that the clock does not run"
        )),
        _ if w1 != expected => Err(format!(
            "{op:?}: a {expected} constant cannot be written to a {w1} wire"
        )),
        _ => Ok(()),
    }
}

// `Not` is unary and `And`/`Or` are binary; all of their wires are Bool.
fn check_bool<R, W, E: fmt::Display>(op: &SPN, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        SPN::Not() => {
            let (r1, None) = (next_sort(&mut read, 0)?, read.next()) else {
                return Err(format!("{op:?}: must read a single value (reads more)"));
            };
            let (w1, None) = (next_sort(&mut write, 0)?, write.next()) else {
                return Err(format!("{op:?}: must write a single value (writes more)"));
            };
            if r1 != Sort::Bool() {
                return Err(format!("{op:?}: input must be Bool, got {r1}"));
            }
            if w1 != Sort::Bool() {
                return Err(format!("{op:?}: output must be Bool, got {w1}"));
            }
            Ok(())
        }
        SPN::And() | SPN::Or() => {
            let (r1, r2, None) = (
                next_sort(&mut read, 0)?,
                next_sort(&mut read, 1)?,
                read.next(),
            ) else {
                return Err(format!("{op:?}: must read exactly two values"));
            };
            let (w1, None) = (next_sort(&mut write, 0)?, write.next()) else {
                return Err(format!("{op:?}: must write exactly one value"));
            };
            if r1 != Sort::Bool() || r2 != Sort::Bool() {
                return Err(format!("{op:?}: inputs must be Bool, got {r1} and {r2}"));
            }
            if w1 != Sort::Bool() {
                return Err(format!("{op:?}: output must be Bool, got {w1}"));
            }
            Ok(())
        }
        _ => unreachable!(),
    }
}

// Both tests are unary and produce a Bool; they differ in the sort they accept:
// `IsZero` tests a place (Nat), `ClkIsZero` a clock value.
fn check_is_zero<R, W, E: fmt::Display>(op: &SPN, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let expected = match op {
        SPN::IsZero() => Sort::Nat(),
        SPN::ClkIsZero() => Sort::clock(),
        _ => unreachable!(),
    };
    let (r1, None) = (next_sort(&mut read, 0)?, read.next()) else {
        return Err(format!("{op:?}: must read exactly one value"));
    };
    let (w1, None) = (next_sort(&mut write, 0)?, write.next()) else {
        return Err(format!("{op:?}: must write exactly one value"));
    };
    if r1 != expected {
        return Err(format!("{op:?}: input must be {expected}, got {r1}"));
    }
    if w1 != Sort::Bool() {
        return Err(format!("{op:?}: output must be Bool, got {w1}"));
    }
    Ok(())
}

// `Inc` and `Dec` are unary maps on token counts.
fn check_nat_ops<R, W, E: fmt::Display>(op: &SPN, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let (r1, None) = (next_sort(&mut read, 0)?, read.next()) else {
        return Err(format!("{op:?}: must read exactly one value"));
    };
    let (w1, None) = (next_sort(&mut write, 0)?, write.next()) else {
        return Err(format!("{op:?}: must write exactly one value"));
    };
    if r1 != Sort::Nat() {
        return Err(format!("{op:?}: input must be Nat, got {r1}"));
    }
    if w1 != Sort::Nat() {
        return Err(format!("{op:?}: output must be Nat, got {w1}"));
    }
    Ok(())
}

// `Id` copies a single value of an arbitrary sort. It applies rank-generically,
// but never across ranks: the sorts of the two wires must agree, tangent grade
// included.
fn check_flow<R, W, E: fmt::Display>(op: &SPN, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let (r1, None) = (next_sort(&mut read, 0)?, read.next()) else {
        return Err(format!("{op:?}: must read exactly one value"));
    };
    let (w1, None) = (next_sort(&mut write, 0)?, write.next()) else {
        return Err(format!("{op:?}: must write exactly one value"));
    };
    if r1 != w1 {
        return Err(format!(
            "{op:?}: input and output must have the same sort, got {r1} and {w1}"
        ));
    }
    Ok(())
}

// Nondeterministic choice is defined on the value sorts only: a tangent sort
// is inhabited by zero alone, so there is nothing to choose there.
fn check_nondet<R, W, E: fmt::Display>(sort: &Sort, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    match sort {
        Sort::Zero() | Sort::Clock { rank: 1.. } => Err(format!(
            "Nondet: the tangent sort {sort} is inhabited by zero alone; use ZERO"
        )),
        Sort::Clock { .. } | Sort::Nat() | Sort::Bool() => check_havoc(sort, read, write),
    }
}

// Sampling a clock reads nothing and writes exactly one Clock value wire; the
// rate is part of the operation.
fn check_sample<R, W, E: fmt::Display>(op: &SPN, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    if read.next().is_some() {
        return Err(format!("{op:?}: cannot read values"));
    }
    let (w1, None) = (next_sort(&mut write, 0)?, write.next()) else {
        return Err(format!("{op:?}: must write exactly one value"));
    };
    if w1 != Sort::clock() {
        return Err(format!("{op:?}: output must be Clock, got {w1}"));
    }
    Ok(())
}

// ZERO on the clock fragment: writes exactly one clock *tangent* wire (rank at
// least 1), and reads nothing.
fn check_clk_zerograd<R, W, E: fmt::Display>(read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    if read.into_iter().next().is_some() {
        return Err("ZERO expects no read wires".to_string());
    }
    let mut write = write.into_iter();
    match write.next() {
        Some(Ok(Sort::Clock { rank })) if rank >= 1 => {}
        Some(Ok(sort)) => {
            return Err(format!("ZERO expects write of a clock tangent, got {sort}"));
        }
        Some(Err(e)) => return Err(e.to_string()),
        None => return Err("ZERO expects exactly one write wire, got none".to_string()),
    }
    if write.next().is_some() {
        return Err("ZERO expects exactly one write wire, got more".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAT: Sort = Sort::Nat();
    const BOOL: Sort = Sort::Bool();
    const CLOCK: Sort = Sort::Clock { rank: 0 };
    /// The clock tangent: the sort of a clock's rate of change.
    const DCLOCK: Sort = Sort::Clock { rank: 1 };
    const ZERO: Sort = Sort::Zero();

    fn ok(s: Sort) -> Result<Sort, String> {
        Ok(s)
    }

    #[test]
    fn sort_predicates() {
        assert!(NAT.is_nat() && !NAT.is_bool() && !NAT.is_clock());
        assert!(BOOL.is_bool() && !BOOL.is_nat() && !BOOL.is_clock());
        // a clock is a clock at every grade
        assert!(CLOCK.is_clock() && DCLOCK.is_clock());
        assert!(!ZERO.is_clock() && !ZERO.is_nat() && !ZERO.is_bool());
        assert_eq!(
            (CLOCK.rank(), DCLOCK.rank(), NAT.rank()),
            (Some(0), Some(1), None)
        );
    }

    #[test]
    fn tangent_grades_the_clock_and_collapses_the_rest() {
        assert_eq!(CLOCK.T(), DCLOCK);
        assert_eq!(DCLOCK.T(), Sort::Clock { rank: 2 });
        // the constant sorts have the trivial tangent, which is a fixed point
        assert_eq!(NAT.T(), ZERO);
        assert_eq!(BOOL.T(), ZERO);
        assert_eq!(ZERO.T(), ZERO);
    }

    #[test]
    fn sort_display() {
        assert_eq!(
            (NAT.to_string(), BOOL.to_string(), CLOCK.to_string()),
            ("Nat".to_string(), "Bool".to_string(), "Clock".to_string())
        );
        assert_eq!(
            (DCLOCK.to_string(), ZERO.to_string()),
            ("T1 Clock".to_string(), "Zero".to_string())
        );
    }

    #[test]
    fn op_display() {
        assert_eq!(SPN::Nat(3).to_string(), "(3 : nat)");
        assert_eq!(SPN::Bool(true).to_string(), "(true : bool)");
        assert_eq!(SPN::Clock(1.5).to_string(), "(1.5 : clk)");
        assert_eq!(SPN::Nondet(NAT).to_string(), "(* : Nat)");
        assert_eq!(SPN::Pos(2.5).to_string(), "Pos(2.5)");
        assert_eq!(SPN::Id().to_string(), "Id");
        assert_eq!(SPN::ClkZerograd().to_string(), "ClkZerograd");
        assert_eq!(SPN::Zero().to_string(), "Zero");
    }

    #[test]
    fn const_ok() {
        assert!(SPN::Nat(7).check([].map(ok), [NAT].map(ok)).is_ok());
        assert!(SPN::Bool(false).check([].map(ok), [BOOL].map(ok)).is_ok());
        assert!(SPN::Clock(0.5).check([].map(ok), [CLOCK].map(ok)).is_ok());
    }

    #[test]
    fn const_wrong_sort_fails() {
        assert!(SPN::Nat(7).check([].map(ok), [BOOL].map(ok)).is_err());
        assert!(SPN::Bool(true).check([].map(ok), [NAT].map(ok)).is_err());
        assert!(SPN::Clock(0.5).check([].map(ok), [NAT].map(ok)).is_err());
    }

    #[test]
    fn const_with_read_fails() {
        assert!(SPN::Nat(7).check([NAT].map(ok), [NAT].map(ok)).is_err());
    }

    #[test]
    fn const_without_write_fails() {
        assert!(SPN::Nat(7).check([].map(ok), [].map(ok)).is_err());
    }

    #[test]
    fn const_with_two_writes_fails() {
        assert!(SPN::Nat(7).check([].map(ok), [NAT, NAT].map(ok)).is_err());
    }

    #[test]
    fn const_on_a_tangent_fails() {
        // a constant writes a value, never a rate of change
        assert!(SPN::Clock(0.5).check([].map(ok), [DCLOCK].map(ok)).is_err());
        assert!(SPN::Nat(7).check([].map(ok), [ZERO].map(ok)).is_err());
    }

    #[test]
    fn not_ok() {
        assert!(SPN::Not().check([BOOL].map(ok), [BOOL].map(ok)).is_ok());
    }

    #[test]
    fn not_non_bool_fails() {
        assert!(SPN::Not().check([NAT].map(ok), [BOOL].map(ok)).is_err());
        assert!(SPN::Not().check([BOOL].map(ok), [NAT].map(ok)).is_err());
    }

    #[test]
    fn not_two_reads_fails() {
        assert!(
            SPN::Not()
                .check([BOOL, BOOL].map(ok), [BOOL].map(ok))
                .is_err()
        );
    }

    #[test]
    fn and_or_ok() {
        assert!(
            SPN::And()
                .check([BOOL, BOOL].map(ok), [BOOL].map(ok))
                .is_ok()
        );
        assert!(
            SPN::Or()
                .check([BOOL, BOOL].map(ok), [BOOL].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn and_non_bool_fails() {
        assert!(
            SPN::And()
                .check([NAT, BOOL].map(ok), [BOOL].map(ok))
                .is_err()
        );
        assert!(
            SPN::And()
                .check([BOOL, CLOCK].map(ok), [BOOL].map(ok))
                .is_err()
        );
        assert!(
            SPN::And()
                .check([BOOL, BOOL].map(ok), [NAT].map(ok))
                .is_err()
        );
    }

    #[test]
    fn and_one_read_fails() {
        assert!(SPN::And().check([BOOL].map(ok), [BOOL].map(ok)).is_err());
    }

    #[test]
    fn is_zero_ok() {
        assert!(SPN::IsZero().check([NAT].map(ok), [BOOL].map(ok)).is_ok());
    }

    #[test]
    fn is_zero_wrong_sorts_fail() {
        // clocks have their own test
        assert!(
            SPN::IsZero()
                .check([CLOCK].map(ok), [BOOL].map(ok))
                .is_err()
        );
        assert!(SPN::IsZero().check([NAT].map(ok), [NAT].map(ok)).is_err());
    }

    #[test]
    fn clk_is_zero_ok() {
        assert!(
            SPN::ClkIsZero()
                .check([CLOCK].map(ok), [BOOL].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn clk_is_zero_wrong_sorts_fail() {
        assert!(
            SPN::ClkIsZero()
                .check([NAT].map(ok), [BOOL].map(ok))
                .is_err()
        );
        assert!(
            SPN::ClkIsZero()
                .check([CLOCK].map(ok), [CLOCK].map(ok))
                .is_err()
        );
        // the test reads a clock value, not a rate
        assert!(
            SPN::ClkIsZero()
                .check([DCLOCK].map(ok), [BOOL].map(ok))
                .is_err()
        );
    }

    #[test]
    fn inc_dec_ok() {
        assert!(SPN::Inc().check([NAT].map(ok), [NAT].map(ok)).is_ok());
        assert!(SPN::Dec().check([NAT].map(ok), [NAT].map(ok)).is_ok());
    }

    #[test]
    fn inc_dec_non_nat_fails() {
        assert!(SPN::Inc().check([CLOCK].map(ok), [CLOCK].map(ok)).is_err());
        assert!(SPN::Dec().check([BOOL].map(ok), [NAT].map(ok)).is_err());
        assert!(SPN::Dec().check([NAT].map(ok), [BOOL].map(ok)).is_err());
    }

    #[test]
    fn inc_arity_fails() {
        assert!(SPN::Inc().check([NAT, NAT].map(ok), [NAT].map(ok)).is_err());
        assert!(SPN::Inc().check([].map(ok), [NAT].map(ok)).is_err());
    }

    #[test]
    fn id_ok_on_every_sort() {
        for s in [NAT, BOOL, CLOCK, DCLOCK, ZERO] {
            assert!(SPN::Id().check([s].map(ok), [s].map(ok)).is_ok());
        }
    }

    #[test]
    fn id_sort_mismatch_fails() {
        assert!(SPN::Id().check([NAT].map(ok), [CLOCK].map(ok)).is_err());
    }

    #[test]
    fn id_rank_mismatch_fails() {
        // `Id` applies rank-generically, but never across ranks
        assert!(SPN::Id().check([CLOCK].map(ok), [DCLOCK].map(ok)).is_err());
    }

    #[test]
    fn id_arity_fails() {
        assert!(SPN::Id().check([NAT].map(ok), [NAT, NAT].map(ok)).is_err());
    }

    #[test]
    fn skip_is_id() {
        assert!(matches!(<SPN as Sequential>::skip(&NAT), SPN::Id()));
    }

    #[test]
    fn nondet_ok_on_every_value_sort() {
        for s in [NAT, BOOL, CLOCK] {
            let op = <SPN as Combinatorial>::havoc(&s);
            assert!(matches!(op, SPN::Nondet(r) if r == s));
            assert!(op.check([].map(ok), [s].map(ok)).is_ok());
        }
    }

    #[test]
    fn nondet_wrong_sort_fails() {
        assert!(SPN::Nondet(NAT).check([].map(ok), [BOOL].map(ok)).is_err());
    }

    #[test]
    fn nondet_with_read_fails() {
        assert!(
            SPN::Nondet(NAT)
                .check([NAT].map(ok), [NAT].map(ok))
                .is_err()
        );
    }

    #[test]
    fn nondet_on_a_tangent_fails() {
        // there is nothing to choose in a sort inhabited by zero alone
        assert!(
            SPN::Nondet(DCLOCK)
                .check([].map(ok), [DCLOCK].map(ok))
                .is_err()
        );
        assert!(SPN::Nondet(ZERO).check([].map(ok), [ZERO].map(ok)).is_err());
    }

    #[test]
    fn havoc_on_a_tangent_is_zero() {
        // the tangent sorts are singletons here, so havoc over them is the
        // single inhabitant
        assert!(matches!(
            <SPN as Combinatorial>::havoc(&DCLOCK),
            SPN::ClkZerograd()
        ));
        assert!(matches!(<SPN as Combinatorial>::havoc(&ZERO), SPN::Zero()));
    }

    #[test]
    fn zero_resolves_per_fragment() {
        // `Differential::zero` is given the tangent sort it writes
        assert!(matches!(
            <SPN as Differential>::zero(&CLOCK.T()),
            SPN::ClkZerograd()
        ));
        for s in [NAT, BOOL] {
            assert!(matches!(<SPN as Differential>::zero(&s.T()), SPN::Zero()));
        }
    }

    #[test]
    fn clk_zerograd_writes_a_clock_tangent() {
        assert!(
            SPN::ClkZerograd()
                .check([].map(ok), [DCLOCK].map(ok))
                .is_ok()
        );
        assert!(
            SPN::ClkZerograd()
                .check([].map(ok), [Sort::Clock { rank: 2 }].map(ok))
                .is_ok()
        );
        // never a value, and never the trivial tangent
        assert!(
            SPN::ClkZerograd()
                .check([].map(ok), [CLOCK].map(ok))
                .is_err()
        );
        assert!(
            SPN::ClkZerograd()
                .check([].map(ok), [ZERO].map(ok))
                .is_err()
        );
    }

    #[test]
    fn zero_writes_the_trivial_tangent() {
        assert!(SPN::Zero().check([].map(ok), [ZERO].map(ok)).is_ok());
        assert!(SPN::Zero().check([].map(ok), [NAT].map(ok)).is_err());
        assert!(SPN::Zero().check([].map(ok), [DCLOCK].map(ok)).is_err());
    }

    #[test]
    fn zero_with_read_fails() {
        assert!(SPN::Zero().check([ZERO].map(ok), [ZERO].map(ok)).is_err());
        assert!(
            SPN::ClkZerograd()
                .check([DCLOCK].map(ok), [DCLOCK].map(ok))
                .is_err()
        );
    }

    #[test]
    fn zero_arity_mismatch_fails() {
        assert!(SPN::Zero().check([].map(ok), [].map(ok)).is_err());
        assert!(SPN::Zero().check([].map(ok), [ZERO, ZERO].map(ok)).is_err());
        assert!(SPN::ClkZerograd().check([].map(ok), [].map(ok)).is_err());
        assert!(
            SPN::ClkZerograd()
                .check([].map(ok), [DCLOCK, DCLOCK].map(ok))
                .is_err()
        );
    }

    #[test]
    fn pos_ok() {
        assert!(SPN::Pos(2.5).check([].map(ok), [CLOCK].map(ok)).is_ok());
    }

    #[test]
    fn pos_non_clock_write_fails() {
        assert!(SPN::Pos(2.5).check([].map(ok), [NAT].map(ok)).is_err());
        // arming a clock produces a value, not a rate
        assert!(SPN::Pos(2.5).check([].map(ok), [DCLOCK].map(ok)).is_err());
    }

    #[test]
    fn pos_with_read_fails() {
        assert!(
            SPN::Pos(2.5)
                .check([CLOCK].map(ok), [CLOCK].map(ok))
                .is_err()
        );
    }

    #[test]
    fn pos_two_writes_fails() {
        assert!(
            SPN::Pos(2.5)
                .check([].map(ok), [CLOCK, CLOCK].map(ok))
                .is_err()
        );
    }

    #[test]
    fn read_error_is_propagated() {
        let err: Result<Sort, String> = Err("broken wire".to_string());
        assert!(SPN::Inc().check([err], [NAT].map(ok)).is_err());
    }
}
