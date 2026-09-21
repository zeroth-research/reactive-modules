/*!
# Stochastic Petri nets

Defines the minimalistic theory [`SPN`] for modelling stochastic Petri nets:
places holding tokens, transitions moving tokens around, and Poisson clocks
deciding when a transition fires.

Unlike [`crate::lia`], the theory this one is derived from, the sorts are not
matrix sorts. A [`Sort`] value is one of three scalars:

- `Nat` — a token count (the marking of a single place),
- `Bool` — a truth value (e.g. whether a transition is enabled),
- `Clock` — the time left until a Poisson clock expires.

The operations in [`SPN`] are:

- [`SPN::Nat`], [`SPN::Bool`], [`SPN::Clock`] — literals; each reads nothing and
  writes a single wire of the corresponding sort.
- [`SPN::And`], [`SPN::Or`], [`SPN::Not`] — boolean operations on `Bool`.
- [`SPN::IsZero`], [`SPN::ClkIsZero`] — tests for an empty place (`Nat`) and for
  an expired clock (`Clock`); both produce a `Bool`.
- [`SPN::Inc`], [`SPN::Dec`] — produce and consume a token: `Nat -> Nat`.
- [`SPN::Id`] — copies its single read wire to its single write wire; it is
  defined on every sort.
- [`SPN::Nondet`]`(s)` — nondeterministic choice of a value of sort `s`; reads
  nothing and writes a single wire of that sort.
- [`SPN::Pos`]`(λ)` — arms a fresh Poisson clock with rate `λ`; reads nothing and
  writes a single `Clock` wire.
- [`SPN::Zero`]`(s)` — the zero flow of sort `s`: reads nothing and writes a
  single wire of that sort, saying that its value does not change.

There are no ordering comparisons and no arithmetic beyond `Inc`/`Dec`: the
guards of a Petri net only ask whether a place is empty or whether a clock has
expired.

`SPN` implements [`Theory`], [`Sequential`], [`Combinatorial`] and
[`Differential`]; [`Theory::check`] validates the sorts of the read/write wires
against the selected operation. Every operation but [`SPN::Zero`] expects its
wires at degree 0, i.e. plain values; [`SPN::Zero`] is also the only operation
accepted on a derivative wire (degree > 0), where it denotes "no change".

## Examples

```
use theory::Theory;
use theory::spn::{SPN, Sort};

// Wires carry a sort and a degree; SPN operands are always of degree 0.
let deg0 = |s| Ok::<_, String>((s, 0u8));

// Emptiness of a place: Nat -> Bool. Clocks have their own test.
assert!(SPN::IsZero().check([Sort::Nat].map(deg0), [Sort::Bool].map(deg0)).is_ok());
assert!(SPN::IsZero().check([Sort::Clock].map(deg0), [Sort::Bool].map(deg0)).is_err());
assert!(SPN::ClkIsZero().check([Sort::Clock].map(deg0), [Sort::Bool].map(deg0)).is_ok());

// Firing a transition consumes a token: Nat -> Nat.
assert!(SPN::Dec().check([Sort::Nat].map(deg0), [Sort::Nat].map(deg0)).is_ok());

// A fresh Poisson clock with rate 2.5 reads nothing and writes a clock.
assert!(SPN::Pos(2.5).check([].map(deg0), [Sort::Clock].map(deg0)).is_ok());
assert!(SPN::Pos(2.5).check([].map(deg0), [Sort::Nat].map(deg0)).is_err());

// A derivative wire (degree 1) is written by `Zero` alone.
let der = |s| Ok::<_, String>((s, 1u8));
assert!(SPN::Zero(Sort::Nat).check([].map(deg0), [Sort::Nat].map(der)).is_ok());
assert!(SPN::Nat(0).check([].map(deg0), [Sort::Nat].map(der)).is_err());
```
*/

use crate::*;
#[cfg(feature = "pyo3")]
use pyo3::pyclass;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyclass(frozen, eq, str))]
pub enum Sort {
    /// The time left until a Poisson clock expires
    Clock,
    /// A token count (the marking of a place)
    Nat,
    /// A truth value
    Bool,
}

impl Sort {
    pub fn is_bool(&self) -> bool {
        matches!(self, Sort::Bool)
    }

    pub fn is_nat(&self) -> bool {
        matches!(self, Sort::Nat)
    }

    pub fn is_clock(&self) -> bool {
        matches!(self, Sort::Clock)
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Nat => write!(f, "Nat"),
            Sort::Bool => write!(f, "Bool"),
            Sort::Clock => write!(f, "Clock"),
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
    /// Nondeterministic choice of a value of the given sort
    #[strum(to_string = "(* : {0})")]
    Nondet(Sort),
    /// Arm a fresh Poisson clock with the given rate
    #[strum(to_string = "Pos({0})")]
    Pos(f64),
    /// Zero flow: the value of the given sort does not change. It is the
    /// operation [`Differential::zero`] resolves to.
    Zero(Sort),
}

impl Sequential for SPN {
    fn skip(_range: &Self::Sort) -> Self {
        SPN::Id()
    }
}

impl Combinatorial for SPN {
    fn havoc(range: &Sort) -> Self {
        SPN::Nondet(*range)
    }
}

impl Differential for SPN {
    fn zero(range: &Sort) -> Self {
        SPN::Zero(*range)
    }
}

impl Theory for SPN {
    type Sort = Sort;
    const NAME: &'static str = "SPN";

    fn check<R, W, E: fmt::Display>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<(Sort, u8), E>>,
        W: IntoIterator<Item = Result<(Sort, u8), E>>,
    {
        match self {
            SPN::Nat(_) | SPN::Bool(_) | SPN::Clock(_) => check_const(self, read, write),
            SPN::And() | SPN::Or() | SPN::Not() => check_bool(self, read, write),
            SPN::IsZero() | SPN::ClkIsZero() => check_is_zero(self, read, write),
            SPN::Inc() | SPN::Dec() => check_nat_ops(self, read, write),
            SPN::Id() => check_flow(self, read, write),
            SPN::Nondet(sort) => check_havoc(sort, read, write),
            SPN::Pos(_) => check_sample(self, read, write),
            SPN::Zero(sort) => check_zero(sort, read, write),
        }
    }
}

/// The sort a literal writes, i.e. the sort of the value it carries.
fn const_sort(op: &SPN) -> Sort {
    match op {
        SPN::Nat(_) => Sort::Nat,
        SPN::Bool(_) => Sort::Bool,
        SPN::Clock(_) => Sort::Clock,
        _ => unreachable!(),
    }
}

// A literal reads nothing and writes exactly one wire, whose sort must be the
// sort of the literal.
fn check_const<R, W, E: fmt::Display>(op: &SPN, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<(Sort, u8), E>>,
    W: IntoIterator<Item = Result<(Sort, u8), E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    if read.next().is_some() {
        return Err(format!("{op:?}: a constant cannot read values"));
    }
    let ((w1, degree), None) = (next_with_degree(&mut write, 0)?, write.next()) else {
        return Err(format!(
            "{op:?}: must write exactly one value (writes more)"
        ));
    };
    if degree != 0 {
        return Err(format!(
            "{op:?}: cannot write a constant to a wire of degree {degree}; \
             use ZERO to say that the value does not change"
        ));
    }
    let expected = const_sort(op);
    if w1 != expected {
        return Err(format!(
            "{op:?}: a {expected} constant cannot be written to a {w1} wire"
        ));
    }
    Ok(())
}

// `Not` is unary and `And`/`Or` are binary; all of their wires are Bool.
fn check_bool<R, W, E: fmt::Display>(op: &SPN, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<(Sort, u8), E>>,
    W: IntoIterator<Item = Result<(Sort, u8), E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        SPN::Not() => {
            let (r1, None) = (next_expect_degree(&mut read, 0, 0)?, read.next()) else {
                return Err(format!("{op:?}: must read a single value (reads more)"));
            };
            let (w1, None) = (next_expect_degree(&mut write, 0, 0)?, write.next()) else {
                return Err(format!("{op:?}: must write a single value (writes more)"));
            };
            if r1 != Sort::Bool {
                return Err(format!("{op:?}: input must be Bool, got {r1}"));
            }
            if w1 != Sort::Bool {
                return Err(format!("{op:?}: output must be Bool, got {w1}"));
            }
            Ok(())
        }
        SPN::And() | SPN::Or() => {
            let (r1, r2, None) = (
                next_expect_degree(&mut read, 0, 0)?,
                next_expect_degree(&mut read, 1, 0)?,
                read.next(),
            ) else {
                return Err(format!("{op:?}: must read exactly two values"));
            };
            let (w1, None) = (next_expect_degree(&mut write, 0, 0)?, write.next()) else {
                return Err(format!("{op:?}: must write exactly one value"));
            };
            if r1 != Sort::Bool || r2 != Sort::Bool {
                return Err(format!("{op:?}: inputs must be Bool, got {r1} and {r2}"));
            }
            if w1 != Sort::Bool {
                return Err(format!("{op:?}: output must be Bool, got {w1}"));
            }
            Ok(())
        }
        _ => unreachable!(),
    }
}

// Both tests are unary and produce a Bool; they differ in the sort they accept:
// `IsZero` tests a place (Nat), `ClkIsZero` a clock.
fn check_is_zero<R, W, E: fmt::Display>(op: &SPN, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<(Sort, u8), E>>,
    W: IntoIterator<Item = Result<(Sort, u8), E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let expected = match op {
        SPN::IsZero() => Sort::Nat,
        SPN::ClkIsZero() => Sort::Clock,
        _ => unreachable!(),
    };
    let (r1, None) = (next_expect_degree(&mut read, 0, 0)?, read.next()) else {
        return Err(format!("{op:?}: must read exactly one value"));
    };
    let (w1, None) = (next_expect_degree(&mut write, 0, 0)?, write.next()) else {
        return Err(format!("{op:?}: must write exactly one value"));
    };
    if r1 != expected {
        return Err(format!("{op:?}: input must be {expected}, got {r1}"));
    }
    if w1 != Sort::Bool {
        return Err(format!("{op:?}: output must be Bool, got {w1}"));
    }
    Ok(())
}

// `Inc` and `Dec` are unary maps on token counts.
fn check_nat_ops<R, W, E: fmt::Display>(op: &SPN, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<(Sort, u8), E>>,
    W: IntoIterator<Item = Result<(Sort, u8), E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let (r1, None) = (next_expect_degree(&mut read, 0, 0)?, read.next()) else {
        return Err(format!("{op:?}: must read exactly one value"));
    };
    let (w1, None) = (next_expect_degree(&mut write, 0, 0)?, write.next()) else {
        return Err(format!("{op:?}: must write exactly one value"));
    };
    if r1 != Sort::Nat {
        return Err(format!("{op:?}: input must be Nat, got {r1}"));
    }
    if w1 != Sort::Nat {
        return Err(format!("{op:?}: output must be Nat, got {w1}"));
    }
    Ok(())
}

// `Id` copies a single value of an arbitrary sort.
fn check_flow<R, W, E: fmt::Display>(op: &SPN, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<(Sort, u8), E>>,
    W: IntoIterator<Item = Result<(Sort, u8), E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let (r1, None) = (next_expect_degree(&mut read, 0, 0)?, read.next()) else {
        return Err(format!("{op:?}: must read exactly one value"));
    };
    let (w1, None) = (next_expect_degree(&mut write, 0, 0)?, write.next()) else {
        return Err(format!("{op:?}: must write exactly one value"));
    };
    if r1 != w1 {
        return Err(format!(
            "{op:?}: input and output must have the same sort, got {r1} and {w1}"
        ));
    }
    Ok(())
}

// Sampling a clock reads nothing and writes exactly one Clock wire; the rate is
// part of the operation.
fn check_sample<R, W, E: fmt::Display>(op: &SPN, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<(Sort, u8), E>>,
    W: IntoIterator<Item = Result<(Sort, u8), E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    if read.next().is_some() {
        return Err(format!("{op:?}: cannot read values"));
    }
    let (w1, None) = (next_expect_degree(&mut write, 0, 0)?, write.next()) else {
        return Err(format!("{op:?}: must write exactly one value"));
    };
    if w1 != Sort::Clock {
        return Err(format!("{op:?}: output must be Clock, got {w1}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAT: Sort = Sort::Nat;
    const BOOL: Sort = Sort::Bool;
    const CLOCK: Sort = Sort::Clock;

    fn deg0(s: Sort) -> Result<(Sort, u8), String> {
        Ok((s, 0))
    }

    fn deg1(s: Sort) -> Result<(Sort, u8), String> {
        Ok((s, 1))
    }

    #[test]
    fn sort_predicates() {
        assert!(NAT.is_nat() && !NAT.is_bool() && !NAT.is_clock());
        assert!(BOOL.is_bool() && !BOOL.is_nat() && !BOOL.is_clock());
        assert!(CLOCK.is_clock() && !CLOCK.is_nat() && !CLOCK.is_bool());
    }

    #[test]
    fn sort_display() {
        assert_eq!(
            (NAT.to_string(), BOOL.to_string(), CLOCK.to_string()),
            ("Nat".to_string(), "Bool".to_string(), "Clock".to_string())
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
        // as for `Any::ZERO`, the sort is not part of the rendering
        assert_eq!(SPN::Zero(CLOCK).to_string(), "Zero");
    }

    #[test]
    fn const_ok() {
        assert!(SPN::Nat(7).check([].map(deg0), [NAT].map(deg0)).is_ok());
        assert!(
            SPN::Bool(false)
                .check([].map(deg0), [BOOL].map(deg0))
                .is_ok()
        );
        assert!(
            SPN::Clock(0.5)
                .check([].map(deg0), [CLOCK].map(deg0))
                .is_ok()
        );
    }

    #[test]
    fn const_wrong_sort_fails() {
        assert!(SPN::Nat(7).check([].map(deg0), [BOOL].map(deg0)).is_err());
        assert!(
            SPN::Bool(true)
                .check([].map(deg0), [NAT].map(deg0))
                .is_err()
        );
        assert!(
            SPN::Clock(0.5)
                .check([].map(deg0), [NAT].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn const_with_read_fails() {
        assert!(SPN::Nat(7).check([NAT].map(deg0), [NAT].map(deg0)).is_err());
    }

    #[test]
    fn const_without_write_fails() {
        assert!(SPN::Nat(7).check([].map(deg0), [].map(deg0)).is_err());
    }

    #[test]
    fn const_with_two_writes_fails() {
        assert!(
            SPN::Nat(7)
                .check([].map(deg0), [NAT, NAT].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn const_nonzero_degree_fails() {
        assert!(SPN::Nat(7).check([].map(deg0), [NAT].map(deg1)).is_err());
    }

    #[test]
    fn not_ok() {
        assert!(SPN::Not().check([BOOL].map(deg0), [BOOL].map(deg0)).is_ok());
    }

    #[test]
    fn not_non_bool_fails() {
        assert!(SPN::Not().check([NAT].map(deg0), [BOOL].map(deg0)).is_err());
        assert!(SPN::Not().check([BOOL].map(deg0), [NAT].map(deg0)).is_err());
    }

    #[test]
    fn not_two_reads_fails() {
        assert!(
            SPN::Not()
                .check([BOOL, BOOL].map(deg0), [BOOL].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn and_or_ok() {
        assert!(
            SPN::And()
                .check([BOOL, BOOL].map(deg0), [BOOL].map(deg0))
                .is_ok()
        );
        assert!(
            SPN::Or()
                .check([BOOL, BOOL].map(deg0), [BOOL].map(deg0))
                .is_ok()
        );
    }

    #[test]
    fn and_non_bool_fails() {
        assert!(
            SPN::And()
                .check([NAT, BOOL].map(deg0), [BOOL].map(deg0))
                .is_err()
        );
        assert!(
            SPN::And()
                .check([BOOL, CLOCK].map(deg0), [BOOL].map(deg0))
                .is_err()
        );
        assert!(
            SPN::And()
                .check([BOOL, BOOL].map(deg0), [NAT].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn and_one_read_fails() {
        assert!(
            SPN::And()
                .check([BOOL].map(deg0), [BOOL].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn is_zero_ok() {
        assert!(
            SPN::IsZero()
                .check([NAT].map(deg0), [BOOL].map(deg0))
                .is_ok()
        );
    }

    #[test]
    fn is_zero_wrong_sorts_fail() {
        // clocks have their own test
        assert!(
            SPN::IsZero()
                .check([CLOCK].map(deg0), [BOOL].map(deg0))
                .is_err()
        );
        assert!(
            SPN::IsZero()
                .check([NAT].map(deg0), [NAT].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn clk_is_zero_ok() {
        assert!(
            SPN::ClkIsZero()
                .check([CLOCK].map(deg0), [BOOL].map(deg0))
                .is_ok()
        );
    }

    #[test]
    fn clk_is_zero_wrong_sorts_fail() {
        assert!(
            SPN::ClkIsZero()
                .check([NAT].map(deg0), [BOOL].map(deg0))
                .is_err()
        );
        assert!(
            SPN::ClkIsZero()
                .check([CLOCK].map(deg0), [CLOCK].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn inc_dec_ok() {
        assert!(SPN::Inc().check([NAT].map(deg0), [NAT].map(deg0)).is_ok());
        assert!(SPN::Dec().check([NAT].map(deg0), [NAT].map(deg0)).is_ok());
    }

    #[test]
    fn inc_dec_non_nat_fails() {
        assert!(
            SPN::Inc()
                .check([CLOCK].map(deg0), [CLOCK].map(deg0))
                .is_err()
        );
        assert!(SPN::Dec().check([BOOL].map(deg0), [NAT].map(deg0)).is_err());
        assert!(SPN::Dec().check([NAT].map(deg0), [BOOL].map(deg0)).is_err());
    }

    #[test]
    fn inc_arity_fails() {
        assert!(
            SPN::Inc()
                .check([NAT, NAT].map(deg0), [NAT].map(deg0))
                .is_err()
        );
        assert!(SPN::Inc().check([].map(deg0), [NAT].map(deg0)).is_err());
    }

    #[test]
    fn id_ok_on_every_sort() {
        for s in [NAT, BOOL, CLOCK] {
            assert!(SPN::Id().check([s].map(deg0), [s].map(deg0)).is_ok());
        }
    }

    #[test]
    fn id_sort_mismatch_fails() {
        assert!(SPN::Id().check([NAT].map(deg0), [CLOCK].map(deg0)).is_err());
    }

    #[test]
    fn id_arity_fails() {
        assert!(
            SPN::Id()
                .check([NAT].map(deg0), [NAT, NAT].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn skip_is_id() {
        assert!(matches!(<SPN as Sequential>::skip(&NAT), SPN::Id()));
    }

    #[test]
    fn nondet_ok_on_every_sort() {
        for s in [NAT, BOOL, CLOCK] {
            let op = <SPN as Combinatorial>::havoc(&s);
            assert!(matches!(op, SPN::Nondet(r) if r == s));
            assert!(op.check([].map(deg0), [s].map(deg0)).is_ok());
        }
    }

    #[test]
    fn nondet_wrong_sort_fails() {
        assert!(
            SPN::Nondet(NAT)
                .check([].map(deg0), [BOOL].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn nondet_with_read_fails() {
        assert!(
            SPN::Nondet(NAT)
                .check([NAT].map(deg0), [NAT].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn zero_ok_on_every_sort() {
        for s in [NAT, BOOL, CLOCK] {
            let op = <SPN as Differential>::zero(&s);
            assert!(matches!(op, SPN::Zero(r) if r == s));
            assert!(op.check([].map(deg0), [s].map(deg0)).is_ok());
        }
    }

    #[test]
    fn zero_writes_derivative_wires() {
        // ZERO is the only operation accepted on a wire of degree > 0
        assert!(SPN::Zero(NAT).check([].map(deg0), [NAT].map(deg1)).is_ok());
        assert!(SPN::Nat(0).check([].map(deg0), [NAT].map(deg1)).is_err());
        assert!(SPN::Id().check([NAT].map(deg1), [NAT].map(deg1)).is_err());
    }

    #[test]
    fn zero_wrong_sort_fails() {
        assert!(
            SPN::Zero(NAT)
                .check([].map(deg0), [CLOCK].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn zero_with_read_fails() {
        assert!(
            SPN::Zero(NAT)
                .check([NAT].map(deg0), [NAT].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn zero_arity_mismatch_fails() {
        assert!(SPN::Zero(NAT).check([].map(deg0), [].map(deg0)).is_err());
        assert!(
            SPN::Zero(NAT)
                .check([].map(deg0), [NAT, NAT].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn pos_ok() {
        assert!(SPN::Pos(2.5).check([].map(deg0), [CLOCK].map(deg0)).is_ok());
    }

    #[test]
    fn pos_non_clock_write_fails() {
        assert!(SPN::Pos(2.5).check([].map(deg0), [NAT].map(deg0)).is_err());
    }

    #[test]
    fn pos_with_read_fails() {
        assert!(
            SPN::Pos(2.5)
                .check([CLOCK].map(deg0), [CLOCK].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn pos_two_writes_fails() {
        assert!(
            SPN::Pos(2.5)
                .check([].map(deg0), [CLOCK, CLOCK].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn read_error_is_propagated() {
        let err: Result<(Sort, u8), String> = Err("broken wire".to_string());
        assert!(SPN::Inc().check([err], [NAT].map(deg0)).is_err());
    }
}
