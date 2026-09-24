/*!
# Stochastic Petri nets

Defines the minimalistic theory [`SPN`] for modelling stochastic Petri nets:
places holding tokens, transitions moving tokens around, and Poisson clocks
deciding when a transition fires.

Unlike [`crate::lia`], the theory this one is derived from, the sorts are not
matrix sorts. A [`Sort`] value is one of four scalars, plus the trivial
tangent:

- `Nat` — a token count (the marking of a single place),
- `Bool` — a truth value (e.g. whether a transition is enabled),
- `Event` — a truth value again, under another name: the momentary signal that
  something has happened (a transition has fired). It is an alias of `Bool` in
  behaviour — every operation takes the one for the other — and a sort of its
  own only in being recognisable as such, see [`Sort::is_event`],
- `Clock { rank }` — the time left until a clock expires; `rank` is the
  differential grade: 0 is a value, 1 a first derivative, and so on — the sort
  former [`Tangent`] raises it,
- `Zero` — the trivial tangent of the constant sorts, the inhabited singleton
  whose only writer is the `zero` generator.

The clock is the only sort that can move: a marking changes by firing a
transition, not by flowing, so `Nat`, `Bool` and `Event` are constant sorts and
their tangent is `Zero`. The clock tangent is a sort of its own, inhabited by
the constant rates a clock can run at.

The operations in [`SPN`] are:

- [`SPN::Nat`], [`SPN::Bool`], [`SPN::Clock`] — literals; each reads nothing and
  writes a single wire of the corresponding value sort — the truth-value literal
  writes a `Bool` and an `Event` wire alike.
- [`SPN::And`], [`SPN::Or`], [`SPN::Not`] — boolean operations on `Bool`, and so
  on `Event`.
- [`SPN::IsZero`], [`SPN::ClkIsZero`] — tests for an empty place (`Nat`) and for
  an expired clock (`Clock`); both produce a `Bool` — or an `Event`, e.g. the
  firing an expiry triggers.
- [`SPN::Inc`], [`SPN::Dec`] — produce and consume a token: `Nat -> Nat`.
- [`SPN::Id`] — copies its single read wire to its single write wire; it is
  defined on every sort and acts rank-generically.
- [`SPN::Ite`] — if-then-else: reads a boolean guard and two branches of one
  and the same sort, and writes that sort.
- [`SPN::IfThen`] — if-then: [`SPN::Ite`] without the else branch, and so the
  one **partial** generator of the theory. It reads a boolean guard and one
  value and writes that value's sort: where the guard holds it is
  [`SPN::Id`] of the branch, and where it does not it has no value at all.
  Sort checking stays total — partiality is about values, not sorts — so what
  a step does where the guard fails is the executor's obligation, as with the
  passage of time below. Read the term as a condition on the steps that exist
  rather than as a choice between two values.
- [`SPN::Nondet`]`(s)` — nondeterministic choice of a value of the value sort
  `s`; reads nothing and writes a single wire of that sort.
- [`SPN::Exp`]`(λ)` — arms a fresh clock, its time to expiry drawn from the
  exponential distribution with rate `λ`; reads nothing and writes a single
  `Clock` value wire.
- [`SPN::ClkRate`]`(c)` — constant flow of a clock: reads nothing and writes a
  single clock *tangent* wire (rank at least 1), saying that the clock runs at
  rate `c` — `-1` for a clock counting down to its expiry.
- [`SPN::ClkZero`] — the zero flow of a clock: `ClkRate(0)` under its own name,
  since the zero flow is the generator [`Differential::zero`] resolves to.
- [`SPN::ClkMul`]`(k)` — a clock's flow relative to another's: reads one clock
  tangent and writes a clock tangent of the same rank scaled by `k`, so
  `d(c) = ClkMul(-1)(d(t))` says `c` counts down at the rate `t` counts up.
- [`SPN::Zero`] — the unique inhabitant of `Zero`, the trivial tangent of the
  constant sorts.

There are no ordering comparisons and no arithmetic beyond `Inc`/`Dec` on tokens
and `ClkMul` on rates: the guards of a Petri net only ask whether a place is
empty or whether a clock has expired.

[`SPN::ClkIsZero`] holds exactly at zero, and the theory says which clocks run
and how fast, never when time stops: advancing to the first expiry — the
minimum over the running clocks — and taking the discrete step there is the
executor's obligation. A minimum is not expressible in this signature, and
deliberately so; a model with several clocks is well defined only under that
convention.

`SPN` implements [`Signature`], [`Sequential`], [`Combinatorial`] and
[`Differential`]; [`Signature::check`] validates the sorts of the read/write
wires against the selected operation. [`Differential::zero`] resolves to
[`SPN::ClkZero`] on a clock tangent and to [`SPN::Zero`] on the trivial
one; together with [`SPN::ClkRate`] they are the only generators that write a
tangent wire.

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

// `Event` is `Bool` under another name: the operations take the one for the
// other, and only `is_event` tells them apart.
assert!(SPN::ClkIsZero().check([clk].map(ok), [Sort::Event()].map(ok)).is_ok());
assert!(SPN::And()
    .check([Sort::Event(), Sort::Bool()].map(ok), [Sort::Event()].map(ok))
    .is_ok());
assert!(Sort::Event().is_bool() && Sort::Event().is_event() && !Sort::Bool().is_event());

// Firing a transition consumes a token: Nat -> Nat.
assert!(SPN::Dec().check([Sort::Nat()].map(ok), [Sort::Nat()].map(ok)).is_ok());

// A fresh exponential clock of rate 2.5 reads nothing and writes a clock value.
assert!(SPN::Exp(2.5).check([].map(ok), [clk].map(ok)).is_ok());
assert!(SPN::Exp(2.5).check([].map(ok), [Sort::Nat()].map(ok)).is_err());

// Tangent wires are written by the flow generators alone: a clock counts
// down at rate -1, a marking does not move at all.
assert!(SPN::ClkRate(-1.0).check([].map(ok), [clk.T()].map(ok)).is_ok());
assert!(SPN::ClkRate(-1.0).check([].map(ok), [clk].map(ok)).is_err());
assert!(SPN::ClkZero().check([].map(ok), [clk.T()].map(ok)).is_ok());
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
    /// Natural number -- for representing a token count (the marking of a place)
    Nat(),
    /// A truth value -- for predicates.
    Bool(),
    /// A truth value again, under another name: the momentary signal that
    /// something has happened (a transition has fired). `Event` is an alias of
    /// [`Sort::Bool`] in behaviour -- every operation takes the one for the
    /// other -- and a sort of its own only in that a wire records which of the
    /// two names it was declared with, see [`Sort::is_event`].
    Event(),
    /// The trivial tangent of the constant sorts: a singleton, inhabited by
    /// exactly the zero value.
    Zero(),
}

impl Sort {
    /// A clock value sort (rank 0).
    pub fn clock() -> Self {
        Sort::Clock { rank: 0 }
    }

    /// True of the boolean fragment: [`Sort::Bool`] and its alias
    /// [`Sort::Event`] alike.
    pub fn is_bool(&self) -> bool {
        matches!(self, Sort::Bool() | Sort::Event())
    }

    /// True of [`Sort::Event`] alone. The alias is transparent to every
    /// operation of the theory; this is what still tells the two apart.
    pub fn is_event(&self) -> bool {
        matches!(self, Sort::Event())
    }

    /// Do the two sorts describe the same values? Sort equality up to the
    /// `Event`/`Bool` alias. The checks below compare sorts with this and
    /// never with `==`, which is what makes an `Event` wire readable and
    /// writable wherever a `Bool` one is.
    pub fn agrees(&self, other: &Sort) -> bool {
        if self.is_bool() {
            other.is_bool()
        } else {
            self == other
        }
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
            Sort::Nat() | Sort::Bool() | Sort::Event() | Sort::Zero() => None,
        }
    }
}

/// The tangent former: a clock grades up (`rank + 1`); the constant sorts
/// `Nat`, `Bool` and `Event` collapse to the trivial tangent `Zero`, which is a
/// fixed point.
impl Tangent for Sort {
    #[allow(non_snake_case)]
    fn T(&self) -> Self {
        match *self {
            Sort::Clock { rank } => Sort::Clock { rank: rank + 1 },
            Sort::Nat() | Sort::Bool() | Sort::Event() | Sort::Zero() => Sort::Zero(),
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
            Sort::Event() => write!(f, "Event"),
            Sort::Zero() => write!(f, "Zero"),
        }
    }
}

/// Theory of stochastic Petri nets over the scalar sorts [`Sort::Nat`],
/// [`Sort::Bool`] and [`Sort::Clock`].
#[derive(Clone, Debug, strum::Display)]
#[cfg_attr(feature = "pyo3", pyclass(frozen, str))]
pub enum SPN {
    // constants
    /// A token-count literal
    #[strum(to_string = "({0} : nat)")]
    Nat(u64),
    /// A truth-value literal; it writes a `Bool` wire and an `Event` one alike
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
    /// Has the clock expired? `Clock -> Bool`. The test is exact: see the
    /// module docs on who is responsible for stopping time at zero.
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
    /// If-then-else
    Ite(),
    /// If-then (**partial operation**): [`SPN::Ite`] without the else branch,
    /// so it copies the branch where the guard holds and has no value where it
    /// does not
    IfThen(),
    /// Nondeterministic choice of a value of the given value sort
    #[strum(to_string = "(* : {0})")]
    Nondet(Sort),
    /// Arm a fresh clock, its time to expiry drawn from the exponential
    /// distribution with the given rate
    #[strum(to_string = "Exp({0})")]
    Exp(f64),
    /// Constant flow of a clock: it runs at the given rate, `-1` for a clock
    /// counting down to its expiry. It writes a clock tangent (rank at least
    /// 1) and reads nothing.
    #[strum(to_string = "ClkRate({0})")]
    ClkRate(f64),
    /// Zero flow of a clock: the clock does not run. It is [`SPN::ClkRate`]`(0)`
    /// under its own name — the generator [`Differential::zero`] resolves to on
    /// the clock fragment.
    ClkZero(),
    /// A clock's flow relative to another's: reads one clock tangent and writes a
    /// clock tangent of the same rank, scaled by the constant.
    #[strum(to_string = "ClkMul({0})")]
    ClkMul(f64),
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
            // the trivial tangent is a singleton, so havoc over it is its
            // single inhabitant; every other sort, clock tangents included,
            // has a choice to make
            Sort::Zero() => SPN::Zero(),
            _ => SPN::Nondet(*range),
        }
    }
}

impl Differential for SPN {
    fn zero(range: &Sort) -> Self {
        // `range` is the tangent sort the generator writes
        match range {
            Sort::Clock { .. } => SPN::ClkZero(),
            Sort::Nat() | Sort::Bool() | Sort::Event() | Sort::Zero() => SPN::Zero(),
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
            SPN::Id() | SPN::Ite() | SPN::IfThen() => check_flow(self, read, write),
            SPN::Nondet(sort) => check_nondet(sort, read, write),
            SPN::Exp(_) => check_sample(self, read, write),
            SPN::ClkRate(_) | SPN::ClkZero() => check_clk_rate(self, read, write),
            SPN::ClkMul(_) => check_clk_mul(self, read, write),
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
        _ if !w1.agrees(&expected) => Err(format!(
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
            if !r1.is_bool() {
                return Err(format!("{op:?}: input must be Bool, got {r1}"));
            }
            if !w1.is_bool() {
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
            if !r1.is_bool() || !r2.is_bool() {
                return Err(format!("{op:?}: inputs must be Bool, got {r1} and {r2}"));
            }
            if !w1.is_bool() {
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
    if !w1.is_bool() {
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

// `Id` copies a single value of an arbitrary sort, `Ite` selects between two
// values of one sort under a boolean guard, and `IfThen` is that guard without
// the second branch. All three apply rank-generically, but never across ranks:
// the sorts they relate must agree, tangent grade included. Only the sorts are
// checked here: that `IfThen` has no value where its guard fails is a fact
// about steps, not about wires.
fn check_flow<R, W, E: fmt::Display>(op: &SPN, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        SPN::Id() => {
            let (r1, None) = (next_sort(&mut read, 0)?, read.next()) else {
                return Err(format!("{op:?}: must read exactly one value"));
            };
            let (w1, None) = (next_sort(&mut write, 0)?, write.next()) else {
                return Err(format!("{op:?}: must write exactly one value"));
            };
            if !r1.agrees(&w1) {
                return Err(format!(
                    "{op:?}: input and output must have the same sort, got {r1} and {w1}"
                ));
            }
            Ok(())
        }
        SPN::Ite() => {
            let (r1, r2, r3, None) = (
                next_sort(&mut read, 0)?,
                next_sort(&mut read, 1)?,
                next_sort(&mut read, 2)?,
                read.next(),
            ) else {
                return Err(format!("{op:?}: must read exactly three values"));
            };
            let (w1, None) = (next_sort(&mut write, 0)?, write.next()) else {
                return Err(format!("{op:?}: must write exactly one value"));
            };
            if !r1.is_bool() {
                return Err(format!("{op:?}: the guard must be Bool, got {r1}"));
            }
            if !r2.agrees(&r3) {
                return Err(format!(
                    "{op:?}: the branches must have the same sort, got {r2} and {r3}"
                ));
            }
            if !w1.agrees(&r2) {
                return Err(format!(
                    "{op:?}: output must have the sort of the branches, got {w1} and {r2}"
                ));
            }
            Ok(())
        }
        SPN::IfThen() => {
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
            if !r1.is_bool() {
                return Err(format!("{op:?}: the guard must be Bool, got {r1}"));
            }
            if !w1.agrees(&r2) {
                return Err(format!(
                    "{op:?}: output must have the sort of the branch, got {w1} and {r2}"
                ));
            }
            Ok(())
        }
        _ => unreachable!(),
    }
}

// Nondeterministic choice is defined on every sort but the trivial tangent,
// which is inhabited by zero alone — there is nothing to choose there. On a
// clock tangent it is the rate that is chosen: a clock of unknown speed. It
// reads nothing and writes one wire of the sort it carries — up to the
// `Event`/`Bool` alias, which is why this is not [`check_havoc`].
fn check_nondet<R, W, E: fmt::Display>(sort: &Sort, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    // exhaustive on purpose: a new sort has to say whether it can be chosen
    match sort {
        Sort::Zero() => {
            return Err(format!(
                "Nondet: the tangent sort {sort} is inhabited by zero alone; use ZERO"
            ));
        }
        Sort::Clock { .. } | Sort::Nat() | Sort::Bool() | Sort::Event() => {}
    }
    if read.into_iter().next().is_some() {
        return Err(format!("Nondet({sort}): cannot read values"));
    }
    let mut write = write.into_iter();
    let (w1, None) = (next_sort(&mut write, 0)?, write.next()) else {
        return Err(format!("Nondet({sort}): must write exactly one value"));
    };
    if !w1.agrees(sort) {
        return Err(format!("Nondet({sort}): output must be {sort}, got {w1}"));
    }
    Ok(())
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

// The clock flows, `ClkRate(c)` and its zero `ClkZero`: each writes exactly one
// clock *tangent* wire (rank at least 1), and reads nothing.
fn check_clk_rate<R, W, E: fmt::Display>(op: &SPN, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    if read.into_iter().next().is_some() {
        return Err(format!("{op:?}: a flow cannot read values"));
    }
    let mut write = write.into_iter();
    match write.next() {
        Some(Ok(Sort::Clock { rank })) if rank >= 1 => {}
        Some(Ok(sort)) => {
            return Err(format!("{op:?}: must write a clock tangent, got {sort}"));
        }
        Some(Err(e)) => return Err(e.to_string()),
        None => return Err(format!("{op:?}: must write exactly one value, got none")),
    }
    if write.next().is_some() {
        return Err(format!("{op:?}: must write exactly one value, got more"));
    }
    Ok(())
}

// ClkMul scales a tangent: it reads exactly one clock tangent and writes exactly
// one of the same rank; a clock value (rank 0) has no rate to scale.
fn check_clk_mul<R, W, E: fmt::Display>(op: &SPN, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    let mut read = read.into_iter();
    let rank = match read.next() {
        Some(Ok(Sort::Clock { rank })) if rank >= 1 => rank,
        Some(Ok(sort)) => return Err(format!("{op:?}: must read a clock tangent, got {sort}")),
        Some(Err(e)) => return Err(e.to_string()),
        None => return Err(format!("{op:?}: must read exactly one value, got none")),
    };
    if read.next().is_some() {
        return Err(format!("{op:?}: must read exactly one value, got more"));
    }
    let mut write = write.into_iter();
    match write.next() {
        Some(Ok(Sort::Clock { rank: r })) if r == rank => {}
        Some(Ok(sort)) => {
            return Err(format!(
                "{op:?}: must write a clock tangent of rank {rank}, got {sort}"
            ));
        }
        Some(Err(e)) => return Err(e.to_string()),
        None => return Err(format!("{op:?}: must write exactly one value, got none")),
    }
    if write.next().is_some() {
        return Err(format!("{op:?}: must write exactly one value, got more"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAT: Sort = Sort::Nat();
    const BOOL: Sort = Sort::Bool();
    /// `Bool` under its other name: the theory's operations cannot tell the two
    /// apart, everything else can.
    const EVENT: Sort = Sort::Event();
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
        // an event is a boolean, and is the only sort that is also an event
        assert!(EVENT.is_bool() && !EVENT.is_nat() && !EVENT.is_clock());
        assert!(EVENT.is_event() && !BOOL.is_event() && !NAT.is_event());
        assert_eq!(EVENT.rank(), None);
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
        assert_eq!(EVENT.T(), ZERO);
        assert_eq!(ZERO.T(), ZERO);
    }

    #[test]
    fn agrees_is_equality_up_to_the_event_alias() {
        assert!(BOOL.agrees(&EVENT) && EVENT.agrees(&BOOL));
        for s in [NAT, BOOL, EVENT, CLOCK, DCLOCK, ZERO] {
            assert!(s.agrees(&s));
        }
        // no other pair of sorts collapses
        for (a, b) in [(NAT, BOOL), (NAT, EVENT), (CLOCK, DCLOCK), (ZERO, EVENT)] {
            assert!(!a.agrees(&b) && !b.agrees(&a));
        }
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
        // the alias is a name of its own, and prints as one
        assert_eq!(EVENT.to_string(), "Event".to_string());
    }

    #[test]
    fn op_display() {
        assert_eq!(SPN::Nat(3).to_string(), "(3 : nat)");
        assert_eq!(SPN::Bool(true).to_string(), "(true : bool)");
        assert_eq!(SPN::Clock(1.5).to_string(), "(1.5 : clk)");
        assert_eq!(SPN::Nondet(NAT).to_string(), "(* : Nat)");
        assert_eq!(SPN::Exp(2.5).to_string(), "Exp(2.5)");
        assert_eq!(SPN::ClkRate(-1.0).to_string(), "ClkRate(-1)");
        assert_eq!(SPN::Id().to_string(), "Id");
        assert_eq!(SPN::Ite().to_string(), "Ite");
        assert_eq!(SPN::IfThen().to_string(), "IfThen");
        assert_eq!(SPN::ClkZero().to_string(), "ClkZero");
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
    fn event_passes_for_bool_everywhere() {
        // a truth-value literal writes either name
        assert!(SPN::Bool(true).check([].map(ok), [EVENT].map(ok)).is_ok());
        // the boolean operations mix the two freely
        assert!(SPN::Not().check([EVENT].map(ok), [BOOL].map(ok)).is_ok());
        assert!(SPN::Not().check([BOOL].map(ok), [EVENT].map(ok)).is_ok());
        assert!(
            SPN::And()
                .check([EVENT, BOOL].map(ok), [EVENT].map(ok))
                .is_ok()
        );
        // a test may write its outcome to an event: the firing an expiry triggers
        assert!(SPN::IsZero().check([NAT].map(ok), [EVENT].map(ok)).is_ok());
        assert!(
            SPN::ClkIsZero()
                .check([CLOCK].map(ok), [EVENT].map(ok))
                .is_ok()
        );
        // `Id` copies across the alias, and `Ite` guards and branches on it
        assert!(SPN::Id().check([EVENT].map(ok), [BOOL].map(ok)).is_ok());
        assert!(SPN::Id().check([BOOL].map(ok), [EVENT].map(ok)).is_ok());
        assert!(
            SPN::Ite()
                .check([EVENT, NAT, NAT].map(ok), [NAT].map(ok))
                .is_ok()
        );
        assert!(
            SPN::Ite()
                .check([BOOL, EVENT, BOOL].map(ok), [EVENT].map(ok))
                .is_ok()
        );
        // and nondeterministic choice writes it under either name
        assert!(SPN::Nondet(EVENT).check([].map(ok), [BOOL].map(ok)).is_ok());
        assert!(SPN::Nondet(BOOL).check([].map(ok), [EVENT].map(ok)).is_ok());
    }

    #[test]
    fn event_is_not_the_other_sorts() {
        // the alias reaches `Bool` alone: everything else still fails
        assert!(SPN::Nat(7).check([].map(ok), [EVENT].map(ok)).is_err());
        assert!(SPN::Not().check([NAT].map(ok), [EVENT].map(ok)).is_err());
        assert!(SPN::Inc().check([EVENT].map(ok), [NAT].map(ok)).is_err());
        assert!(
            SPN::IsZero()
                .check([EVENT].map(ok), [BOOL].map(ok))
                .is_err()
        );
        assert!(SPN::Id().check([EVENT].map(ok), [NAT].map(ok)).is_err());
        assert!(SPN::Nondet(EVENT).check([].map(ok), [NAT].map(ok)).is_err());
        assert!(SPN::Zero().check([].map(ok), [EVENT].map(ok)).is_err());
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
        for s in [NAT, BOOL, EVENT, CLOCK, DCLOCK, ZERO] {
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
    fn ite_ok_on_every_sort() {
        // the guard is a Bool, the branches and the result share one sort
        for s in [NAT, BOOL, EVENT, CLOCK, DCLOCK, ZERO] {
            assert!(SPN::Ite().check([BOOL, s, s].map(ok), [s].map(ok)).is_ok());
        }
    }

    #[test]
    fn ite_non_bool_guard_fails() {
        assert!(
            SPN::Ite()
                .check([NAT, NAT, NAT].map(ok), [NAT].map(ok))
                .is_err()
        );
        assert!(
            SPN::Ite()
                .check([CLOCK, CLOCK, CLOCK].map(ok), [CLOCK].map(ok))
                .is_err()
        );
    }

    #[test]
    fn ite_branch_mismatch_fails() {
        assert!(
            SPN::Ite()
                .check([BOOL, NAT, CLOCK].map(ok), [NAT].map(ok))
                .is_err()
        );
        // the branches agree, the output does not
        assert!(
            SPN::Ite()
                .check([BOOL, NAT, NAT].map(ok), [BOOL].map(ok))
                .is_err()
        );
    }

    #[test]
    fn ite_rank_mismatch_fails() {
        // `Ite` applies rank-generically, but never across ranks
        assert!(
            SPN::Ite()
                .check([BOOL, CLOCK, DCLOCK].map(ok), [CLOCK].map(ok))
                .is_err()
        );
        assert!(
            SPN::Ite()
                .check([BOOL, DCLOCK, DCLOCK].map(ok), [CLOCK].map(ok))
                .is_err()
        );
    }

    #[test]
    fn ite_arity_fails() {
        assert!(
            SPN::Ite()
                .check([BOOL, NAT].map(ok), [NAT].map(ok))
                .is_err()
        );
        assert!(
            SPN::Ite()
                .check([BOOL, NAT, NAT, NAT].map(ok), [NAT].map(ok))
                .is_err()
        );
        assert!(
            SPN::Ite()
                .check([BOOL, NAT, NAT].map(ok), [NAT, NAT].map(ok))
                .is_err()
        );
        assert!(
            SPN::Ite()
                .check([BOOL, NAT, NAT].map(ok), [].map(ok))
                .is_err()
        );
    }

    #[test]
    fn ifthen_ok_on_every_sort() {
        // the guard is a Bool, the one branch and the result share a sort
        for s in [NAT, BOOL, EVENT, CLOCK, DCLOCK, ZERO] {
            assert!(SPN::IfThen().check([BOOL, s].map(ok), [s].map(ok)).is_ok());
        }
    }

    #[test]
    fn ifthen_non_bool_guard_fails() {
        assert!(
            SPN::IfThen()
                .check([NAT, NAT].map(ok), [NAT].map(ok))
                .is_err()
        );
        assert!(
            SPN::IfThen()
                .check([CLOCK, CLOCK].map(ok), [CLOCK].map(ok))
                .is_err()
        );
    }

    #[test]
    fn ifthen_branch_mismatch_fails() {
        assert!(
            SPN::IfThen()
                .check([BOOL, NAT].map(ok), [CLOCK].map(ok))
                .is_err()
        );
    }

    #[test]
    fn ifthen_rank_mismatch_fails() {
        // rank-generic like `Id` and `Ite`, and like them never across ranks
        assert!(
            SPN::IfThen()
                .check([BOOL, CLOCK].map(ok), [DCLOCK].map(ok))
                .is_err()
        );
    }

    #[test]
    fn ifthen_arity_fails() {
        // the else branch is exactly the argument it does not take
        assert!(
            SPN::IfThen()
                .check([BOOL, NAT, NAT].map(ok), [NAT].map(ok))
                .is_err()
        );
        assert!(SPN::IfThen().check([BOOL].map(ok), [NAT].map(ok)).is_err());
        assert!(
            SPN::IfThen()
                .check([BOOL, NAT].map(ok), [NAT, NAT].map(ok))
                .is_err()
        );
        assert!(
            SPN::IfThen()
                .check([BOOL, NAT].map(ok), [].map(ok))
                .is_err()
        );
    }

    #[test]
    fn skip_is_id() {
        assert!(matches!(<SPN as Sequential>::skip(&NAT), SPN::Id()));
    }

    #[test]
    fn nondet_ok_on_every_value_sort() {
        for s in [NAT, BOOL, EVENT, CLOCK] {
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
    fn nondet_on_the_trivial_tangent_fails() {
        // there is nothing to choose in a sort inhabited by zero alone
        assert!(SPN::Nondet(ZERO).check([].map(ok), [ZERO].map(ok)).is_err());
    }

    #[test]
    fn nondet_on_a_clock_tangent_ok() {
        // a clock of unknown speed: the rate is what is chosen
        assert!(
            SPN::Nondet(DCLOCK)
                .check([].map(ok), [DCLOCK].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn havoc_on_the_trivial_tangent_is_zero() {
        // a singleton sort leaves havoc no choice
        assert!(matches!(<SPN as Combinatorial>::havoc(&ZERO), SPN::Zero()));
        // a clock tangent is not a singleton: its rate is up for choosing
        assert!(matches!(
            <SPN as Combinatorial>::havoc(&DCLOCK),
            SPN::Nondet(DCLOCK)
        ));
    }

    #[test]
    fn zero_resolves_per_fragment() {
        // `Differential::zero` is given the tangent sort it writes
        assert!(matches!(
            <SPN as Differential>::zero(&CLOCK.T()),
            SPN::ClkZero()
        ));
        for s in [NAT, BOOL, EVENT] {
            assert!(matches!(<SPN as Differential>::zero(&s.T()), SPN::Zero()));
        }
    }

    #[test]
    fn clk_zerograd_writes_a_clock_tangent() {
        assert!(SPN::ClkZero().check([].map(ok), [DCLOCK].map(ok)).is_ok());
        assert!(
            SPN::ClkZero()
                .check([].map(ok), [Sort::Clock { rank: 2 }].map(ok))
                .is_ok()
        );
        // never a value, and never the trivial tangent
        assert!(SPN::ClkZero().check([].map(ok), [CLOCK].map(ok)).is_err());
        assert!(SPN::ClkZero().check([].map(ok), [ZERO].map(ok)).is_err());
    }

    #[test]
    fn clk_mul_scales_a_tangent() {
        assert!(
            SPN::ClkMul(-1.0)
                .check([DCLOCK].map(ok), [DCLOCK].map(ok))
                .is_ok()
        );
        assert!(
            SPN::ClkMul(2.0)
                .check(
                    [Sort::Clock { rank: 2 }].map(ok),
                    [Sort::Clock { rank: 2 }].map(ok)
                )
                .is_ok()
        );
    }

    #[test]
    fn clk_mul_wrong_sorts_and_arity_fail() {
        // a value has no rate to scale, and the result keeps the rank of the operand
        assert!(
            SPN::ClkMul(-1.0)
                .check([CLOCK].map(ok), [DCLOCK].map(ok))
                .is_err()
        );
        assert!(
            SPN::ClkMul(-1.0)
                .check([DCLOCK].map(ok), [CLOCK].map(ok))
                .is_err()
        );
        assert!(
            SPN::ClkMul(-1.0)
                .check([DCLOCK].map(ok), [Sort::Clock { rank: 2 }].map(ok))
                .is_err()
        );
        assert!(
            SPN::ClkMul(-1.0)
                .check([NAT].map(ok), [DCLOCK].map(ok))
                .is_err()
        );
        assert!(
            SPN::ClkMul(-1.0)
                .check([].map(ok), [DCLOCK].map(ok))
                .is_err()
        );
        assert!(
            SPN::ClkMul(-1.0)
                .check([DCLOCK, DCLOCK].map(ok), [DCLOCK].map(ok))
                .is_err()
        );
        assert!(
            SPN::ClkMul(-1.0)
                .check([DCLOCK].map(ok), [].map(ok))
                .is_err()
        );
    }

    #[test]
    fn clk_rate_writes_a_clock_tangent() {
        assert!(
            SPN::ClkRate(-1.0)
                .check([].map(ok), [DCLOCK].map(ok))
                .is_ok()
        );
        assert!(
            SPN::ClkRate(-1.0)
                .check([].map(ok), [Sort::Clock { rank: 2 }].map(ok))
                .is_ok()
        );
        // a rate is not a value, and the constant sorts have no rate at all
        assert!(
            SPN::ClkRate(-1.0)
                .check([].map(ok), [CLOCK].map(ok))
                .is_err()
        );
        assert!(
            SPN::ClkRate(-1.0)
                .check([].map(ok), [ZERO].map(ok))
                .is_err()
        );
        assert!(SPN::ClkRate(-1.0).check([].map(ok), [NAT].map(ok)).is_err());
    }

    #[test]
    fn clk_rate_with_read_fails() {
        assert!(
            SPN::ClkRate(-1.0)
                .check([DCLOCK].map(ok), [DCLOCK].map(ok))
                .is_err()
        );
    }

    #[test]
    fn clk_rate_arity_fails() {
        assert!(SPN::ClkRate(-1.0).check([].map(ok), [].map(ok)).is_err());
        assert!(
            SPN::ClkRate(-1.0)
                .check([].map(ok), [DCLOCK, DCLOCK].map(ok))
                .is_err()
        );
    }

    #[test]
    fn ite_selects_a_clock_flow() {
        // preemptive resume: der(clk) = if enabled then -1 else 0, with both
        // branches written by the flow generators
        assert!(
            SPN::Ite()
                .check([BOOL, DCLOCK, DCLOCK].map(ok), [DCLOCK].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn ifthen_guards_a_transitions_update() {
        // `X(p) := if fires then Dec(p)`: the marking is written by the steps
        // that fire, and the guard -- an `Event`, as a firing tends to be -- is
        // what says which those are
        assert!(
            SPN::IfThen()
                .check([EVENT, NAT].map(ok), [NAT].map(ok))
                .is_ok()
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
            SPN::ClkZero()
                .check([DCLOCK].map(ok), [DCLOCK].map(ok))
                .is_err()
        );
    }

    #[test]
    fn zero_arity_mismatch_fails() {
        assert!(SPN::Zero().check([].map(ok), [].map(ok)).is_err());
        assert!(SPN::Zero().check([].map(ok), [ZERO, ZERO].map(ok)).is_err());
        assert!(SPN::ClkZero().check([].map(ok), [].map(ok)).is_err());
        assert!(
            SPN::ClkZero()
                .check([].map(ok), [DCLOCK, DCLOCK].map(ok))
                .is_err()
        );
    }

    #[test]
    fn exp_ok() {
        assert!(SPN::Exp(2.5).check([].map(ok), [CLOCK].map(ok)).is_ok());
    }

    #[test]
    fn exp_non_clock_write_fails() {
        assert!(SPN::Exp(2.5).check([].map(ok), [NAT].map(ok)).is_err());
        // arming a clock produces a value, not a rate
        assert!(SPN::Exp(2.5).check([].map(ok), [DCLOCK].map(ok)).is_err());
    }

    #[test]
    fn exp_with_read_fails() {
        assert!(
            SPN::Exp(2.5)
                .check([CLOCK].map(ok), [CLOCK].map(ok))
                .is_err()
        );
    }

    #[test]
    fn exp_two_writes_fails() {
        assert!(
            SPN::Exp(2.5)
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
