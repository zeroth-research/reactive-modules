/*!
# Linear real arithmetic

Defines the theory [`LRA`] of linear real arithmetic over matrices,
mixing real and boolean matrices in a single signature.

A [`Sort`] value is `Real(shape)`, `DeltaReal { shape, order }`, `Bool(shape)`,
or `Zero`. Values and their derivatives are different sorts: `DeltaReal` is the
`order`-th derivative of a real matrix, and the sort former [`Tangent`] maps
`Real` to `DeltaReal` and raises the order of a `DeltaReal`. Booleans are
constant sorts: their tangent is [`Sort::Zero`], the inhabited singleton whose
only writer is the `zero` generator. The linear operations (`Add`, `Sub`,
`Linear`, `Transpose`, `Ite`, `Id`) act on values and derivatives alike; the
others on values only. The operations in [`LRA`] are:

- [`LRA::Real`], [`LRA::Bool`] — real and boolean matrix literals; the
  tensor's element kind must match the variant.
- [`LRA::And`], [`LRA::Or`], [`LRA::Xor`], [`LRA::Not`]
  — boolean operations on the boolean fragment.
- [`LRA::Le`], [`LRA::Lt`], [`LRA::Ge`], [`LRA::Gt`], [`LRA::Eq`], [`LRA::Ne`]
  — pointwise real comparisons producing a boolean of the same shape.
- [`LRA::Ite`] — if-then-else: reads a boolean guard and two same-typed branches.
- [`LRA::Linear`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and
  `B` constant real matrices of compatible shapes.
- [`LRA::ReLU`] — the shape-preserving rectified-linear map on real matrices.

`LRA` implements [`Signature`]; [`Signature::check`] validates read/write
sorts against the selected operation.

## Examples

```
use theory::Signature;
use theory::lra::{LRA, Sort};

let ok = Ok::<_, String>;

// Pointwise less-than on scalars: Real(1,1), Real(1,1) -> Bool(1,1).
let i = Sort::real([1, 1]);
let b = Sort::Bool([1, 1]);
assert!(LRA::Lt().check([i, i].map(ok), [b].map(ok)).is_ok());

// ReLU preserves shape and stays in the real fragment.
let m = Sort::real([3, 4]);
assert!(LRA::ReLU().check([m].map(ok), [m].map(ok)).is_ok());
assert!(LRA::ReLU().check([b].map(ok), [b].map(ok)).is_err());
```
*/

use crate::*;
#[cfg(feature = "pyo3")]
use pyo3::pyclass;
use std::fmt;
use std::num::NonZeroU8;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Sort {
    /// A real tensor: a value.
    Real([usize; 2]),
    /// The `order`-th derivative of a real tensor of the given shape.
    DeltaReal { shape: [usize; 2], order: NonZeroU8 },
    /// A boolean tensor: a constant sort — it cannot move during delay.
    Bool([usize; 2]),
    /// The trivial tangent: a singleton, inhabited by exactly the zero
    /// value. Terminal, not empty.
    Zero,
}

impl Sort {
    /// A real value sort.
    pub fn real(shape: [usize; 2]) -> Self {
        Sort::Real(shape)
    }

    /// The `order`-th derivative of a real value (the value itself for
    /// `order = 0`): the sorts the linear operations act on.
    pub fn graded(shape: [usize; 2], order: u8) -> Self {
        match NonZeroU8::new(order) {
            None => Sort::Real(shape),
            Some(order) => Sort::DeltaReal { shape, order },
        }
    }

    /// The shape and derivative order of a real value (order 0) or
    /// derivative; `None` for the other sorts.
    pub fn as_graded(&self) -> Option<([usize; 2], u8)> {
        match *self {
            Sort::Real(shape) => Some((shape, 0)),
            Sort::DeltaReal { shape, order } => Some((shape, order.get())),
            Sort::Bool(_) | Sort::Zero => None,
        }
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, Sort::Bool(..))
    }

    /// Whether this is a real value (not a derivative).
    pub fn is_real(&self) -> bool {
        matches!(self, Sort::Real(..))
    }

    pub fn shape(&self) -> Option<&[usize; 2]> {
        match self {
            Sort::Bool(shape) | Sort::Real(shape) | Sort::DeltaReal { shape, .. } => Some(shape),
            Sort::Zero => None,
        }
    }
}

impl Tangent for Sort {
    #[allow(non_snake_case)]
    fn T(&self) -> Self {
        match *self {
            // the carrier (shape) is unchanged, the order goes up
            Sort::Real(shape) => Sort::DeltaReal {
                shape,
                order: NonZeroU8::MIN,
            },
            Sort::DeltaReal { shape, order } => Sort::DeltaReal {
                shape,
                order: order.checked_add(1).expect("derivative order overflows u8"),
            },
            // constant sorts have the trivial tangent
            Sort::Bool(_) => Sort::Zero,
            // the tangent tower stabilizes at the first step
            Sort::Zero => Sort::Zero,
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Real([i, j]) => write!(f, "Real({i}, {j})"),
            Sort::DeltaReal {
                shape: [i, j],
                order,
            } => write!(f, "T{order} Real({i}, {j})"),
            Sort::Bool([i, j]) => write!(f, "Bool({i}, {j})"),
            Sort::Zero => write!(f, "Zero"),
        }
    }
}

#[derive(Clone, Debug, strum::Display)]
#[cfg_attr(feature = "pyo3", pyclass(frozen))]
pub enum LRA {
    // constant matrix literals; the variant is the sort (Real or Bool)
    #[strum(to_string = "{0}")]
    Real(crate::PyTensor),
    #[strum(to_string = "{0}")]
    Bool(crate::PyTensor),
    // boolean operations
    And(),
    Or(),
    Xor(),
    Not(),
    // real comparisons
    Le(),
    Lt(),
    Ge(),
    Gt(),
    Eq(),
    Ne(),
    // linear / matrix operations
    // A*x + B where `A` and `B` are constants
    #[strum(to_string = "Linear({0}, {1})")]
    Linear(crate::PyTensor, crate::PyTensor),
    Add(),
    Sub(),
    // XXX: should these be in RLA?
    ReLU(),
    Argmax(),
    Min(),
    Max(),
    // matrix operations
    Transpose(),
    // control flow
    Ite(),
    Id(),
    #[strum(to_string = "Uninterpreted({0})")]
    Uninterpreted(String),
    /// The unique inhabitant of the `Zero` sort: the only generator
    /// writing a `Zero` wire (the trivial tangent of the constant sorts).
    Zero(),
    RealZerograd([usize; 2]), // unstable
    AnyBool([usize; 2]),
    AnyReal([usize; 2]),
}

impl Sequential for LRA {
    fn skip(_range: &Self::Sort) -> Self {
        LRA::Id()
    }
}

impl Combinatorial for LRA {
    fn havoc(range: &Self::Sort) -> Self {
        match range {
            Sort::Bool(shape) => LRA::AnyBool(*shape),
            Sort::Real(shape) | Sort::DeltaReal { shape, .. } => LRA::AnyReal(*shape),
            // havoc over a singleton is the singleton
            Sort::Zero => LRA::Zero(),
        }
    }
}

impl Differential for LRA {
    fn zero(range: &Self::Sort) -> Self {
        // `range` is the tangent sort the generator writes
        match range {
            Sort::Real(shape) | Sort::DeltaReal { shape, .. } => LRA::RealZerograd(*shape),
            Sort::Bool(_) | Sort::Zero => LRA::Zero(),
        }
    }
}

impl Signature for LRA {
    type Sort = Sort;
    const NAME: &'static str = "LRA";

    fn check<R, W, E: fmt::Display>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<Sort, E>>,
        W: IntoIterator<Item = Result<Sort, E>>,
    {
        match self {
            LRA::Real(cm) => check_const(cm, false, read, write),
            LRA::Bool(cm) => check_const(cm, true, read, write),
            LRA::Zero() => check_zero(&Sort::Zero, read, write),
            LRA::RealZerograd(shape) => check_real_source("ZERO", shape, 1, read, write),
            LRA::AnyBool(shape) => check_havoc(&Sort::Bool(*shape), read, write),
            LRA::AnyReal(shape) => check_real_source("HAVOC", shape, 0, read, write),
            LRA::And() | LRA::Or() | LRA::Xor() | LRA::Not() => check_bool(self, read, write),
            LRA::Le() | LRA::Lt() | LRA::Ge() | LRA::Gt() | LRA::Eq() | LRA::Ne() => {
                check_cmp(self, read, write)
            }
            LRA::Linear(a, b) => {
                let mut read = read.into_iter();
                let mut write = write.into_iter();
                check_linear_affine(self, a, b, &mut read, &mut write)
            }
            LRA::Add() | LRA::Sub() | LRA::ReLU() | LRA::Argmax() | LRA::Min() | LRA::Max() => {
                check_mat_ops(self, read, write)
            }
            LRA::Transpose() => check_transpose(self, read, write),
            LRA::Ite() | LRA::Id() => check_flow(self, read, write),
            LRA::Uninterpreted(_) => {
                let mut read = read.into_iter();
                let mut write = write.into_iter();
                // uninterpreted has either one read or one write
                if read.next().is_some() {
                    if read.next().is_some() {
                        return Err(format!("{:?}: expected exactly one read, got more", self));
                    }
                    if write.next().is_some() {
                        return Err(format!(
                            "{:?}: expected exactly one read, got also write",
                            self
                        ));
                    }
                    return Ok(());
                }
                if write.next().is_some() {
                    if write.next().is_some() {
                        return Err(format!("{:?}: expected exactly one write, got more", self));
                    }
                    if read.next().is_some() {
                        return Err(format!(
                            "{:?}: expected exactly one write, got also read",
                            self
                        ));
                    }
                    return Ok(());
                }
                Err(format!(
                    "{:?}: expected exactly one write or one read, got none",
                    self
                ))
            }
        }
    }
}

// `bool_lit` is whether the literal is the `Bool` variant: it writes a `Bool`
// wire, the other variant a `Real` one.
fn check_const<R, W, E: fmt::Display>(
    cm: &crate::PyTensor,
    bool_lit: bool,
    read: R,
    write: W,
) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    if read.next().is_some() {
        return Err("Const: cannot read values".into());
    }
    // the write wire must have the sort of the variant, and the tensor its kind
    let [i, j] = match next_sort(&mut write, 0)? {
        Sort::DeltaReal { .. } => {
            return Err("Cannot derive a real. Use ZERO to apply a no change".to_string());
        }
        Sort::Real(shape) => {
            if bool_lit {
                return Err("Const: a Bool literal cannot write a Real wire".into());
            }
            if cm.is_bool() {
                return Err("Const: write wire is Real but initializer is a boolean tensor".into());
            }
            shape
        }
        Sort::Bool([i, j]) => {
            if !bool_lit {
                return Err("Const: a Real literal cannot write a Bool wire".into());
            }
            if !cm.is_bool() {
                return Err(
                    "Const: write wire is Bool but initializer is not a boolean tensor".into(),
                );
            }
            [i, j]
        }
        Sort::Zero => {
            return Err("Const: cannot write a Zero wire. Use ZERO to apply a no change".into());
        }
    };
    let size = cm.size();
    if size.len() != 2 {
        return Err(format!(
            "Const: initializer must be a 2D tensor, got {}D",
            size.len()
        ));
    }
    if size[0] as usize != i {
        return Err(format!(
            "Const: initializer has wrong number of rows (has {}, expected {})",
            size[0], i
        ));
    }
    if size[1] as usize != j {
        return Err(format!("Const: some row has wrong length, expected {}", j));
    }
    if write.next().is_some() {
        return Err("Const: returns more than one value".into());
    }
    Ok(())
}

// ZERO and HAVOC on the real fragment: write exactly one real wire of the
// declared shape and of derivative order at least `min_order`, and read
// nothing. ZERO writes derivatives (order at least 1); HAVOC values or
// derivatives, so that
// `havoc(range)` writes `range` for every real sort.
fn check_real_source<R, W, E: fmt::Display>(
    name: &str,
    shape: &[usize; 2],
    min_order: u8,
    read: R,
    write: W,
) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    if read.into_iter().next().is_some() {
        return Err(format!("{name} expects no read wires"));
    }
    let mut write = write.into_iter();
    match write.next() {
        Some(Ok(sort))
            if sort
                .as_graded()
                .is_some_and(|(s, order)| s == *shape && order >= min_order) => {}
        Some(Ok(sort)) => {
            return Err(format!(
                "{name} expects write of a real of shape {:?} and derivative order at least {min_order}, got {}",
                shape, sort
            ));
        }
        Some(Err(e)) => return Err(e.to_string()),
        None => return Err(format!("{name} expects exactly one write wire, got none")),
    }
    if write.next().is_some() {
        return Err(format!("{name} expects exactly one write wire, got more"));
    }
    Ok(())
}

fn check_bool<R, W, E: fmt::Display>(op: &LRA, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        LRA::Not() => {
            let (r, w) = (next_sort(&mut read, 0)?, next_sort(&mut write, 0)?);
            if !matches!(r, Sort::Bool(..)) {
                return Err(format!("{:?}: input must be Bool", op));
            }
            if r != w {
                return Err(format!("{:?}: input and output type must be the same", op));
            }
            if read.next().is_some() {
                return Err(format!("{:?}: must read a single value (reads more)", op));
            }
            if write.next().is_some() {
                return Err(format!("{:?}: must write a single value (writes more)", op));
            }
            Ok(())
        }
        LRA::And() | LRA::Or() | LRA::Xor() => {
            let w1 = next_sort(&mut write, 0)?;
            let (r1, r2, None) = (
                next_sort(&mut read, 0)?,
                next_sort(&mut read, 1)?,
                read.next(),
            ) else {
                return Err(format!("{:?}: must read exactly two values", op));
            };
            if !matches!(w1, Sort::Bool(..)) {
                return Err(format!("{:?}: output must be Bool", op));
            }
            if r1 != r2 {
                return Err(format!("{:?}: input values must have the same type", op));
            }
            if w1 != r1 {
                return Err(format!(
                    "{:?}: input and output values must have the same type",
                    op
                ));
            }
            Ok(())
        }
        _ => unreachable!(),
    }
}

fn check_cmp<R, W, E: fmt::Display>(op: &LRA, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let (r1, r2, None) = (
        next_sort(&mut read, 0)?,
        next_sort(&mut read, 1)?,
        read.next(),
    ) else {
        return Err(format!("{:?}: must read exactly two values", op));
    };
    if r1 != r2 {
        return Err(format!("{:?}: input values must have the same type", op));
    }
    // comparisons are not linear: they compare values, not derivatives
    let shape = match r1 {
        Sort::Real(shape) => shape,
        _ => {
            return Err(format!("{:?}: inputs must be real values, got {}", op, r1));
        }
    };
    let (w1, None) = (next_sort(&mut write, 0)?, write.next()) else {
        return Err(format!("{:?}: must write exactly one value", op));
    };
    if w1 != Sort::Bool(shape) {
        return Err(format!(
            "{:?}: output must be Bool({:?}), got {w1}",
            op, shape
        ));
    }
    Ok(())
}

fn check_mat_ops<R, W, E: fmt::Display>(op: &LRA, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        // element-wise, as `minimum`/`maximum` in the evaluator and the SMT encoding
        LRA::Add() | LRA::Sub() | LRA::Min() | LRA::Max() => {
            let (r1, r2, None) = (
                next_sort(&mut read, 0)?,
                next_sort(&mut read, 1)?,
                read.next(),
            ) else {
                return Err(format!("{:?}: must read exactly two values", op));
            };
            let (w1, None) = (next_sort(&mut write, 0)?, write.next()) else {
                return Err(format!("{:?}: must write exactly one value", op));
            };

            if r1 != r2 {
                return Err(format!("{:?}: inputs must have the same type", op));
            }
            if r1 != w1 {
                return Err(format!(
                    "{:?}: input and output must have the same type",
                    op
                ));
            }
            // `Add` and `Sub` are linear, so they also act on derivatives
            let linear = matches!(op, LRA::Add() | LRA::Sub());
            if !(w1.is_real() || linear && w1.as_graded().is_some()) {
                return Err(format!(
                    "{:?}: input and output values must be real {}",
                    op,
                    if linear { "matrices" } else { "values" }
                ));
            }
            Ok(())
        }
        LRA::ReLU() => {
            let (r1, None) = (next_sort(&mut read, 0)?, read.next()) else {
                return Err(format!("{:?}: must read exactly one value", op));
            };
            let (w1, None) = (next_sort(&mut write, 0)?, write.next()) else {
                return Err(format!("{:?}: must write exactly one value", op));
            };
            if r1 != w1 {
                return Err(format!(
                    "{:?}: input and output must have the same type",
                    op
                ));
            }
            if !w1.is_real() {
                return Err(format!("{:?}: input and output must be real values", op));
            }
            Ok(())
        }
        LRA::Argmax() => {
            let (r1, None) = (next_sort(&mut read, 0)?, read.next()) else {
                return Err(format!("{:?}: must read exactly one value", op));
            };
            if !r1.is_real() {
                return Err(format!("{:?}: input must be a real value, got {r1}", op));
            }
            let (w1, None) = (next_sort(&mut write, 0)?, write.next()) else {
                return Err(format!("{:?}: must write exactly one value", op));
            };
            // the index of the maximum in the flattened matrix, as in the evaluator
            if w1 != Sort::real([1, 1]) {
                return Err(format!(
                    "{:?}: output must be the index Real(1, 1), got {w1}",
                    op
                ));
            }
            Ok(())
        }
        _ => unreachable!(),
    }
}

fn check_linear_affine<R, W, E: fmt::Display>(
    op: &LRA,
    a: &crate::PyTensor,
    b: &crate::PyTensor,
    read: &mut R,
    write: &mut W,
) -> Result<(), String>
where
    R: Iterator<Item = Result<Sort, E>>,
    W: Iterator<Item = Result<Sort, E>>,
{
    let (r1, None) = (next_sort(read, 0)?, read.next()) else {
        return Err(format!("{:?}: must read exactly one value", op));
    };
    let (w1, None) = (next_sort(write, 0)?, write.next()) else {
        return Err(format!("{:?}: must write exactly one value", op));
    };

    // Convention: Y = A·X + B  where X=[in,batch], A=[out,in], B=[out,1], Y=[out,batch].
    let a_size = a.size();
    if a_size.len() != 2 {
        return Err(format!("{:?}: `A` must be a 2D tensor", op));
    }
    let a_rows = a_size[0] as usize; // out_features
    let a_cols = a_size[1] as usize; // in_features
    if a_rows == 0 {
        return Err(format!("{:?}: `A` is empty", op));
    }

    let b_size = b.size();
    let (b_rows, b_cols) = if b.numel() == 0 {
        (0usize, 0usize)
    } else {
        if b_size.len() != 2 {
            return Err(format!("{:?}: `B` must be a 2D tensor", op));
        }
        (b_size[0] as usize, b_size[1] as usize)
    };

    // linear, so it acts on derivatives too, keeping their order
    match (r1.as_graded(), w1.as_graded()) {
        (Some(([d1, d2], order0)), Some(([d3, d4], order1))) => {
            // X has shape [d1=in, d2=batch]; A has shape [a_rows=out, a_cols=in].
            if d1 != a_cols {
                return Err(format!(
                    "{:?}: dimension mismatch: X has {}x{} but A has {}x{} (need X.rows == A.cols)",
                    op, d1, d2, a_rows, a_cols
                ));
            }
            // B must be a column vector [out, 1] matching the output rows.
            if b_rows > 0 && (b_rows != a_rows || b_cols != 1) {
                return Err(format!(
                    "{:?}: B must be a column vector [{}x1], got {}x{}",
                    op, a_rows, b_rows, b_cols
                ));
            }
            // Output Y = A·X has shape [a_rows=out, d2=batch].
            if a_rows != d3 || d2 != d4 {
                return Err(format!(
                    "{:?}: bad output matrix dimensions, expected {}x{} but got {}x{}",
                    op, a_rows, d2, d3, d4
                ));
            }
            if order0 != order1 {
                return Err("Derivative order mismatch".to_string());
            }
            Ok(())
        }
        _ => Err(format!("{:?}: input and output must be real matrices", op)),
    }
}

fn check_transpose<R, W, E: fmt::Display>(op: &LRA, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let (r1, None) = (next_sort(&mut read, 0)?, read.next()) else {
        return Err(format!("{:?}: must read exactly one value", op));
    };
    let (w1, None) = (next_sort(&mut write, 0)?, write.next()) else {
        return Err(format!("{:?}: must write exactly one value", op));
    };
    // linear, so it acts on derivatives too, keeping their order
    match (r1.as_graded(), w1.as_graded()) {
        (Some(([d1, d2], order0)), Some(([e1, e2], order1))) => {
            if d2 != e1 || d1 != e2 {
                return Err(format!(
                    "{:?}: transpose of {}x{} must produce {}x{}, got {}x{}",
                    op, d1, d2, d2, d1, e1, e2
                ));
            }
            if order0 != order1 {
                return Err("Derivative order mismatch".to_string());
            }
            Ok(())
        }
        _ => Err(format!("{:?}: input and output must be real matrices", op)),
    }
}

fn check_flow<R, W, E: fmt::Display>(op: &LRA, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<Sort, E>>,
    W: IntoIterator<Item = Result<Sort, E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        LRA::Id() => {
            let (r1, None) = (next_sort(&mut read, 0)?, read.next()) else {
                return Err(format!("{:?}: must read exactly one value", op));
            };
            let (w1, None) = (next_sort(&mut write, 0)?, write.next()) else {
                return Err(format!("{:?}: must write exactly one value", op));
            };
            if r1 != w1 {
                return Err(format!(
                    "{:?}: input and output must have the same type",
                    op
                ));
            }
            Ok(())
        }
        LRA::Ite() => {
            let (r1, r2, r3, None) = (
                next_sort(&mut read, 0)?,
                next_sort(&mut read, 1)?,
                next_sort(&mut read, 2)?,
                read.next(),
            ) else {
                return Err(format!("{:?}: must read exactly three values", op));
            };
            let (w1, None) = (next_sort(&mut write, 0)?, write.next()) else {
                return Err(format!("{:?}: must write exactly one value", op));
            };
            if r2 != r3 {
                return Err(format!(
                    "{:?}: 2nd and 3rd inputs must have the same type",
                    op
                ));
            }
            if w1 != r2 {
                return Err(format!(
                    "{:?}: inputs and output must have the same type",
                    op
                ));
            }
            if r1 != Sort::Bool([1, 1]) {
                return Err(format!(
                    "{:?}: input and output values must have the same type",
                    op
                ));
            }
            Ok(())
        }
        _ => unreachable!(),
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    fn real(r: usize, c: usize) -> Sort {
        Sort::real([r, c])
    }

    /// A real tangent (first derivative).
    fn dreal(r: usize, c: usize) -> Sort {
        Sort::graded([r, c], 1)
    }

    fn bool_t(r: usize, c: usize) -> Sort {
        Sort::Bool([r, c])
    }

    fn ok(s: Sort) -> Result<Sort, String> {
        Ok(s)
    }

    #[test]
    fn type_kind_and_shape() {
        assert!(real(2, 3).is_real() && !real(2, 3).is_bool());
        assert_eq!(real(2, 3).shape(), Some(&[2, 3]));
        // a derivative is a real matrix, but not a value
        assert!(!dreal(2, 3).is_real());
        assert_eq!(dreal(2, 3).shape(), Some(&[2, 3]));
        assert_eq!(dreal(2, 3).as_graded(), Some(([2, 3], 1)));
        assert!(bool_t(1, 1).is_bool() && !bool_t(1, 1).is_real());
        assert_eq!(Sort::Zero.shape(), None);
    }

    #[test]
    fn tangent_grades_reals_and_collapses_bools() {
        // Real -> DeltaReal, and one order up: the carrier is unchanged
        assert_eq!(real(2, 3).T(), dreal(2, 3));
        assert_eq!(dreal(2, 3).T(), Sort::graded([2, 3], 2));
        // constant sorts have the trivial tangent, which is a fixed point
        assert_eq!(bool_t(1, 1).T(), Sort::Zero);
        assert_eq!(Sort::Zero.T(), Sort::Zero);
    }

    #[test]
    fn const_real_ok() {
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0.0f64, 1.0], [2.0, 3.0]]).into();
        assert!(
            LRA::Real(cm)
                .check([].map(ok), [real(2, 2)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn const_real_covector_write_fails() {
        // a real constant writes a value, never a derivative:
        // a tangent write wire must be rejected
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0.0f64]]).into();
        assert!(
            LRA::Real(cm)
                .check([].map(ok), [dreal(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_zero_write_fails() {
        // the only writer of a Zero wire is the zero generator
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[true]]).into();
        assert!(
            LRA::Real(cm)
                .check([].map(ok), [Sort::Zero].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_real_bool_write_fails() {
        assert!(
            LRA::Real(tch::Tensor::from_slice2(&[[0.0f64]]).into())
                .check([].map(ok), [bool_t(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_real_wrong_rows_fails() {
        assert!(
            LRA::Real(tch::Tensor::from_slice2(&[[0.0f64]]).into())
                .check([].map(ok), [real(2, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_real_with_read_fails() {
        let t = real(1, 1);
        assert!(
            LRA::Real(tch::Tensor::from_slice2(&[[0.0f64]]).into())
                .check([t].map(ok), [t].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_bool_ok() {
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[true, false], [false, true]]).into();
        assert!(
            LRA::Bool(cm)
                .check([].map(ok), [bool_t(2, 2)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn const_variant_decides_the_sort() {
        // the variant, not the tensor, decides the sort of the literal
        let b = || -> crate::PyTensor { tch::Tensor::from_slice2(&[[true]]).into() };
        assert!(
            LRA::Real(b())
                .check([].map(ok), [bool_t(1, 1)].map(ok))
                .is_err()
        );
        assert!(
            LRA::Bool(b())
                .check([].map(ok), [real(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_bool_real_write_fails() {
        assert!(
            LRA::Real(tch::Tensor::from_slice2(&[[true]]).into())
                .check([].map(ok), [real(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn not_ok() {
        let b = bool_t(2, 3);
        assert!(LRA::Not().check([b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn not_real_input_fails() {
        let t = real(1, 1);
        assert!(LRA::Not().check([t].map(ok), [t].map(ok)).is_err());
    }

    #[test]
    fn and_ok() {
        let b = bool_t(2, 2);
        assert!(LRA::And().check([b, b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn or_ok() {
        let b = bool_t(1, 1);
        assert!(LRA::Or().check([b, b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn xor_ok() {
        let b = bool_t(3, 1);
        assert!(LRA::Xor().check([b, b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn and_real_output_fails() {
        let b = bool_t(1, 1);
        assert!(
            LRA::And()
                .check([b, b].map(ok), [real(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn and_type_mismatch_fails() {
        assert!(
            LRA::And()
                .check([bool_t(1, 1), bool_t(1, 2)].map(ok), [bool_t(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn lt_ok() {
        assert!(
            LRA::Lt()
                .check([real(1, 1), real(1, 1)].map(ok), [bool_t(1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn le_ok() {
        assert!(
            LRA::Le()
                .check([real(2, 3), real(2, 3)].map(ok), [bool_t(2, 3)].map(ok))
                .is_ok()
        );

        assert!(
            LRA::Le()
                .check([real(3, 3), real(2, 3)].map(ok), [bool_t(2, 3)].map(ok))
                .is_err()
        );

        assert!(
            LRA::Le()
                .check([real(2, 3), real(2, 3)].map(ok), [bool_t(3, 3)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn eq_ok() {
        assert!(
            LRA::Eq()
                .check([real(2, 2), real(2, 2)].map(ok), [bool_t(2, 2)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn cmp_non_bool_output_fails() {
        let t = real(1, 1);
        assert!(LRA::Lt().check([t, t].map(ok), [t].map(ok)).is_err());
    }

    #[test]
    fn non_linear_ops_reject_derivatives() {
        // comparisons and the non-linear operations take values only
        let (d, b) = (dreal(1, 1), bool_t(1, 1));
        assert!(LRA::Lt().check([d, d].map(ok), [b].map(ok)).is_err());
        assert!(LRA::Eq().check([d, d].map(ok), [b].map(ok)).is_err());
        assert!(LRA::ReLU().check([d].map(ok), [d].map(ok)).is_err());
        assert!(LRA::Min().check([d, d].map(ok), [d].map(ok)).is_err());
        assert!(LRA::Max().check([d, d].map(ok), [d].map(ok)).is_err());
        assert!(
            LRA::Argmax()
                .check([d].map(ok), [real(1, 1)].map(ok))
                .is_err()
        );
        // the linear ones act on derivatives
        assert!(LRA::Add().check([d, d].map(ok), [d].map(ok)).is_ok());
        assert!(LRA::Sub().check([d, d].map(ok), [d].map(ok)).is_ok());
    }

    #[test]
    fn cmp_surplus_wires_fail() {
        let (t, b) = (real(1, 1), bool_t(1, 1));
        assert!(LRA::Lt().check([t, t, t].map(ok), [b].map(ok)).is_err());
        assert!(LRA::Lt().check([t, t].map(ok), [b, b].map(ok)).is_err());
    }

    #[test]
    fn cmp_input_mismatch_fails() {
        assert!(
            LRA::Eq()
                .check([real(1, 1), real(1, 2)].map(ok), [bool_t(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn add_ok() {
        let t = real(3, 4);
        assert!(LRA::Add().check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn add_tangents_ok() {
        // bundle addition: Add acts on derivatives too
        let dt = dreal(3, 4);
        assert!(LRA::Add().check([dt, dt].map(ok), [dt].map(ok)).is_ok());
        // but never across orders
        assert!(
            LRA::Add()
                .check([dt, real(3, 4)].map(ok), [dt].map(ok))
                .is_err()
        );
    }

    #[test]
    fn add_shape_mismatch_fails() {
        assert!(
            LRA::Add()
                .check([real(1, 2), real(2, 1)].map(ok), [real(1, 2)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn add_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LRA::Add().check([b, b].map(ok), [b].map(ok)).is_err());
    }

    #[test]
    fn relu_ok() {
        let t = real(3, 4);
        assert!(LRA::ReLU().check([t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn relu_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LRA::ReLU().check([b].map(ok), [b].map(ok)).is_err());
    }

    #[test]
    fn argmax_ok() {
        assert!(
            LRA::Argmax()
                .check([real(3, 4)].map(ok), [real(1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn argmax_matrix_output_fails() {
        assert!(
            LRA::Argmax()
                .check([real(3, 4)].map(ok), [real(3, 4)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn argmax_vector_output_fails() {
        // the output is a single index, not a vector of indices
        assert!(
            LRA::Argmax()
                .check([real(3, 4)].map(ok), [real(1, 4)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn argmax_bool_input_fails() {
        let w = real(1, 1);
        assert!(
            LRA::Argmax()
                .check([bool_t(3, 4)].map(ok), [w].map(ok))
                .is_err()
        );
    }

    #[test]
    fn min_max_ok() {
        // element-wise, like `Add`
        let t = real(4, 1);
        assert!(LRA::Min().check([t, t].map(ok), [t].map(ok)).is_ok());
        assert!(LRA::Max().check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn min_max_fail_as_reductions() {
        let (t, b) = (real(4, 1), bool_t(4, 1));
        assert!(LRA::Min().check([t].map(ok), [real(1, 1)].map(ok)).is_err());
        assert!(
            LRA::Max()
                .check([t, t].map(ok), [real(1, 1)].map(ok))
                .is_err()
        );
        assert!(LRA::Min().check([b, b].map(ok), [b].map(ok)).is_err());
    }

    #[test]
    fn linear_ok() {
        // A=[2,3] maps 3 features to 2; X=[3,4] is 3 features × 4 batch items.
        // Convention: Y = A·X  →  Y=[2,4].
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Double, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([0, 0], (tch::Kind::Double, tch::Device::Cpu)).into();
        assert!(
            LRA::Linear(a, b)
                .check([real(3, 4)].map(ok), [real(2, 4)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn linear_order_mismatch_fails() {
        // a linear map applies to derivatives too, but never across orders
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Double, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([0, 0], (tch::Kind::Double, tch::Device::Cpu)).into();
        let lin = LRA::Linear(a, b);
        assert!(
            lin.check([dreal(3, 4)].map(ok), [dreal(2, 4)].map(ok))
                .is_ok()
        );
        assert!(
            lin.check([dreal(3, 4)].map(ok), [real(2, 4)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn linear_with_bias_ok() {
        // A=[2,3], b=[2,1] column bias, X=[3,1] single sample → Y=[2,1].
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Double, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([2, 1], (tch::Kind::Double, tch::Device::Cpu)).into();
        assert!(
            LRA::Linear(a, b)
                .check([real(3, 1)].map(ok), [real(2, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn linear_dim_mismatch_fails() {
        // A=[2,3] but X has 4 rows — inner dimension mismatch.
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Double, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([0, 0], (tch::Kind::Double, tch::Device::Cpu)).into();
        assert!(
            LRA::Linear(a, b)
                .check([real(4, 1)].map(ok), [real(2, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn transpose_ok() {
        assert!(
            LRA::Transpose()
                .check([real(3, 4)].map(ok), [real(4, 3)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn transpose_wrong_shape_fails() {
        assert!(
            LRA::Transpose()
                .check([real(3, 4)].map(ok), [real(3, 4)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn ite_ok() {
        let t = real(3, 4);
        assert!(
            LRA::Ite()
                .check([bool_t(1, 1), t, t].map(ok), [t].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn ite_non_bool_guard_fails() {
        let t = real(1, 1);
        assert!(LRA::Ite().check([t, t, t].map(ok), [t].map(ok)).is_err());
    }

    #[test]
    fn ite_arm_mismatch_fails() {
        assert!(
            LRA::Ite()
                .check(
                    [bool_t(1, 1), real(1, 1), real(1, 2)].map(ok),
                    [real(1, 1)].map(ok)
                )
                .is_err()
        );
    }

    #[test]
    fn id_ok() {
        let t = real(4, 4);
        assert!(LRA::Id().check([t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn id_type_mismatch_fails() {
        assert!(
            LRA::Id()
                .check([real(1, 1)].map(ok), [real(2, 2)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn id_order_mismatch_fails() {
        assert!(
            LRA::Id()
                .check([real(1, 1)].map(ok), [dreal(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn id_arity_mismatch_fails() {
        let t = real(1, 1);
        assert!(LRA::Id().check([t, t].map(ok), [t, t].map(ok)).is_err());
        assert!(LRA::Id().check([t, t].map(ok), [t].map(ok)).is_err());
        assert!(LRA::Id().check([t].map(ok), [t, t].map(ok)).is_err());
    }

    #[test]
    fn havoc_ok() {
        assert!(
            LRA::AnyReal([2, 1])
                .check([].map(ok), [real(2, 1)].map(ok))
                .is_ok()
        );
        assert!(
            LRA::AnyBool([1, 1])
                .check([].map(ok), [bool_t(1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn havoc_writes_every_real_sort() {
        // `havoc(range)` writes `range`, derivatives included
        for range in [real(2, 1), dreal(2, 1), bool_t(2, 1), Sort::Zero] {
            assert!(
                LRA::havoc(&range)
                    .check([].map(ok), [range].map(ok))
                    .is_ok(),
                "havoc({range})"
            );
        }
    }

    #[test]
    fn havoc_read_fails() {
        let t = real(1, 1);
        assert!(
            LRA::AnyReal([1, 1])
                .check([t].map(ok), [t].map(ok))
                .is_err()
        );
    }

    #[test]
    fn havoc_arity_mismatch_fails() {
        assert!(
            LRA::AnyReal([2, 1])
                .check([].map(ok), [real(2, 1), bool_t(1, 1)].map(ok))
                .is_err()
        );
        assert!(LRA::AnyReal([1, 1]).check([].map(ok), [].map(ok)).is_err());
        // the write sort must match the op's declared range
        assert!(
            LRA::AnyReal([2, 1])
                .check([].map(ok), [real(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn zero_ok() {
        // ZERO writes a real tangent, never a value
        assert!(
            LRA::RealZerograd([2, 1])
                .check([].map(ok), [dreal(2, 1)].map(ok))
                .is_ok()
        );
        assert!(
            LRA::RealZerograd([2, 1])
                .check([].map(ok), [real(2, 1)].map(ok))
                .is_err()
        );
        // the trivial tangent's zero writes exactly the Zero sort
        assert!(LRA::Zero().check([].map(ok), [Sort::Zero].map(ok)).is_ok());
        assert!(
            LRA::Zero()
                .check([].map(ok), [bool_t(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn zero_is_the_tangent_of_bools() {
        // Differential::zero follows the tangent: reals get RealZerograd,
        // the collapsed sorts get Zero
        assert!(matches!(
            LRA::zero(&real(2, 1).T()),
            LRA::RealZerograd([2, 1])
        ));
        assert!(matches!(LRA::zero(&bool_t(1, 1).T()), LRA::Zero()));
    }

    #[test]
    fn zero_arity_mismatch_fails() {
        let t = dreal(1, 1);
        assert!(
            LRA::RealZerograd([1, 1])
                .check([t].map(ok), [t].map(ok))
                .is_err()
        );
        assert!(
            LRA::RealZerograd([2, 1])
                .check([].map(ok), [dreal(2, 1), dreal(1, 1)].map(ok))
                .is_err()
        );
        assert!(
            LRA::RealZerograd([1, 1])
                .check([].map(ok), [].map(ok))
                .is_err()
        );
    }
}
