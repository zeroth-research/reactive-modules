/*!
# Linear integer arithmetic

Defines the theory [`LRA`] of linear integer arithmetic over matrices,
mixing integer and boolean matrices in a single signature.

A [`Sort`] value is either `Int(rows, cols)` or `Bool(rows, cols)`.
`Type` converts to and from [`int::IntType`] and [`bool::PropType`]
so that integer and propositional terms embed directly into `RLA`. The
operations in [`LRA`] are:

- [`LRA::Const`] — a matrix literal whose sort (real or boolean) is taken
  from the write wire; the tensor's element kind must match that sort.
- [`LRA::And`], [`LRA::Or`], [`LRA::Xor`], [`LRA::Not`]
  — boolean operations on the boolean fragment of `Type`.
- [`LRA::Le`], [`LRA::Lt`], [`LRA::Ge`], [`LRA::Gt`], [`LRA::Eq`], [`LRA::Ne`]
  — pointwise integer comparisons producing a scalar `Bool(1,1)`.
- [`LRA::Ite`] — if-then-else: reads a boolean guard and two same-typed branches.
- [`LRA::Linear`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and
  `B` constant integer matrices of compatible shapes.
- [`LRA::ReLU`] — the shape-preserving rectified-linear map on integer matrices.

`RLA` implements [`Theory`]; [`Theory::check`] validates read/write
shapes against the selected operation.

## Examples

```
use theory::Theory;
use theory::lra::{LRA, Sort};

// Wires carry a sort and a degree; ordinary operands are degree 0.
let deg0 = |s| Ok::<_, String>((s, 0u8));

// Pointwise less-than on scalars: Real(1,1), Real(1,1) -> Bool(1,1).
let i = Sort::Real([1, 1]);
let b = Sort::Bool([1, 1]);
assert!(LRA::Lt().check([i, i].map(deg0), [b].map(deg0)).is_ok());

// ReLU preserves shape and stays in the real fragment.
let m = Sort::Real([3, 4]);
assert!(LRA::ReLU().check([m].map(deg0), [m].map(deg0)).is_ok());
assert!(LRA::ReLU().check([b].map(deg0), [b].map(deg0)).is_err());
```
*/

use crate::*;
#[cfg(feature = "pyo3")]
use pyo3::pyclass;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Sort {
    Real([usize; 2]),
    Bool([usize; 2]),
}

impl Sort {
    pub fn is_bool(&self) -> bool {
        matches!(self, Sort::Bool(..))
    }

    pub fn is_real(&self) -> bool {
        matches!(self, Sort::Real(..))
    }

    pub fn shape(&self) -> &[usize; 2] {
        match self {
            Sort::Bool(shape) | Sort::Real(shape) => shape,
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Real([i, j]) => write!(f, "Real({i}, {j})"),
            Sort::Bool([i, j]) => write!(f, "Bool({i}, {j})"),
        }
    }
}

#[derive(Clone, Debug, strum::Display)]
#[cfg_attr(feature = "pyo3", pyclass(frozen))]
pub enum LRA {
    // constant matrix literal; its sort (Real or Bool) is taken from the write wire
    #[strum(to_string = "{0}")]
    Const(crate::PyTensor),
    // boolean operations
    And(),
    Or(),
    Xor(),
    Not(),
    // integer comparisons
    Le(),
    Lt(),
    Ge(),
    Gt(),
    Eq(),
    Ne(),
    // linear / matrix operations
    // A*x + B where `A` and `B` are constants
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
    Zero(),
    Havoc(),
}

impl Sequential for LRA {
    const SKIP: Self = LRA::Id();
}

impl Combinatorial for LRA {
    const HAVOC: Self = LRA::Havoc();
}

impl Differential for LRA {
    const ZERO: Self = LRA::Zero();
}

impl Theory for LRA {
    type Sort = Sort;
    const NAME: &'static str = "LRA";

    fn check<R, W, E: fmt::Display>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<(Sort, u8), E>>,
        W: IntoIterator<Item = Result<(Sort, u8), E>>,
    {
        match self {
            LRA::Const(cm) => check_const(cm, read, write),
            LRA::Zero() => check_zero(read, write),
            LRA::Havoc() => check_havoc(read, write),
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

fn check_const<R, W, E: fmt::Display>(cm: &crate::PyTensor, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<(Sort, u8), E>>,
    W: IntoIterator<Item = Result<(Sort, u8), E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    if read.next().is_some() {
        return Err("Const: cannot read values".into());
    }
    // the sort comes from the write wire; validate the tensor's kind matches it
    let [i, j] = match next_with_degree(&mut write, 0)? {
        (Sort::Real([i, j]), _) => {
            if cm.is_bool() {
                return Err("Const: write wire is Real but initializer is a boolean tensor".into());
            }
            [i, j]
        }
        (Sort::Bool([i, j]), degree) => {
            if degree != 0 {
                return Err("Cannot derive a bool. Use ZERO to apply a no change".to_string());
            }
            if !cm.is_bool() {
                return Err(
                    "Const: write wire is Bool but initializer is not a boolean tensor".into(),
                );
            }
            [i, j]
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

fn check_bool<R, W, E: fmt::Display>(op: &LRA, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<(Sort, u8), E>>,
    W: IntoIterator<Item = Result<(Sort, u8), E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        LRA::Not() => {
            let (r, w) = (
                next_expect_degree(&mut read, 0, 0)?,
                next_expect_degree(&mut write, 0, 0)?,
            );
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
            let w1 = next_expect_degree(&mut write, 0, 0)?;
            let (r1, r2, None) = (
                next_expect_degree(&mut read, 0, 0)?,
                next_expect_degree(&mut read, 1, 0)?,
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
    R: IntoIterator<Item = Result<(Sort, u8), E>>,
    W: IntoIterator<Item = Result<(Sort, u8), E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let r1 = next_with_degree(&mut read, 0)?;
    let r2 = next_with_degree(&mut read, 1)?;
    if r1 != r2 {
        return Err(format!("{:?}: input values must have the same type", op));
    }
    let shape = match r1 {
        (Sort::Real(s), _) => s,
        _ => {
            return Err(format!(
                "{:?}: inputs must be Real matrices, got {}",
                r1.0, op
            ));
        }
    };
    let w1 = next_expect_degree(&mut write, 0, 0)?;
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
    R: IntoIterator<Item = Result<(Sort, u8), E>>,
    W: IntoIterator<Item = Result<(Sort, u8), E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        LRA::Add() | LRA::Sub() => {
            let (r1, r2, None) = (
                next_with_degree(&mut read, 0)?,
                next_with_degree(&mut read, 1)?,
                read.next(),
            ) else {
                return Err(format!("{:?}: must read exactly two values", op));
            };
            let (w1, None) = (next_with_degree(&mut write, 0)?, write.next()) else {
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
            if !matches!(w1.0, Sort::Real(..)) {
                return Err(format!(
                    "{:?}: input and output values must be real matrices",
                    op
                ));
            }
            Ok(())
        }
        LRA::ReLU() => {
            let (r1, None) = (next_with_degree(&mut read, 0)?, read.next()) else {
                return Err(format!("{:?}: must read exactly one value", op));
            };
            let (w1, None) = (next_with_degree(&mut write, 0)?, write.next()) else {
                return Err(format!("{:?}: must write exactly one value", op));
            };
            if r1 != w1 {
                return Err(format!(
                    "{:?}: input and output must have the same type",
                    op
                ));
            }
            if !matches!(w1.0, Sort::Real(..)) {
                return Err(format!(
                    "{:?}: input and output values must be real matrices",
                    op
                ));
            }
            Ok(())
        }
        LRA::Argmax() | LRA::Min() | LRA::Max() => {
            // TODO: check whether the conditions of the read are sound
            let (_r1, None) = (next_with_degree(&mut read, 0)?, read.next()) else {
                return Err(format!("{:?}: must read exactly one value", op));
            };
            let (w1, None) = (next_with_degree(&mut write, 0)?, write.next()) else {
                return Err(format!("{:?}: must write exactly one value", op));
            };
            match w1 {
                (Sort::Real([i, j]), _) => {
                    // FIXME: we should fix which dimension is 1..
                    if i == 1 || j == 1 {
                        return Ok(());
                    }
                    Err(format!(
                        "{:?}: output must be a vector, got matrix {}x{}",
                        op, i, j
                    ))
                }
                _ => Err(format!("{:?}: output must be real matrix", op)),
            }
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
    R: Iterator<Item = Result<(Sort, u8), E>>,
    W: Iterator<Item = Result<(Sort, u8), E>>,
{
    let (r1, None) = (next_with_degree(read, 0)?, read.next()) else {
        return Err(format!("{:?}: must read exactly one value", op));
    };
    let (w1, None) = (next_with_degree(write, 0)?, write.next()) else {
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

    match (r1, w1) {
        ((Sort::Real([d1, d2]), deg0), (Sort::Real([d3, d4]), deg1)) => {
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
            if deg0 != deg1 {
                return Err("Differential form degree mismatch".to_string());
            }
            Ok(())
        }
        _ => Err(format!("{:?}: input and output must be real matrices", op)),
    }
}

fn check_transpose<R, W, E: fmt::Display>(op: &LRA, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<(Sort, u8), E>>,
    W: IntoIterator<Item = Result<(Sort, u8), E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let (r1, None) = (next_with_degree(&mut read, 0)?, read.next()) else {
        return Err(format!("{:?}: must read exactly one value", op));
    };
    let (w1, None) = (next_with_degree(&mut write, 0)?, write.next()) else {
        return Err(format!("{:?}: must write exactly one value", op));
    };
    match (r1, w1) {
        ((Sort::Real([d1, d2]), deg0), (Sort::Real([e1, e2]), deg1)) => {
            if d2 != e1 || d1 != e2 {
                return Err(format!(
                    "{:?}: transpose of {}x{} must produce {}x{}, got {}x{}",
                    op, d1, d2, d2, d1, e1, e2
                ));
            }
            if deg0 != deg1 {
                return Err("Differential form degree mismatch".to_string());
            }
            Ok(())
        }
        _ => Err(format!("{:?}: input and output must be real matrices", op)),
    }
}

fn check_flow<R, W, E: fmt::Display>(op: &LRA, read: R, write: W) -> Result<(), String>
where
    R: IntoIterator<Item = Result<(Sort, u8), E>>,
    W: IntoIterator<Item = Result<(Sort, u8), E>>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        LRA::Id() => {
            let (r1, None) = (next_with_degree(&mut read, 0)?, read.next()) else {
                return Err(format!("{:?}: must read exactly one value", op));
            };
            let (w1, None) = (next_with_degree(&mut write, 0)?, write.next()) else {
                return Err(format!("{:?}: must write exactly one value", op));
            };
            if r1 != w1 {
                return Err(format!(
                    "{:?}: input and output must have the same type and degree",
                    op
                ));
            }
            Ok(())
        }
        LRA::Ite() => {
            let (r1, r2, r3, None) = (
                next_expect_degree(&mut read, 0, 0)?,
                next_with_degree(&mut read, 1)?,
                next_with_degree(&mut read, 2)?,
                read.next(),
            ) else {
                return Err(format!("{:?}: must read exactly three values", op));
            };
            let (w1, None) = (next_with_degree(&mut write, 0)?, write.next()) else {
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
        Sort::Real([r, c])
    }

    fn bool_t(r: usize, c: usize) -> Sort {
        Sort::Bool([r, c])
    }

    fn deg0(s: Sort) -> Result<(Sort, u8), String> {
        Ok((s, 0))
    }

    #[test]
    fn type_kind_and_shape() {
        assert!(real(2, 3).is_real() && !real(2, 3).is_bool());
        assert_eq!(real(2, 3).shape(), &[2, 3]);
        assert!(bool_t(1, 1).is_bool() && !bool_t(1, 1).is_real());
    }

    #[test]
    fn const_real_ok() {
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0.0f64, 1.0], [2.0, 3.0]]).into();
        assert!(LRA::Const(cm).check([].map(deg0), [real(2, 2)].map(deg0)).is_ok());
    }

    #[test]
    fn const_real_bool_write_fails() {
        assert!(
            LRA::Const(tch::Tensor::from_slice2(&[[0.0f64]]).into())
                .check([].map(deg0), [bool_t(1, 1)].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn const_real_wrong_rows_fails() {
        assert!(
            LRA::Const(tch::Tensor::from_slice2(&[[0.0f64]]).into())
                .check([].map(deg0), [real(2, 1)].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn const_real_with_read_fails() {
        let t = real(1, 1);
        assert!(
            LRA::Const(tch::Tensor::from_slice2(&[[0.0f64]]).into())
                .check([t].map(deg0), [t].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn const_bool_ok() {
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[true, false], [false, true]]).into();
        assert!(
            LRA::Const(cm)
                .check([].map(deg0), [bool_t(2, 2)].map(deg0))
                .is_ok()
        );
    }

    #[test]
    fn const_bool_real_write_fails() {
        assert!(
            LRA::Const(tch::Tensor::from_slice2(&[[true]]).into())
                .check([].map(deg0), [real(1, 1)].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn not_ok() {
        let b = bool_t(2, 3);
        assert!(LRA::Not().check([b].map(deg0), [b].map(deg0)).is_ok());
    }

    #[test]
    fn not_real_input_fails() {
        let t = real(1, 1);
        assert!(LRA::Not().check([t].map(deg0), [t].map(deg0)).is_err());
    }

    #[test]
    fn and_ok() {
        let b = bool_t(2, 2);
        assert!(LRA::And().check([b, b].map(deg0), [b].map(deg0)).is_ok());
    }

    #[test]
    fn or_ok() {
        let b = bool_t(1, 1);
        assert!(LRA::Or().check([b, b].map(deg0), [b].map(deg0)).is_ok());
    }

    #[test]
    fn xor_ok() {
        let b = bool_t(3, 1);
        assert!(LRA::Xor().check([b, b].map(deg0), [b].map(deg0)).is_ok());
    }

    #[test]
    fn and_real_output_fails() {
        let b = bool_t(1, 1);
        assert!(LRA::And().check([b, b].map(deg0), [real(1, 1)].map(deg0)).is_err());
    }

    #[test]
    fn and_type_mismatch_fails() {
        assert!(
            LRA::And()
                .check([bool_t(1, 1), bool_t(1, 2)].map(deg0), [bool_t(1, 1)].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn lt_ok() {
        assert!(
            LRA::Lt()
                .check([real(1, 1), real(1, 1)].map(deg0), [bool_t(1, 1)].map(deg0))
                .is_ok()
        );
    }

    #[test]
    fn le_ok() {
        assert!(
            LRA::Le()
                .check([real(2, 3), real(2, 3)].map(deg0), [bool_t(2, 3)].map(deg0))
                .is_ok()
        );

        assert!(
            LRA::Le()
                .check([real(3, 3), real(2, 3)].map(deg0), [bool_t(2, 3)].map(deg0))
                .is_err()
        );

        assert!(
            LRA::Le()
                .check([real(2, 3), real(2, 3)].map(deg0), [bool_t(3, 3)].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn eq_ok() {
        assert!(
            LRA::Eq()
                .check([real(2, 2), real(2, 2)].map(deg0), [bool_t(2, 2)].map(deg0))
                .is_ok()
        );
    }

    #[test]
    fn cmp_non_bool_output_fails() {
        let t = real(1, 1);
        assert!(LRA::Lt().check([t, t].map(deg0), [t].map(deg0)).is_err());
    }

    #[test]
    fn cmp_input_mismatch_fails() {
        assert!(
            LRA::Eq()
                .check([real(1, 1), real(1, 2)].map(deg0), [bool_t(1, 1)].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn add_ok() {
        let t = real(3, 4);
        assert!(LRA::Add().check([t, t].map(deg0), [t].map(deg0)).is_ok());
    }

    #[test]
    fn add_shape_mismatch_fails() {
        assert!(
            LRA::Add()
                .check([real(1, 2), real(2, 1)].map(deg0), [real(1, 2)].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn add_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LRA::Add().check([b, b].map(deg0), [b].map(deg0)).is_err());
    }

    #[test]
    fn relu_ok() {
        let t = real(3, 4);
        assert!(LRA::ReLU().check([t].map(deg0), [t].map(deg0)).is_ok());
    }

    #[test]
    fn relu_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LRA::ReLU().check([b].map(deg0), [b].map(deg0)).is_err());
    }

    #[test]
    fn argmax_ok() {
        assert!(LRA::Argmax().check([real(3, 4)].map(deg0), [real(1, 4)].map(deg0)).is_ok());
    }

    #[test]
    fn argmax_matrix_output_fails() {
        assert!(LRA::Argmax().check([real(3, 4)].map(deg0), [real(3, 4)].map(deg0)).is_err());
    }

    #[test]
    fn min_ok() {
        assert!(LRA::Min().check([real(4, 1)].map(deg0), [real(1, 1)].map(deg0)).is_ok());
    }

    #[test]
    fn linear_ok() {
        // A=[2,3] maps 3 features to 2; X=[3,4] is 3 features × 4 batch items.
        // Convention: Y = A·X  →  Y=[2,4].
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Double, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([0, 0], (tch::Kind::Double, tch::Device::Cpu)).into();
        assert!(LRA::Linear(a, b).check([real(3, 4)].map(deg0), [real(2, 4)].map(deg0)).is_ok());
    }

    #[test]
    fn linear_with_bias_ok() {
        // A=[2,3], b=[2,1] column bias, X=[3,1] single sample → Y=[2,1].
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Double, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([2, 1], (tch::Kind::Double, tch::Device::Cpu)).into();
        assert!(LRA::Linear(a, b).check([real(3, 1)].map(deg0), [real(2, 1)].map(deg0)).is_ok());
    }

    #[test]
    fn linear_dim_mismatch_fails() {
        // A=[2,3] but X has 4 rows — inner dimension mismatch.
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Double, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([0, 0], (tch::Kind::Double, tch::Device::Cpu)).into();
        assert!(LRA::Linear(a, b).check([real(4, 1)].map(deg0), [real(2, 1)].map(deg0)).is_err());
    }

    #[test]
    fn transpose_ok() {
        assert!(LRA::Transpose().check([real(3, 4)].map(deg0), [real(4, 3)].map(deg0)).is_ok());
    }

    #[test]
    fn transpose_wrong_shape_fails() {
        assert!(LRA::Transpose().check([real(3, 4)].map(deg0), [real(3, 4)].map(deg0)).is_err());
    }

    #[test]
    fn ite_ok() {
        let t = real(3, 4);
        assert!(LRA::Ite().check([bool_t(1, 1), t, t].map(deg0), [t].map(deg0)).is_ok());
    }

    #[test]
    fn ite_non_bool_guard_fails() {
        let t = real(1, 1);
        assert!(LRA::Ite().check([t, t, t].map(deg0), [t].map(deg0)).is_err());
    }

    #[test]
    fn ite_arm_mismatch_fails() {
        assert!(
            LRA::Ite()
                .check([bool_t(1, 1), real(1, 1), real(1, 2)].map(deg0), [real(1, 1)].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn id_ok() {
        let t = real(4, 4);
        assert!(LRA::Id().check([t].map(deg0), [t].map(deg0)).is_ok());
    }

    #[test]
    fn id_type_mismatch_fails() {
        assert!(LRA::Id().check([real(1, 1)].map(deg0), [real(2, 2)].map(deg0)).is_err());
    }

    #[test]
    fn id_arity_mismatch_fails() {
        let t = real(1, 1);
        assert!(LRA::Id().check([t, t].map(deg0), [t, t].map(deg0)).is_err());
        assert!(LRA::Id().check([t, t].map(deg0), [t].map(deg0)).is_err());
        assert!(LRA::Id().check([t].map(deg0), [t, t].map(deg0)).is_err());
    }

    #[test]
    fn havoc_ok() {
        assert!(LRA::Havoc().check([].map(deg0), [real(2, 1)].map(deg0)).is_ok());
    }

    #[test]
    fn havoc_read_fails() {
        let t = real(1, 1);
        assert!(LRA::Havoc().check([t].map(deg0), [t].map(deg0)).is_err());
    }

    #[test]
    fn havoc_arity_mismatch_fails() {
        assert!(
            LRA::Havoc()
                .check([].map(deg0), [real(2, 1), bool_t(1, 1)].map(deg0))
                .is_err()
        );
        assert!(
            LRA::Havoc()
                .check([].map(deg0), [].map(deg0))
                .is_err()
        );
    }

    #[test]
    fn zero_ok() {
        assert!(LRA::Zero().check([].map(deg0), [real(2, 1)].map(deg0)).is_ok());
    }

    #[test]
    fn zero_arity_mismatch_fails() {
        let t = real(1, 1);
        assert!(LRA::Zero().check([t].map(deg0), [t].map(deg0)).is_err());
        assert!(
            LRA::Zero()
                .check([].map(deg0), [real(2, 1), real(1, 1)].map(deg0))
                .is_err()
        );
        assert!(LRA::Zero().check([].map(deg0), [].map(deg0)).is_err());
    }
}
