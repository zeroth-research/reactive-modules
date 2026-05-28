/*!
# Linear integer arithmetic

Defines the theory [`LRA`] of linear integer arithmetic over matrices,
mixing integer and boolean matrices in a single signature.

A [`Type`] value is either `Int(rows, cols)` or `Bool(rows, cols)`.
`Type` converts to and from [`int::IntType`] and [`bool::PropType`]
so that integer and propositional terms embed directly into `RLA`. The
operations in [`LRA`] are:

- [`LRA::Const`] — an real matrix literal whose shape must match the
  declared (integer) write type.
- [`LRA::BoolConst`], [`LRA::And`], [`LRA::Or`], [`LRA::Xor`], [`LRA::Not`]
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
use theory::lra::{LRA, Type};

// Pointwise less-than on scalars: Real(1,1), Real(1,1) -> Bool(1,1).
let i = Type::Real([1, 1]);
let b = Type::Bool([1, 1]);
assert!(LRA::Lt.check([i, i], [b]).is_ok());

// ReLU preserves shape and stays in the real fragment.
let m = Type::Real([3, 4]);
assert!(LRA::ReLU.check([m], [m]).is_ok());
assert!(LRA::ReLU.check([b], [b]).is_err());
```
*/

use std::fmt;

use crate::*;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Type {
    Real([usize; 2]),
    Bool([usize; 2]),
}

impl Type {
    pub fn is_bool(&self) -> bool {
        matches!(self, Type::Bool(..))
    }

    pub fn is_real(&self) -> bool {
        matches!(self, Type::Real(..))
    }

    pub fn shape(&self) -> &[usize; 2] {
        match self {
            Type::Bool(shape) | Type::Real(shape) => shape,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Real([i, j]) => write!(f, "Real({i}, {j})"),
            Type::Bool([i, j]) => write!(f, "Bool({i}, {j})"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum LRA {
    // constants
    ConstReal(crate::Tensor),
    ConstBool(crate::Tensor),
    // boolean operations
    And,
    Or,
    Xor,
    Not,
    // integer comparisons
    Le,
    Lt,
    Ge,
    Gt,
    Eq,
    Ne,
    // linear / matrix operations
    // A*x + B where `A` and `B` are constants
    Linear(crate::Tensor, crate::Tensor),
    Add,
    // XXX: should these be in RLA?
    ReLU,
    Argmax,
    Min,
    Max,
    // control flow
    Ite,
    Id,
    Uninterpreted(String),
}

impl fmt::Display for LRA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LRA::ConstReal(cm) => write!(f, "{}", cm),
            LRA::ConstBool(cm) => write!(f, "{}", cm),
            LRA::And => write!(f, "And"),
            LRA::Or => write!(f, "Or"),
            LRA::Xor => write!(f, "Xor"),
            LRA::Not => write!(f, "Not"),
            LRA::Le => write!(f, "Le"),
            LRA::Lt => write!(f, "Lt"),
            LRA::Ge => write!(f, "Ge"),
            LRA::Gt => write!(f, "Gt"),
            LRA::Eq => write!(f, "Eq"),
            LRA::Ne => write!(f, "Ne"),
            LRA::Linear(..) => write!(f, "Linear"),
            LRA::Add => write!(f, "Add"),
            LRA::ReLU => write!(f, "ReLU"),
            LRA::Argmax => write!(f, "Argmax"),
            LRA::Min => write!(f, "Min"),
            LRA::Max => write!(f, "Max"),
            LRA::Ite => write!(f, "Ite"),
            LRA::Id => write!(f, "Id"),
            LRA::Uninterpreted(name) => write!(f, "Uninterpreted({name})"),
        }
    }
}

impl Theory for LRA {
    type DType = Type;

    fn check<R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<Self::DType>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        match self {
            LRA::ConstReal(cm) => check_const(cm, read, write),
            LRA::ConstBool(_) | LRA::And | LRA::Or | LRA::Xor | LRA::Not => {
                check_bool(self, read, write)
            }
            LRA::Le | LRA::Lt | LRA::Ge | LRA::Gt | LRA::Eq | LRA::Ne => {
                check_cmp(self, read, write)
            }
            LRA::Linear(a, b) => {
                let mut read = read.into_iter();
                let mut write = write.into_iter();
                check_linear_affine(self, a, b, &mut read, &mut write)
            }
            LRA::Add | LRA::ReLU | LRA::Argmax | LRA::Min | LRA::Max => {
                check_mat_ops(self, read, write)
            }
            LRA::Ite | LRA::Id => check_flow(self, read, write),
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
                return Err(format!(
                    "{:?}: expected exactly one write or one read, got none",
                    self
                ));
            }
        }
    }
}

fn check_const<R, W, D>(cm: &tch::Tensor, read: R, write: W) -> Result<(), String>
where
    D: TryInto<Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    if read.next().is_some() {
        return Err("Const: cannot read values".into());
    }
    let dtype = write_nxt(&mut write, 0)?;
    match dtype {
        Type::Real([i, j]) => {
            let size = cm.size();
            if size.len() != 2 {
                return Err(format!(
                    "ConstReal: initializer must be a 2D tensor, got {}D",
                    size.len()
                ));
            }
            if size[0] as usize != i {
                return Err(format!(
                    "ConstReal: initializer has wrong number of rows (has {}, expected {})",
                    size[0], i
                ));
            }
            if size[1] as usize != j {
                return Err(format!(
                    "ConstReal: some row has wrong length, expected {}",
                    j
                ));
            }
        }
        Type::Bool(..) => {
            return Err("Const must be real matrix, not boolean".into());
        }
    }
    if write.next().is_some() {
        return Err("Const: returns more than one value".into());
    }
    Ok(())
}

fn check_bool<R, W, D>(op: &LRA, read: R, write: W) -> Result<(), String>
where
    D: TryInto<Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        LRA::ConstBool(cm) => {
            if read.next().is_some() {
                return Err("ConstBool: cannot read values".into());
            }
            let Type::Bool([i, j]) = write_nxt(&mut write, 0)? else {
                return Err("ConstBool: write type must be Bool".into());
            };
            let size = cm.size();
            if size.len() != 2 {
                return Err(format!(
                    "ConstBool: initializer must be a 2D tensor, got {}D",
                    size.len()
                ));
            }
            if size[0] as usize != i {
                return Err(format!(
                    "ConstBool: initializer has wrong number of rows (has {}, expected {})",
                    size[0], i
                ));
            }
            if size[1] as usize != j {
                return Err(format!(
                    "ConstBool: some row has wrong length, expected {}",
                    j
                ));
            }
            if write.next().is_some() {
                return Err("ConstBool: returns more than one value".into());
            }
            Ok(())
        }
        LRA::Not => {
            let (r, w) = (read_nxt(&mut read, 0)?, write_nxt(&mut write, 0)?);
            if !matches!(r, Type::Bool(..)) {
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
        LRA::And | LRA::Or | LRA::Xor => {
            let w1 = write_nxt(&mut write, 0)?;
            let (r1, r2, None) = (
                read_nxt(&mut read, 0)?,
                read_nxt(&mut read, 1)?,
                read.next(),
            ) else {
                return Err(format!("{:?}: must read exactly two values", op));
            };
            if !matches!(w1, Type::Bool(..)) {
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

fn check_cmp<R, W, D>(op: &LRA, read: R, write: W) -> Result<(), String>
where
    D: TryInto<Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let r1 = read_nxt(&mut read, 0)?;
    let r2 = read_nxt(&mut read, 1)?;
    if r1 != r2 {
        return Err(format!("{:?}: input values must have the same type", op));
    }
    let shape = match r1 {
        Type::Real(s) => s,
        _ => return Err(format!("{:?}: inputs must be Real matrices, got {r1}", op)),
    };
    let w1 = write_nxt(&mut write, 0)?;
    if w1 != Type::Bool(shape) {
        return Err(format!(
            "{:?}: output must be Bool({:?}), got {w1}",
            op, shape
        ));
    }
    Ok(())
}

fn check_mat_ops<R, W, D>(op: &LRA, read: R, write: W) -> Result<(), String>
where
    D: TryInto<Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        LRA::Add => {
            let (r1, r2, None) = (
                read_nxt(&mut read, 0)?,
                read_nxt(&mut read, 1)?,
                read.next(),
            ) else {
                return Err(format!("{:?}: must read exactly two values", op));
            };
            let (w1, None) = (write_nxt(&mut write, 0)?, write.next()) else {
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
            if !matches!(w1, Type::Real(..)) {
                return Err(format!(
                    "{:?}: input and output values must be real matrices",
                    op
                ));
            }
            Ok(())
        }
        LRA::ReLU => {
            let (r1, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                return Err(format!("{:?}: must read exactly one value", op));
            };
            let (w1, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                return Err(format!("{:?}: must write exactly one value", op));
            };
            if r1 != w1 {
                return Err(format!(
                    "{:?}: input and output must have the same type",
                    op
                ));
            }
            if !matches!(w1, Type::Real(..)) {
                return Err(format!(
                    "{:?}: input and output values must be real matrices",
                    op
                ));
            }
            Ok(())
        }
        LRA::Argmax | LRA::Min | LRA::Max => {
            let (_r1, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                return Err(format!("{:?}: must read exactly one value", op));
            };
            let (w1, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                return Err(format!("{:?}: must write exactly one value", op));
            };
            match w1 {
                Type::Real([i, j]) => {
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

fn check_linear_affine<D>(
    op: &LRA,
    a: &tch::Tensor,
    b: &tch::Tensor,
    read: &mut impl Iterator<Item = D>,
    write: &mut impl Iterator<Item = D>,
) -> Result<(), String>
where
    D: TryInto<Type>,
{
    let (r1, None) = (read_nxt(read, 0)?, read.next()) else {
        return Err(format!("{:?}: must read exactly one value", op));
    };
    let (w1, None) = (write_nxt(write, 0)?, write.next()) else {
        return Err(format!("{:?}: must write exactly one value", op));
    };

    let a_size = a.size();
    if a_size.len() != 2 {
        return Err(format!("{:?}: `A` must be a 2D tensor", op));
    }
    let a_rows = a_size[0] as usize;
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
        let br = b_size[0] as usize;
        let bc = b_size[1] as usize;
        if br != 1 && bc != 1 {
            return Err(format!(
                "{:?}: `B` has to be a vector, got matrix {}x{}",
                op, br, bc
            ));
        }
        (br, bc)
    };

    match (r1, w1) {
        (Type::Real([d1, d2]), Type::Real([d3, d4])) => {
            if d2 != a_rows {
                return Err(format!(
                    "{:?}: mismatch in inner dimensions of `A` and `x`: A has {}x{}, x has {}x{}",
                    op, d1, d2, a_rows, a_rows
                ));
            }
            // `A*x` is a a_rows x d2 matrix, `B` has to have these dimensions (if non-empty)
            if b_rows > 0 && (a_rows != b_rows || d2 != b_cols) {
                return Err(format!(
                    "{:?}: A*x has dimension {}x{} while B has {}x{}",
                    op, a_rows, d2, b_rows, b_cols
                ));
            }
            if a_rows != d3 || d2 != d4 {
                return Err(format!(
                    "{:?}: bad output matrix dimensions, expected {}x{} but got {}x{}",
                    op, a_rows, d2, d3, d4
                ));
            }
            Ok(())
        }
        // TODO: should we allow also boolean matrices?
        _ => Err(format!("{:?}: input and output must be real matrices", op)),
    }
}

fn check_flow<R, W, D>(op: &LRA, read: R, write: W) -> Result<(), String>
where
    D: TryInto<Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        LRA::Id => {
            let (r1, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                return Err(format!("{:?}: must read exactly one value", op));
            };
            let (w1, None) = (write_nxt(&mut write, 0)?, write.next()) else {
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
        LRA::Ite => {
            let (r1, r2, r3, None) = (
                read_nxt(&mut read, 0)?,
                read_nxt(&mut read, 1)?,
                read_nxt(&mut read, 2)?,
                read.next(),
            ) else {
                return Err(format!("{:?}: must read exactly three values", op));
            };
            let (w1, None) = (write_nxt(&mut write, 0)?, write.next()) else {
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
            if r1 != Type::Bool([1, 1]) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn real(r: usize, c: usize) -> Type {
        Type::Real([r, c])
    }

    fn bool_t(r: usize, c: usize) -> Type {
        Type::Bool([r, c])
    }

    #[test]
    fn type_kind_and_shape() {
        assert!(real(2, 3).is_real() && !real(2, 3).is_bool());
        assert_eq!(real(2, 3).shape(), &[2, 3]);
        assert!(bool_t(1, 1).is_bool() && !bool_t(1, 1).is_real());
    }

    #[test]
    fn const_real_ok() {
        let cm: crate::Tensor = tch::Tensor::from_slice2(&[[0.0f64, 1.0], [2.0, 3.0]]).into();
        assert!(
            LRA::ConstReal(cm)
                .check([] as [Type; 0], [real(2, 2)])
                .is_ok()
        );
    }

    #[test]
    fn const_real_bool_write_fails() {
        assert!(
            LRA::ConstReal(tch::Tensor::from_slice2(&[[0.0f64]]).into())
                .check([] as [Type; 0], [bool_t(1, 1)])
                .is_err()
        );
    }

    #[test]
    fn const_real_wrong_rows_fails() {
        assert!(
            LRA::ConstReal(tch::Tensor::from_slice2(&[[0.0f64]]).into())
                .check([] as [Type; 0], [real(2, 1)])
                .is_err()
        );
    }

    #[test]
    fn const_real_with_read_fails() {
        let t = real(1, 1);
        assert!(
            LRA::ConstReal(tch::Tensor::from_slice2(&[[0.0f64]]).into())
                .check([t], [t])
                .is_err()
        );
    }

    #[test]
    fn const_bool_ok() {
        let cm: crate::Tensor = tch::Tensor::from_slice2(&[[true, false], [false, true]]).into();
        assert!(
            LRA::ConstBool(cm)
                .check([] as [Type; 0], [bool_t(2, 2)])
                .is_ok()
        );
    }

    #[test]
    fn const_bool_real_write_fails() {
        assert!(
            LRA::ConstBool(tch::Tensor::from_slice2(&[[true]]).into())
                .check([] as [Type; 0], [real(1, 1)])
                .is_err()
        );
    }

    #[test]
    fn not_ok() {
        let b = bool_t(2, 3);
        assert!(LRA::Not.check([b], [b]).is_ok());
    }

    #[test]
    fn not_real_input_fails() {
        let t = real(1, 1);
        assert!(LRA::Not.check([t], [t]).is_err());
    }

    #[test]
    fn and_ok() {
        let b = bool_t(2, 2);
        assert!(LRA::And.check([b, b], [b]).is_ok());
    }

    #[test]
    fn or_ok() {
        let b = bool_t(1, 1);
        assert!(LRA::Or.check([b, b], [b]).is_ok());
    }

    #[test]
    fn xor_ok() {
        let b = bool_t(3, 1);
        assert!(LRA::Xor.check([b, b], [b]).is_ok());
    }

    #[test]
    fn and_real_output_fails() {
        let b = bool_t(1, 1);
        assert!(LRA::And.check([b, b], [real(1, 1)]).is_err());
    }

    #[test]
    fn and_type_mismatch_fails() {
        assert!(
            LRA::And
                .check([bool_t(1, 1), bool_t(1, 2)], [bool_t(1, 1)])
                .is_err()
        );
    }

    #[test]
    fn lt_ok() {
        assert!(
            LRA::Lt
                .check([real(1, 1), real(1, 1)], [bool_t(1, 1)])
                .is_ok()
        );
    }

    #[test]
    fn le_ok() {
        assert!(
            LRA::Le
                .check([real(2, 3), real(2, 3)], [bool_t(1, 1)])
                .is_ok()
        );
    }

    #[test]
    fn eq_ok() {
        assert!(
            LRA::Eq
                .check([real(1, 1), real(1, 1)], [bool_t(1, 1)])
                .is_ok()
        );
    }

    #[test]
    fn cmp_non_bool_output_fails() {
        let t = real(1, 1);
        assert!(LRA::Lt.check([t, t], [t]).is_err());
    }

    #[test]
    fn cmp_input_mismatch_fails() {
        assert!(
            LRA::Eq
                .check([real(1, 1), real(1, 2)], [bool_t(1, 1)])
                .is_err()
        );
    }

    #[test]
    fn add_ok() {
        let t = real(3, 4);
        assert!(LRA::Add.check([t, t], [t]).is_ok());
    }

    #[test]
    fn add_shape_mismatch_fails() {
        assert!(
            LRA::Add
                .check([real(1, 2), real(2, 1)], [real(1, 2)])
                .is_err()
        );
    }

    #[test]
    fn add_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LRA::Add.check([b, b], [b]).is_err());
    }

    #[test]
    fn relu_ok() {
        let t = real(3, 4);
        assert!(LRA::ReLU.check([t], [t]).is_ok());
    }

    #[test]
    fn relu_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LRA::ReLU.check([b], [b]).is_err());
    }

    #[test]
    fn argmax_ok() {
        assert!(LRA::Argmax.check([real(3, 4)], [real(1, 4)]).is_ok());
    }

    #[test]
    fn argmax_matrix_output_fails() {
        assert!(LRA::Argmax.check([real(3, 4)], [real(3, 4)]).is_err());
    }

    #[test]
    fn min_ok() {
        assert!(LRA::Min.check([real(4, 1)], [real(1, 1)]).is_ok());
    }

    #[test]
    fn linear_ok() {
        let a: crate::Tensor = tch::Tensor::from_slice2(&[[1.0f64, 0.0], [0.0, 1.0]]).into();
        let b: crate::Tensor =
            tch::Tensor::zeros(&[0, 0], (tch::Kind::Double, tch::Device::Cpu)).into();
        assert!(LRA::Linear(a, b).check([real(1, 2)], [real(2, 2)]).is_ok());
    }

    #[test]
    fn linear_dim_mismatch_fails() {
        let a: crate::Tensor = tch::Tensor::from_slice2(&[[1.0f64, 0.0], [0.0, 1.0]]).into();
        let b: crate::Tensor =
            tch::Tensor::zeros(&[0, 0], (tch::Kind::Double, tch::Device::Cpu)).into();
        assert!(LRA::Linear(a, b).check([real(1, 3)], [real(2, 3)]).is_err());
    }

    #[test]
    fn ite_ok() {
        let t = real(3, 4);
        assert!(LRA::Ite.check([bool_t(1, 1), t, t], [t]).is_ok());
    }

    #[test]
    fn ite_non_bool_guard_fails() {
        let t = real(1, 1);
        assert!(LRA::Ite.check([t, t, t], [t]).is_err());
    }

    #[test]
    fn ite_arm_mismatch_fails() {
        assert!(
            LRA::Ite
                .check([bool_t(1, 1), real(1, 1), real(1, 2)], [real(1, 1)])
                .is_err()
        );
    }

    #[test]
    fn id_ok() {
        let t = real(4, 4);
        assert!(LRA::Id.check([t], [t]).is_ok());
    }

    #[test]
    fn id_type_mismatch_fails() {
        assert!(LRA::Id.check([real(1, 1)], [real(2, 2)]).is_err());
    }
}
