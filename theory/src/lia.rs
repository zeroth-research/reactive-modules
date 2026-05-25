/*!
# Linear integer arithmetic

Defines the theory [`LIA`] of linear integer arithmetic over matrices,
mixing integer and boolean matrices in a single signature.

A [`Type`] value is either `Int(rows, cols)` or `Bool(rows, cols)`.
`Type` converts to and from [`int::IntType`] and [`bool::PropType`]
so that integer and propositional terms embed directly into `LIA`. The
operations in [`LIA`] are:

- [`LIA::Const`] — an integer matrix literal whose shape must match the
  declared (integer) write type.
- [`LIA::BoolConst`], [`LIA::And`], [`LIA::Or`], [`LIA::Xor`], [`LIA::Not`]
  — boolean operations on the boolean fragment of `Type`.
- [`LIA::Le`], [`LIA::Lt`], [`LIA::Ge`], [`LIA::Gt`], [`LIA::Eq`], [`LIA::Ne`]
  — pointwise integer comparisons producing a scalar `Bool(1,1)`.
- [`LIA::Ite`] — if-then-else: reads a boolean guard and two same-typed branches.
- [`LIA::Linear`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and
  `B` constant integer matrices of compatible shapes.
- [`LIA::ReLU`] — the shape-preserving rectified-linear map on integer matrices.

`LIA` implements [`Theory`]; [`Theory::check`] validates read/write
shapes against the selected operation.

## Examples

```
use theory::Theory;
use theory::lia::{LIA, Type};

// Pointwise less-than on scalars: Int(1,1), Int(1,1) -> Bool(1,1).
let i = Type::Int([1, 1]);
let b = Type::Bool([1, 1]);
assert!(LIA::Lt.check([i, i], [b]).is_ok());

// ReLU preserves shape and stays in the integer fragment.
let m = Type::Int([3, 4]);
assert!(LIA::ReLU.check([m], [m]).is_ok());
assert!(LIA::ReLU.check([b], [b]).is_err());
```
*/

use std::fmt;

use crate::*;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Type {
    Int([usize; 2]),
    Bool([usize; 2]),
}

impl Type {
    pub fn is_bool(&self) -> bool {
        matches!(self, Type::Bool(..))
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Type::Int(..))
    }

    pub fn shape(&self) -> &[usize; 2] {
        match self {
            Type::Bool(shape) | Type::Int(shape) => shape,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int([i, j]) => write!(f, "Int({i}, {j})"),
            Type::Bool([i, j]) => write!(f, "Bool({i}, {j})"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum LIA {
    // constants
    ConstInt(crate::Tensor),
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
    // XXX: should these be in LIA?
    ReLU,
    Argmax,
    Min,
    Max,
    // control flow
    Ite,
    Id,
}

impl fmt::Display for LIA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LIA::ConstInt(cm) => write!(f, "{}", cm),
            LIA::ConstBool(cm) => write!(f, "{}", cm),
            LIA::And => write!(f, "And"),
            LIA::Or => write!(f, "Or"),
            LIA::Xor => write!(f, "Xor"),
            LIA::Not => write!(f, "Not"),
            LIA::Le => write!(f, "Le"),
            LIA::Lt => write!(f, "Lt"),
            LIA::Ge => write!(f, "Ge"),
            LIA::Gt => write!(f, "Gt"),
            LIA::Eq => write!(f, "Eq"),
            LIA::Ne => write!(f, "Ne"),
            LIA::Linear(..) => write!(f, "Linear"),
            LIA::Add => write!(f, "Add"),
            LIA::ReLU => write!(f, "ReLU"),
            LIA::Argmax => write!(f, "Argmax"),
            LIA::Min => write!(f, "Min"),
            LIA::Max => write!(f, "Max"),
            LIA::Ite => write!(f, "Ite"),
            LIA::Id => write!(f, "Id"),
        }
    }
}

impl Theory for LIA {
    type DType = Type;

    fn check<R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<Type>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        match self {
            LIA::ConstInt(cm) => check_const(cm, read, write),
            LIA::ConstBool(_) | LIA::And | LIA::Or | LIA::Xor | LIA::Not => {
                check_bool(self, read, write)
            }
            LIA::Le | LIA::Lt | LIA::Ge | LIA::Gt | LIA::Eq | LIA::Ne => {
                check_cmp(self, read, write)
            }
            LIA::Linear(a, b) => {
                let mut read = read.into_iter();
                let mut write = write.into_iter();
                check_linear_affine(self, a, b, &mut read, &mut write)
            }
            LIA::Add | LIA::ReLU | LIA::Argmax | LIA::Min | LIA::Max => {
                check_mat_ops(self, read, write)
            }
            LIA::Ite | LIA::Id => check_flow(self, read, write),
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
        Type::Int([i, j]) => {
            let size = cm.size();
            if size.len() != 2 {
                return Err(format!(
                    "ConstInt: initializer must be a 2D tensor, got {}D",
                    size.len()
                ));
            }
            if size[0] as usize != i {
                return Err(format!(
                    "ConstInt: initializer has wrong number of rows (has {}, expected {})",
                    size[0], i
                ));
            }
            if size[1] as usize != j {
                return Err(format!(
                    "ConstInt: some row has wrong length, expected {}",
                    j
                ));
            }
        }
        Type::Bool(..) => {
            return Err("Const must be integer matrix, not boolean".into());
        }
    }
    if write.next().is_some() {
        return Err("Const: returns more than one value".into());
    }
    Ok(())
}

fn check_bool<R, W, D>(op: &LIA, read: R, write: W) -> Result<(), String>
where
    D: TryInto<Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        LIA::ConstBool(cm) => {
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
        LIA::Not => {
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
        LIA::And | LIA::Or | LIA::Xor => {
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

fn check_cmp<R, W, D>(op: &LIA, read: R, write: W) -> Result<(), String>
where
    D: TryInto<Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    if read_nxt(&mut read, 0)? != read_nxt(&mut read, 1)? {
        return Err(format!("{:?}: input values must have the same type", op));
    }
    let Type::Bool([1, 1]) = write_nxt(&mut write, 0)? else {
        return Err(format!(
            "{:?}: input and output values must have the same type",
            op
        ));
    };
    Ok(())
}

fn check_mat_ops<R, W, D>(op: &LIA, read: R, write: W) -> Result<(), String>
where
    D: TryInto<Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        LIA::Add => {
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
            if !matches!(w1, Type::Int(..)) {
                return Err(format!(
                    "{:?}: input and output values must be int matrices",
                    op
                ));
            }
            Ok(())
        }
        LIA::ReLU => {
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
            if !matches!(w1, Type::Int(..)) {
                return Err(format!(
                    "{:?}: input and output values must be int matrices",
                    op
                ));
            }
            Ok(())
        }
        LIA::Argmax | LIA::Min | LIA::Max => {
            let (_r1, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                return Err(format!("{:?}: must read exactly one value", op));
            };
            let (w1, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                return Err(format!("{:?}: must write exactly one value", op));
            };
            match w1 {
                Type::Int([i, j]) => {
                    // FIXME: we should fix which dimension is 1..
                    if i == 1 || j == 1 {
                        return Ok(());
                    }
                    Err(format!(
                        "{:?}: output must be a vector, got matrix {}x{}",
                        op, i, j
                    ))
                }
                _ => Err(format!("{:?}: output must be integer matrix", op)),
            }
        }
        _ => unreachable!(),
    }
}

fn check_linear_affine<D>(
    op: &LIA,
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
        (Type::Int([d1, d2]), Type::Int([d3, d4])) => {
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
        _ => Err(format!("{:?}: input and output must be int matrices", op)),
    }
}

fn check_flow<R, W, D>(op: &LIA, read: R, write: W) -> Result<(), String>
where
    D: TryInto<Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        LIA::Id => {
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
        LIA::Ite => {
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

    fn int(r: usize, c: usize) -> Type {
        Type::Int([r, c])
    }

    fn bool_t(r: usize, c: usize) -> Type {
        Type::Bool([r, c])
    }

    #[test]
    fn type_kind_and_shape() {
        assert!(int(2, 3).is_int() && !int(2, 3).is_bool());
        assert_eq!(int(2, 3).shape(), &[2, 3]);
        assert!(bool_t(1, 1).is_bool() && !bool_t(1, 1).is_int());
    }

    #[test]
    fn const_int_ok() {
        let cm: crate::Tensor = tch::Tensor::from_slice2(&[[0i64, 1], [2, 3]]).into();
        assert!(LIA::ConstInt(cm).check([] as [Type; 0], [int(2, 2)]).is_ok());
    }

    #[test]
    fn const_int_bool_write_fails() {
        assert!(LIA::ConstInt(tch::Tensor::from_slice2(&[[0i64]]).into()).check([] as [Type; 0], [bool_t(1, 1)]).is_err());
    }

    #[test]
    fn const_int_wrong_rows_fails() {
        assert!(LIA::ConstInt(tch::Tensor::from_slice2(&[[0i64]]).into()).check([] as [Type; 0], [int(2, 1)]).is_err());
    }

    #[test]
    fn const_int_with_read_fails() {
        let t = int(1, 1);
        assert!(LIA::ConstInt(tch::Tensor::from_slice2(&[[0i64]]).into()).check([t], [t]).is_err());
    }

    #[test]
    fn const_bool_ok() {
        let cm: crate::Tensor = tch::Tensor::from_slice2(&[[true, false], [false, true]]).into();
        assert!(LIA::ConstBool(cm).check([] as [Type; 0], [bool_t(2, 2)]).is_ok());
    }

    #[test]
    fn const_bool_int_write_fails() {
        assert!(LIA::ConstBool(tch::Tensor::from_slice2(&[[true]]).into()).check([] as [Type; 0], [int(1, 1)]).is_err());
    }

    #[test]
    fn not_ok() {
        let b = bool_t(2, 3);
        assert!(LIA::Not.check([b], [b]).is_ok());
    }

    #[test]
    fn not_int_input_fails() {
        let t = int(1, 1);
        assert!(LIA::Not.check([t], [t]).is_err());
    }

    #[test]
    fn and_ok() {
        let b = bool_t(2, 2);
        assert!(LIA::And.check([b, b], [b]).is_ok());
    }

    #[test]
    fn or_ok() {
        let b = bool_t(1, 1);
        assert!(LIA::Or.check([b, b], [b]).is_ok());
    }

    #[test]
    fn xor_ok() {
        let b = bool_t(3, 1);
        assert!(LIA::Xor.check([b, b], [b]).is_ok());
    }

    #[test]
    fn and_int_output_fails() {
        let b = bool_t(1, 1);
        assert!(LIA::And.check([b, b], [int(1, 1)]).is_err());
    }

    #[test]
    fn and_type_mismatch_fails() {
        assert!(LIA::And.check([bool_t(1, 1), bool_t(1, 2)], [bool_t(1, 1)]).is_err());
    }

    #[test]
    fn lt_ok() {
        assert!(LIA::Lt.check([int(1, 1), int(1, 1)], [bool_t(1, 1)]).is_ok());
    }

    #[test]
    fn le_ok() {
        assert!(LIA::Le.check([int(2, 3), int(2, 3)], [bool_t(1, 1)]).is_ok());
    }

    #[test]
    fn eq_ok() {
        assert!(LIA::Eq.check([int(1, 1), int(1, 1)], [bool_t(1, 1)]).is_ok());
    }

    #[test]
    fn cmp_non_bool_output_fails() {
        let t = int(1, 1);
        assert!(LIA::Lt.check([t, t], [t]).is_err());
    }

    #[test]
    fn cmp_input_mismatch_fails() {
        assert!(LIA::Eq.check([int(1, 1), int(1, 2)], [bool_t(1, 1)]).is_err());
    }

    #[test]
    fn add_ok() {
        let t = int(3, 4);
        assert!(LIA::Add.check([t, t], [t]).is_ok());
    }

    #[test]
    fn add_shape_mismatch_fails() {
        assert!(LIA::Add.check([int(1, 2), int(2, 1)], [int(1, 2)]).is_err());
    }

    #[test]
    fn add_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LIA::Add.check([b, b], [b]).is_err());
    }

    #[test]
    fn relu_ok() {
        let t = int(3, 4);
        assert!(LIA::ReLU.check([t], [t]).is_ok());
    }

    #[test]
    fn relu_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LIA::ReLU.check([b], [b]).is_err());
    }

    #[test]
    fn argmax_ok() {
        assert!(LIA::Argmax.check([int(3, 4)], [int(1, 4)]).is_ok());
    }

    #[test]
    fn argmax_matrix_output_fails() {
        assert!(LIA::Argmax.check([int(3, 4)], [int(3, 4)]).is_err());
    }

    #[test]
    fn min_ok() {
        assert!(LIA::Min.check([int(4, 1)], [int(1, 1)]).is_ok());
    }

    #[test]
    fn linear_ok() {
        let a: crate::Tensor = tch::Tensor::from_slice2(&[[1i64, 0], [0, 1]]).into();
        let b: crate::Tensor = tch::Tensor::zeros(&[0, 0], (tch::Kind::Int64, tch::Device::Cpu)).into();
        assert!(LIA::Linear(a, b).check([int(1, 2)], [int(2, 2)]).is_ok());
    }

    #[test]
    fn linear_dim_mismatch_fails() {
        let a: crate::Tensor = tch::Tensor::from_slice2(&[[1i64, 0], [0, 1]]).into();
        let b: crate::Tensor = tch::Tensor::zeros(&[0, 0], (tch::Kind::Int64, tch::Device::Cpu)).into();
        assert!(LIA::Linear(a, b).check([int(1, 3)], [int(2, 3)]).is_err());
    }

    #[test]
    fn ite_ok() {
        let t = int(3, 4);
        assert!(LIA::Ite.check([bool_t(1, 1), t, t], [t]).is_ok());
    }

    #[test]
    fn ite_non_bool_guard_fails() {
        let t = int(1, 1);
        assert!(LIA::Ite.check([t, t, t], [t]).is_err());
    }

    #[test]
    fn ite_arm_mismatch_fails() {
        assert!(LIA::Ite.check([bool_t(1, 1), int(1, 1), int(1, 2)], [int(1, 1)]).is_err());
    }

    #[test]
    fn id_ok() {
        let t = int(4, 4);
        assert!(LIA::Id.check([t], [t]).is_ok());
    }

    #[test]
    fn id_type_mismatch_fails() {
        assert!(LIA::Id.check([int(1, 1)], [int(2, 2)]).is_err());
    }
}
