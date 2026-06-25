/*!
# Linear integer arithmetic

Defines the theory [`LIA`] of linear integer arithmetic over matrices,
mixing integer and boolean matrices in a single signature.

A [`Sort`] value is either `Int(rows, cols)` or `Bool(rows, cols)`.
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
use theory::lia::{LIA, Sort};

// Pointwise less-than on scalars: Int(1,1), Int(1,1) -> Bool(1,1).
let i = Sort::Int([1, 1]);
let b = Sort::Bool([1, 1]);
assert!(LIA::Lt().check([i, i], [b]).is_ok());

// ReLU preserves shape and stays in the integer fragment.
let m = Sort::Int([3, 4]);
assert!(LIA::ReLU().check([m], [m]).is_ok());
assert!(LIA::ReLU().check([b], [b]).is_err());
```
*/

use crate::*;
use pyo3::pyclass;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Sort {
    Int([usize; 2]),
    Bool([usize; 2]),
}

impl Sort {
    pub fn is_bool(&self) -> bool {
        matches!(self, Sort::Bool(..))
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Sort::Int(..))
    }

    pub fn shape(&self) -> &[usize; 2] {
        match self {
            Sort::Bool(shape) | Sort::Int(shape) => shape,
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Int([i, j]) => write!(f, "Int({i}, {j})"),
            Sort::Bool([i, j]) => write!(f, "Bool({i}, {j})"),
        }
    }
}

#[derive(Clone, Debug)]
#[pyclass(frozen)]
pub enum LIA {
    // constants
    ConstInt(crate::PyTensor),
    ConstBool(crate::PyTensor),
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
    // XXX: should these be in LIA?
    ReLU(),
    Argmax(),
    Min(),
    Max(),
    // matrix operations
    Transpose(),
    // control flow
    Ite(),
    Id(),
    Uninterpreted(String),
}

impl fmt::Display for LIA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LIA::ConstInt(cm) => write!(f, "{}", cm),
            LIA::ConstBool(cm) => write!(f, "{}", cm),
            LIA::And() => write!(f, "And"),
            LIA::Or() => write!(f, "Or"),
            LIA::Xor() => write!(f, "Xor"),
            LIA::Not() => write!(f, "Not"),
            LIA::Le() => write!(f, "Le"),
            LIA::Lt() => write!(f, "Lt"),
            LIA::Ge() => write!(f, "Ge"),
            LIA::Gt() => write!(f, "Gt"),
            LIA::Eq() => write!(f, "Eq"),
            LIA::Ne() => write!(f, "Ne"),
            LIA::Linear(..) => write!(f, "Linear"),
            LIA::Add() => write!(f, "Add"),
            LIA::Sub() => write!(f, "Sub"),
            LIA::ReLU() => write!(f, "ReLU"),
            LIA::Argmax() => write!(f, "Argmax"),
            LIA::Min() => write!(f, "Min"),
            LIA::Max() => write!(f, "Max"),
            LIA::Transpose() => write!(f, "Transpose"),
            LIA::Ite() => write!(f, "Ite"),
            LIA::Id() => write!(f, "Id"),
            LIA::Uninterpreted(name) => write!(f, "Uninterpreted({name})"),
        }
    }
}

impl Theory for LIA {
    type Sort = Sort;
    const NAME: &'static str = "LIA";

    fn check<R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<Sort> + fmt::Display,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        match self {
            LIA::ConstInt(cm) => check_const(cm, read, write),
            LIA::ConstBool(_) | LIA::And() | LIA::Or() | LIA::Xor() | LIA::Not() => {
                check_bool(self, read, write)
            }
            LIA::Le() | LIA::Lt() | LIA::Ge() | LIA::Gt() | LIA::Eq() | LIA::Ne() => {
                check_cmp(self, read, write)
            }
            LIA::Linear(a, b) => {
                let mut read = read.into_iter();
                let mut write = write.into_iter();
                check_linear_affine(self, a, b, &mut read, &mut write)
            }
            LIA::Add() | LIA::Sub() | LIA::ReLU() | LIA::Argmax() | LIA::Min() | LIA::Max() => {
                check_mat_ops(self, read, write)
            }
            LIA::Transpose() => check_transpose(self, read, write),
            LIA::Ite() | LIA::Id() => check_flow(self, read, write),

            LIA::Uninterpreted(_) => {
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

fn check_const<R, W, D>(cm: &crate::PyTensor, read: R, write: W) -> Result<(), String>
where
    D: TryInto<Sort> + fmt::Display,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    if read.next().is_some() {
        return Err("Const: cannot read values".into());
    }
    let dtype = write_nxt(&mut write, 0, "LIA")?;
    match dtype {
        Sort::Int([i, j]) => {
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
        Sort::Bool(..) => {
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
    D: TryInto<Sort> + fmt::Display,
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
            let Sort::Bool([i, j]) = write_nxt(&mut write, 0, "LIA")? else {
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
        LIA::Not() => {
            let (r, w) = (
                read_nxt(&mut read, 0, "LIA")?,
                write_nxt(&mut write, 0, "LIA")?,
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
        LIA::And() | LIA::Or() | LIA::Xor() => {
            let w1 = write_nxt(&mut write, 0, "LIA")?;
            let (r1, r2, None) = (
                read_nxt(&mut read, 0, "LIA")?,
                read_nxt(&mut read, 1, "LIA")?,
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

fn check_cmp<R, W, D>(op: &LIA, read: R, write: W) -> Result<(), String>
where
    D: TryInto<Sort> + fmt::Display,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let r1 = read_nxt(&mut read, 0, "LIA")?;
    let r2 = read_nxt(&mut read, 1, "LIA")?;
    if r1 != r2 {
        return Err(format!("{:?}: input values must have the same type", op));
    }
    let shape = match r1 {
        Sort::Int(s) => s,
        _ => return Err(format!("{:?}: inputs must be Int matrices, got {r1}", op)),
    };
    let w1 = write_nxt(&mut write, 0, "LIA")?;
    if w1 != Sort::Bool(shape) {
        return Err(format!(
            "{:?}: output must be Bool({:?}), got {w1}",
            op, shape
        ));
    }
    Ok(())
}

fn check_mat_ops<R, W, D>(op: &LIA, read: R, write: W) -> Result<(), String>
where
    D: TryInto<Sort> + fmt::Display,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        LIA::Add() | LIA::Sub() => {
            let (r1, r2, None) = (
                read_nxt(&mut read, 0, "LIA")?,
                read_nxt(&mut read, 1, "LIA")?,
                read.next(),
            ) else {
                return Err(format!("{:?}: must read exactly two values", op));
            };
            let (w1, None) = (write_nxt(&mut write, 0, "LIA")?, write.next()) else {
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
            if !matches!(w1, Sort::Int(..)) {
                return Err(format!(
                    "{:?}: input and output values must be int matrices",
                    op
                ));
            }
            Ok(())
        }
        LIA::ReLU() => {
            let (r1, None) = (read_nxt(&mut read, 0, "LIA")?, read.next()) else {
                return Err(format!("{:?}: must read exactly one value", op));
            };
            let (w1, None) = (write_nxt(&mut write, 0, "LIA")?, write.next()) else {
                return Err(format!("{:?}: must write exactly one value", op));
            };
            if r1 != w1 {
                return Err(format!(
                    "{:?}: input and output must have the same type",
                    op
                ));
            }
            if !matches!(w1, Sort::Int(..)) {
                return Err(format!(
                    "{:?}: input and output values must be int matrices",
                    op
                ));
            }
            Ok(())
        }
        LIA::Argmax() | LIA::Min() | LIA::Max() => {
            let (_r1, None) = (read_nxt(&mut read, 0, "LIA")?, read.next()) else {
                return Err(format!("{:?}: must read exactly one value", op));
            };
            let (w1, None) = (write_nxt(&mut write, 0, "LIA")?, write.next()) else {
                return Err(format!("{:?}: must write exactly one value", op));
            };
            match w1 {
                Sort::Int([i, j]) => {
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
    a: &crate::PyTensor,
    b: &crate::PyTensor,
    read: &mut impl Iterator<Item = D>,
    write: &mut impl Iterator<Item = D>,
) -> Result<(), String>
where
    D: TryInto<Sort> + fmt::Display,
{
    let (r1, None) = (read_nxt(read, 0, "LIA")?, read.next()) else {
        return Err(format!("{:?}: must read exactly one value", op));
    };
    let (w1, None) = (write_nxt(write, 0, "LIA")?, write.next()) else {
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
        (Sort::Int([d1, d2]), Sort::Int([d3, d4])) => {
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
            Ok(())
        }
        _ => Err(format!("{:?}: input and output must be int matrices", op)),
    }
}

fn check_transpose<R, W, D>(op: &LIA, read: R, write: W) -> Result<(), String>
where
    D: TryInto<Sort> + fmt::Display,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    let (r1, None) = (read_nxt(&mut read, 0, "LIA")?, read.next()) else {
        return Err(format!("{:?}: must read exactly one value", op));
    };
    let (w1, None) = (write_nxt(&mut write, 0, "LIA")?, write.next()) else {
        return Err(format!("{:?}: must write exactly one value", op));
    };
    match (r1, w1) {
        (Sort::Int([d1, d2]), Sort::Int([e1, e2])) => {
            if d2 != e1 || d1 != e2 {
                return Err(format!(
                    "{:?}: transpose of {}x{} must produce {}x{}, got {}x{}",
                    op, d1, d2, d2, d1, e1, e2
                ));
            }
            Ok(())
        }
        _ => Err(format!("{:?}: input and output must be int matrices", op)),
    }
}

fn check_flow<R, W, D>(op: &LIA, read: R, write: W) -> Result<(), String>
where
    D: TryInto<Sort> + fmt::Display,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        LIA::Id() => {
            let (r1, None) = (read_nxt(&mut read, 0, "LIA")?, read.next()) else {
                return Err(format!("{:?}: must read exactly one value", op));
            };
            let (w1, None) = (write_nxt(&mut write, 0, "LIA")?, write.next()) else {
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
        LIA::Ite() => {
            let (r1, r2, r3, None) = (
                read_nxt(&mut read, 0, "LIA")?,
                read_nxt(&mut read, 1, "LIA")?,
                read_nxt(&mut read, 2, "LIA")?,
                read.next(),
            ) else {
                return Err(format!("{:?}: must read exactly three values", op));
            };
            let (w1, None) = (write_nxt(&mut write, 0, "LIA")?, write.next()) else {
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

    fn int(r: usize, c: usize) -> Sort {
        Sort::Int([r, c])
    }

    fn bool_t(r: usize, c: usize) -> Sort {
        Sort::Bool([r, c])
    }

    #[test]
    fn type_kind_and_shape() {
        assert!(int(2, 3).is_int() && !int(2, 3).is_bool());
        assert_eq!(int(2, 3).shape(), &[2, 3]);
        assert!(bool_t(1, 1).is_bool() && !bool_t(1, 1).is_int());
    }

    #[test]
    fn const_int_ok() {
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0i64, 1], [2, 3]]).into();
        assert!(
            LIA::ConstInt(cm)
                .check([] as [Sort; 0], [int(2, 2)])
                .is_ok()
        );
    }

    #[test]
    fn const_int_bool_write_fails() {
        assert!(
            LIA::ConstInt(tch::Tensor::from_slice2(&[[0i64]]).into())
                .check([] as [Sort; 0], [bool_t(1, 1)])
                .is_err()
        );
    }

    #[test]
    fn const_int_wrong_rows_fails() {
        assert!(
            LIA::ConstInt(tch::Tensor::from_slice2(&[[0i64]]).into())
                .check([] as [Sort; 0], [int(2, 1)])
                .is_err()
        );
    }

    #[test]
    fn const_int_with_read_fails() {
        let t = int(1, 1);
        assert!(
            LIA::ConstInt(tch::Tensor::from_slice2(&[[0i64]]).into())
                .check([t], [t])
                .is_err()
        );
    }

    #[test]
    fn const_bool_ok() {
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[true, false], [false, true]]).into();
        assert!(
            LIA::ConstBool(cm)
                .check([] as [Sort; 0], [bool_t(2, 2)])
                .is_ok()
        );
    }

    #[test]
    fn const_bool_int_write_fails() {
        assert!(
            LIA::ConstBool(tch::Tensor::from_slice2(&[[true]]).into())
                .check([] as [Sort; 0], [int(1, 1)])
                .is_err()
        );
    }

    #[test]
    fn not_ok() {
        let b = bool_t(2, 3);
        assert!(LIA::Not().check([b], [b]).is_ok());
    }

    #[test]
    fn not_int_input_fails() {
        let t = int(1, 1);
        assert!(LIA::Not().check([t], [t]).is_err());
    }

    #[test]
    fn and_ok() {
        let b = bool_t(2, 2);
        assert!(LIA::And().check([b, b], [b]).is_ok());
    }

    #[test]
    fn or_ok() {
        let b = bool_t(1, 1);
        assert!(LIA::Or().check([b, b], [b]).is_ok());
    }

    #[test]
    fn xor_ok() {
        let b = bool_t(3, 1);
        assert!(LIA::Xor().check([b, b], [b]).is_ok());
    }

    #[test]
    fn and_int_output_fails() {
        let b = bool_t(1, 1);
        assert!(LIA::And().check([b, b], [int(1, 1)]).is_err());
    }

    #[test]
    fn and_type_mismatch_fails() {
        assert!(
            LIA::And()
                .check([bool_t(1, 1), bool_t(1, 2)], [bool_t(1, 1)])
                .is_err()
        );
    }

    #[test]
    fn lt_ok() {
        assert!(
            LIA::Lt()
                .check([int(1, 1), int(1, 1)], [bool_t(1, 1)])
                .is_ok()
        );
    }

    #[test]
    fn le_ok() {
        let res = LIA::Le().check([int(2, 3), int(2, 3)], [bool_t(2, 3)]);
        assert!(res.is_ok(), "result: {:?}", res);
    }

    #[test]
    fn eq_ok() {
        assert!(
            LIA::Eq()
                .check([int(3, 2), int(3, 2)], [bool_t(3, 2)])
                .is_ok()
        );
        assert!(
            LIA::Eq()
                .check([int(3, 3), int(3, 2)], [bool_t(3, 2)])
                .is_err()
        );
        assert!(
            LIA::Eq()
                .check([int(3, 2), int(3, 2)], [bool_t(3, 3)])
                .is_err()
        );
    }

    #[test]
    fn cmp_non_bool_output_fails() {
        let t = int(1, 1);
        assert!(LIA::Lt().check([t, t], [t]).is_err());
    }

    #[test]
    fn cmp_input_mismatch_fails() {
        assert!(
            LIA::Eq()
                .check([int(1, 1), int(1, 2)], [bool_t(1, 1)])
                .is_err()
        );
    }

    #[test]
    fn add_ok() {
        let t = int(3, 4);
        assert!(LIA::Add().check([t, t], [t]).is_ok());
    }

    #[test]
    fn add_shape_mismatch_fails() {
        assert!(
            LIA::Add()
                .check([int(1, 2), int(2, 1)], [int(1, 2)])
                .is_err()
        );
    }

    #[test]
    fn add_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LIA::Add().check([b, b], [b]).is_err());
    }

    #[test]
    fn relu_ok() {
        let t = int(3, 4);
        assert!(LIA::ReLU().check([t], [t]).is_ok());
    }

    #[test]
    fn relu_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LIA::ReLU().check([b], [b]).is_err());
    }

    #[test]
    fn argmax_ok() {
        assert!(LIA::Argmax().check([int(3, 4)], [int(1, 4)]).is_ok());
    }

    #[test]
    fn argmax_matrix_output_fails() {
        assert!(LIA::Argmax().check([int(3, 4)], [int(3, 4)]).is_err());
    }

    #[test]
    fn min_ok() {
        assert!(LIA::Min().check([int(4, 1)], [int(1, 1)]).is_ok());
    }

    #[test]
    fn linear_ok() {
        // A=[2,3] maps 3 features to 2; X=[3,4] is 3 features × 4 batch items.
        // Convention: Y = A·X  →  Y=[2,4].
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Int64, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([0, 0], (tch::Kind::Int64, tch::Device::Cpu)).into();
        assert!(LIA::Linear(a, b).check([int(3, 4)], [int(2, 4)]).is_ok());
    }

    #[test]
    fn linear_with_bias_ok() {
        // A=[2,3], b=[2,1] column bias, X=[3,1] single sample → Y=[2,1].
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Int64, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([2, 1], (tch::Kind::Int64, tch::Device::Cpu)).into();
        assert!(LIA::Linear(a, b).check([int(3, 1)], [int(2, 1)]).is_ok());
    }

    #[test]
    fn linear_dim_mismatch_fails() {
        // A=[2,3] but X has 4 rows — inner dimension mismatch.
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Int64, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([0, 0], (tch::Kind::Int64, tch::Device::Cpu)).into();
        assert!(LIA::Linear(a, b).check([int(4, 1)], [int(2, 1)]).is_err());
    }

    #[test]
    fn transpose_ok() {
        assert!(LIA::Transpose().check([int(3, 4)], [int(4, 3)]).is_ok());
    }

    #[test]
    fn transpose_wrong_shape_fails() {
        assert!(LIA::Transpose().check([int(3, 4)], [int(3, 4)]).is_err());
    }

    #[test]
    fn ite_ok() {
        let t = int(3, 4);
        assert!(LIA::Ite().check([bool_t(1, 1), t, t], [t]).is_ok());
    }

    #[test]
    fn ite_non_bool_guard_fails() {
        let t = int(1, 1);
        assert!(LIA::Ite().check([t, t, t], [t]).is_err());
    }

    #[test]
    fn ite_arm_mismatch_fails() {
        assert!(
            LIA::Ite()
                .check([bool_t(1, 1), int(1, 1), int(1, 2)], [int(1, 1)])
                .is_err()
        );
    }

    #[test]
    fn id_ok() {
        let t = int(4, 4);
        assert!(LIA::Id().check([t], [t]).is_ok());
    }

    #[test]
    fn id_type_mismatch_fails() {
        assert!(LIA::Id().check([int(1, 1)], [int(2, 2)]).is_err());
    }
}
