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
let i = Type::Int(1, 1);
let b = Type::Bool(1, 1);
assert!(LIA::Lt.check(&[i, i], &[b]).is_ok());

// ReLU preserves shape and stays in the integer fragment.
let m = Type::Int(3, 4);
assert!(LIA::ReLU.check(&[m], &[m]).is_ok());
assert!(LIA::ReLU.check(&[b], &[b]).is_err());
```
*/

use std::fmt;

use crate::*;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Type {
    Int(usize, usize),
    Bool(usize, usize),
}

impl Type {
    pub fn is_bool(&self) -> bool {
        matches!(self, Type::Bool(..))
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Type::Int(..))
    }

    pub fn shape(&self) -> (usize, usize) {
        match self {
            Type::Bool(i, j) | Type::Int(i, j) => (*i, *j),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int(i, j) => write!(f, "Int({i}, {j})"),
            Type::Bool(i, j) => write!(f, "Bool({i}, {j})"),
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum LIA {
    // constants
    ConstInt(Vec<Vec<i64>>),
    ConstBool(Vec<Vec<bool>>),
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
    Linear(Vec<Vec<i64>>, Vec<Vec<i64>>),
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
            LIA::ConstInt(cm) => fmt_matrix(cm, f),
            LIA::ConstBool(cm) => fmt_matrix(cm, f),
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

fn fmt_matrix<T: fmt::Display>(cm: &[Vec<T>], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "[")?;
    for (i, row) in cm.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "[")?;
        for (j, v) in row.iter().enumerate() {
            if j > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v}")?;
        }
        write!(f, "]")?;
    }
    write!(f, "]")
}

impl Theory for LIA {
    type DType = Type;

    fn check<'a, R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<&'a Type>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
        Type: 'a,
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

fn check_const<'a, R, W, D>(cm: &[Vec<i64>], read: R, write: W) -> Result<(), String>
where
    D: TryInto<&'a Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
    Type: 'a,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    if read.next().is_some() {
        return Err("Const: cannot read values".into());
    }
    let dtype = write_nxt(&mut write, 0)?;
    match dtype {
        Type::Int(i, j) => {
            if cm.len() != *i {
                return Err(format!(
                    "ConstInt: initializer has wrong number of rows (has {}, expected {})",
                    cm.len(),
                    i
                ));
            }
            if cm.iter().any(|row| row.len() != *j) {
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

fn check_bool<'a, R, W, D>(op: &LIA, read: R, write: W) -> Result<(), String>
where
    D: TryInto<&'a Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
    Type: 'a,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        LIA::ConstBool(cm) => {
            if read.next().is_some() {
                return Err("ConstBool: cannot read values".into());
            }
            let Type::Bool(i, j) = write_nxt(&mut write, 0)? else {
                return Err("ConstBool: write type must be Bool".into());
            };
            if cm.len() != *i {
                return Err(format!(
                    "ConstBool: initializer has wrong number of rows (has {}, expected {})",
                    cm.len(),
                    i
                ));
            }
            if cm.iter().any(|row| row.len() != *j) {
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

fn check_cmp<'a, R, W, D>(op: &LIA, read: R, write: W) -> Result<(), String>
where
    D: TryInto<&'a Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
    Type: 'a,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    if *read_nxt(&mut read, 0)? != *read_nxt(&mut read, 1)? {
        return Err(format!("{:?}: input values must have the same type", op));
    }
    let Type::Bool(1, 1) = write_nxt(&mut write, 0)? else {
        return Err(format!(
            "{:?}: input and output values must have the same type",
            op
        ));
    };
    Ok(())
}

fn check_mat_ops<'a, R, W, D>(op: &LIA, read: R, write: W) -> Result<(), String>
where
    D: TryInto<&'a Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
    Type: 'a,
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

            if *r1 != *r2 {
                return Err(format!("{:?}: inputs must have the same type", op));
            }
            if *r1 != *w1 {
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
            if *r1 != *w1 {
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
                Type::Int(i, j) => {
                    // FIXME: we should fix which dimension is 1..
                    if *i == 1 || *j == 1 {
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

fn check_linear_affine<'a, D>(
    op: &LIA,
    a: &[Vec<i64>],
    b: &[Vec<i64>],
    read: &mut impl Iterator<Item = D>,
    write: &mut impl Iterator<Item = D>,
) -> Result<(), String>
where
    D: TryInto<&'a Type>,
    Type: 'a,
{
    let (r1, None) = (read_nxt(read, 0)?, read.next()) else {
        return Err(format!("{:?}: must read exactly one value", op));
    };
    let (w1, None) = (write_nxt(write, 0)?, write.next()) else {
        return Err(format!("{:?}: must write exactly one value", op));
    };

    let a_rows = a.len();
    if a_rows == 0 {
        return Err(format!("{:?}: `A` is empty", op));
    }
    let a_cols = a[0].len();
    if a.iter().any(|row| row.len() != a_cols) {
        return Err(format!(
            "{:?}: `A` has invalid dimensions, rows have different lengths",
            op
        ));
    }

    let b_rows = b.len();
    let mut b_cols: usize = 0;
    if b_rows != 0 {
        b_cols = b[0].len();
        if b.iter().any(|row| row.len() != b_cols) {
            return Err(format!(
                "{:?}: `B` has invalid dimensions, rows have different lengths",
                op
            ));
        }
        if b_rows != 1 && b_cols != 1 {
            return Err(format!(
                "{:?}: `B` has to be a vector, got matrix {}x{}",
                op, b_rows, b_cols
            ));
        }
    }

    match (r1, w1) {
        (Type::Int(d1, d2), Type::Int(d3, d4)) => {
            if *d2 != a_rows {
                return Err(format!(
                    "{:?}: mismatch in inner dimensions of `A` and `x`: A has {}x{}, x has {}x{}",
                    op, d1, d2, a_rows, a_rows
                ));
            }
            // `A*x` is a a_rows x d2 matrix, `B` has to have these dimensions (if non-empty)
            if b_rows > 0 && (a_rows != b_rows || *d2 != b_cols) {
                return Err(format!(
                    "{:?}: A*x has dimension {}x{} while B has {}x{}",
                    op, a_rows, d2, b_rows, b_cols
                ));
            }
            if a_rows != *d3 || *d2 != *d4 {
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

fn check_flow<'a, R, W, D>(op: &LIA, read: R, write: W) -> Result<(), String>
where
    D: TryInto<&'a Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
    Type: 'a,
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
            if *r1 != *w1 {
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
            if *r2 != *r3 {
                return Err(format!(
                    "{:?}: 2nd and 3rd inputs must have the same type",
                    op
                ));
            }
            if *w1 != *r2 {
                return Err(format!(
                    "{:?}: inputs and output must have the same type",
                    op
                ));
            }
            if *r1 != Type::Bool(1, 1) {
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
