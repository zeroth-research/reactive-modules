/*!
# Linear integer arithmetic

Defines the theory [`LIA`] of linear integer arithmetic over matrices,
mixing integer and boolean matrices in a single signature.

A [`Type`] value is either `Int(rows, cols)` or `Bool(rows, cols)`.
`Type` converts to and from [`int::IntType`] and [`bool::BoolOpType`]
so that integer and propositional terms embed directly into `LIA`. The
operations in [`LIA`] are:

- [`LIA::Const`] — an integer matrix literal whose shape must match the
  declared (integer) write type.
- [`LIA::Bool`] — lifts any [`bool::BoolOp`] operation to act on the
  boolean fragment of `Type`.
- [`LIA::Cmp`] — pointwise integer comparisons
  ([`CmpOp::Le`], [`CmpOp::Lt`], [`CmpOp::Ge`], [`CmpOp::Gt`],
  [`CmpOp::Eq`], [`CmpOp::Ne`]) producing a scalar `Bool(1,1)`.
- [`FlowOp::Ite`] — if-then-else: reads a boolean guard and two same-typed
  branches.
- [`LIA::Linear`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and
  `B` constant integer matrices of compatible shapes.
- [`LinearOp::ReLU`] — the shape-preserving rectified-linear map on integer
  matrices.

`LIA` implements [`Theory`]; [`Theory::check`] validates read/write
shapes against the selected operation.

## Examples

```
use theory::Theory;
use theory::lia::{LIA, Type, CmpOp, LinearOp};

// Pointwise less-than on scalars: Int(1,1), Int(1,1) -> Bool(1,1).
let i = Type::Int(1, 1);
let b = Type::Bool(1, 1);
assert!(LIA::Cmp(CmpOp::Lt).check::<Type>(&[i, i], &[b]).is_ok());

// ReLU preserves shape and stays in the integer fragment.
let m = Type::Int(3, 4);
assert!(LIA::Linear(LinearOp::ReLU).check::<Type>(&[m], &[m]).is_ok());
assert!(LIA::Linear(LinearOp::ReLU).check::<Type>(&[b], &[b]).is_err());
```
*/

use std::fmt;

use crate::*;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Type {
    Int(int::Int),
    Bool(bool::Bool),
}

impl Type {
    pub fn is_bool(&self) -> bool {
        matches!(self, Type::Bool(_))
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Type::Int(_))
    }

    pub fn shape(&self) -> (usize, usize) {
        match self {
            Type::Bool(b) => b.shape(),
            Type::Int(i) => i.shape(),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int(t) => fmt::Display::fmt(t, f),
            Type::Bool(t) => fmt::Display::fmt(t, f),
        }
    }
}

// -- From subtypes to Type --
impl From<int::Int> for Type {
    fn from(t: int::Int) -> Self {
        Type::Int(t)
    }
}

impl From<bool::Bool> for Type {
    fn from(b: bool::Bool) -> Self {
        Type::Bool(b)
    }
}

// -- From Type to its subtypes --
impl TryFrom<Type> for int::Int {
    type Error = ();
    fn try_from(lia_t: Type) -> Result<int::Int, Self::Error> {
        match lia_t {
            Type::Int(i) => Ok(i),
            _ => Err(()),
        }
    }
}

impl TryFrom<Type> for bool::Bool {
    type Error = ();

    fn try_from(lia_t: Type) -> Result<bool::Bool, Self::Error> {
        match lia_t {
            Type::Bool(b) => Ok(b),
            _ => Err(()),
        }
    }
}

// -- From Type to its subtypes, references --
impl<'a> TryFrom<&'a Type> for &'a bool::Bool {
    type Error = ();

    fn try_from(lia_t: &'a Type) -> Result<&'a bool::Bool, Self::Error> {
        match lia_t {
            Type::Bool(b) => Ok(b),
            _ => Err(()),
        }
    }
}

impl<'a> TryFrom<&'a Type> for &'a int::Int {
    type Error = ();

    fn try_from(lia_t: &'a Type) -> Result<&'a int::Int, Self::Error> {
        match lia_t {
            Type::Int(i) => Ok(i),
            _ => Err(()),
        }
    }
}

// XXX: this seems a bit hacky.. but that's just an "identity" that we need
// for calling type checking on sub-types
impl<'a, E> TryFrom<Result<&'a bool::Bool, E>> for &'a bool::Bool {
    type Error = E;

    fn try_from(lia_t: Result<&'a bool::Bool, E>) -> Result<&'a bool::Bool, Self::Error> {
        lia_t
    }
}

impl<'a, E> TryFrom<Result<&'a int::Int, E>> for &'a int::Int {
    type Error = E;

    fn try_from(lia_t: Result<&'a int::Int, E>) -> Result<&'a int::Int, Self::Error> {
        lia_t
    }
}

pub use crate::CmpOp;

#[derive(Clone, PartialEq, Debug)]
pub enum LinearOp {
    // A*x + B where `A` and `B` are constants
    Linear(Vec<Vec<i64>>, Vec<Vec<i64>>),
    Add,
    // TODO: where to put this one?
    ReLU,
    // TODO: move these to MatOp
    Argmax,
    Min,
    Max,
    //Neg
}

#[derive(Clone, PartialEq, Debug)]
pub enum FlowOp {
    Ite,
    Id, // this could probably be in the top-level enum directly..
}

#[derive(Clone, PartialEq, Debug)]
pub enum LIA {
    Const(Vec<Vec<i64>>),
    Bool(bool::BoolOp),
    Linear(LinearOp),
    Cmp(CmpOp),
    Flow(FlowOp),
}

impl fmt::Display for LIA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LIA::Const(cm) => fmt_matrix(cm, f),
            LIA::Bool(op) => match op {
                bool::BoolOp::Const(cm) => fmt_matrix(cm, f),
                bool::BoolOp::And => write!(f, "And"),
                bool::BoolOp::Or => write!(f, "Or"),
                bool::BoolOp::Xor => write!(f, "Xor"),
                bool::BoolOp::Not => write!(f, "Not"),
                bool::BoolOp::Xnor => write!(f, "Xnor"),
                bool::BoolOp::Implies => write!(f, "Implies"),
            },
            LIA::Cmp(op) => match op {
                CmpOp::Le => write!(f, "Le"),
                CmpOp::Lt => write!(f, "Lt"),
                CmpOp::Ge => write!(f, "Ge"),
                CmpOp::Gt => write!(f, "Gt"),
                CmpOp::Eq => write!(f, "Eq"),
                CmpOp::Ne => write!(f, "Ne"),
            },
            LIA::Linear(op) => match op {
                LinearOp::Linear(_, _) => write!(f, "Linear"),
                LinearOp::Add => write!(f, "Add"),
                LinearOp::ReLU => write!(f, "ReLU"),
                LinearOp::Argmax => write!(f, "Argmax"),
                LinearOp::Min => write!(f, "Min"),
                LinearOp::Max => write!(f, "Max"),
            },
            LIA::Flow(op) => match op {
                FlowOp::Id => write!(f, "Id"),
                FlowOp::Ite => write!(f, "Ite"),
            },
        }
    }
}

impl Theory for LIA {
    type DType = Type;

    fn type_check<'a, R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<&'a Type>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
        Type: 'a,
    {
        match self {
            LIA::Const(cm) => check_const(cm, read, write),
            LIA::Bool(op) => check_bool(op, read, write),
            LIA::Cmp(op) => op.type_check(read, write),
            LIA::Linear(op) => check_linear(op, read, write),
            LIA::Flow(op) => check_flow(op, read, write),
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
        Type::Int(int::Int(i, j)) => int::check_init_dims(cm, *i, *j)?,
        Type::Bool(_) => {
            return Err("Const must be integer matrix, not boolean".into());
        }
    }
    if write.next().is_some() {
        return Err("Const: returns more than one value".into());
    }
    Ok(())
}

fn check_bool<'a, R, W, D>(op: &bool::BoolOp, read: R, write: W) -> Result<(), String>
where
    D: TryInto<&'a Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
    Type: 'a,
{
    let r = read.into_iter().map(|d: D| {
        d.try_into()
            .map_err(|_| ())
            .and_then(|d: &Type| d.try_into())
            .map_err(|_| "Type was supposed to be Bool")
    });
    let w = write.into_iter().map(|d: D| {
        d.try_into()
            .map_err(|_| ())
            .and_then(|d: &Type| d.try_into())
            .map_err(|_| "Type was supposed to be Bool")
    });
    op.type_check(r, w)
}

fn check_linear<'a, R, W, D>(op: &LinearOp, read: R, write: W) -> Result<(), String>
where
    D: TryInto<&'a Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
    Type: 'a,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        LinearOp::Linear(a, b) => check_linear_affine(op, a, b, &mut read, &mut write),
        LinearOp::Add => {
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
            if !matches!(w1, Type::Int(_)) {
                return Err(format!(
                    "{:?}: input and output values must be int matrices",
                    op
                ));
            }
            Ok(())
        }
        LinearOp::ReLU => {
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
            if !matches!(w1, Type::Int(_)) {
                return Err(format!(
                    "{:?}: input and output values must be int matrices",
                    op
                ));
            }
            Ok(())
        }
        LinearOp::Argmax | LinearOp::Min | LinearOp::Max => {
            let (_r1, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                return Err(format!("{:?}: must read exactly one value", op));
            };
            let (w1, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                return Err(format!("{:?}: must write exactly one value", op));
            };
            match w1 {
                Type::Int(int::Int(i, j)) => {
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
    }
}

fn check_linear_affine<'a, D>(
    op: &LinearOp,
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
        (Type::Int(int::Int(d1, d2)), Type::Int(int::Int(d3, d4))) => {
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

fn check_flow<'a, R, W, D>(op: &FlowOp, read: R, write: W) -> Result<(), String>
where
    D: TryInto<&'a Type>,
    R: IntoIterator<Item = D>,
    W: IntoIterator<Item = D>,
    Type: 'a,
{
    let mut read = read.into_iter();
    let mut write = write.into_iter();
    match op {
        FlowOp::Id => {
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
        FlowOp::Ite => {
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
            if *r1 != Type::Bool(bool::Bool(1, 1)) {
                return Err(format!(
                    "{:?}: input and output values must have the same type",
                    op
                ));
            }
            Ok(())
        }
    }
}
