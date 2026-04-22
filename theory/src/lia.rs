/*!
# Linear integer arithmetic

Defines the theory [`LIA`] of linear integer arithmetic over matrices,
mixing integer and boolean matrices in a single signature.

A [`DType`] value is either `Int(rows, cols)` or `Bool(rows, cols)`.
`DType` converts to and from [`int::IntDType`] and [`bool::PropDType`]
so that integer and propositional terms embed directly into `LIA`. The
operations in [`LIA`] are:

- [`LIA::Const`] — an integer matrix literal whose shape must match the
  declared (integer) write type.
- [`LIA::Bool`] — lifts any [`bool::Prop`] operation to act on the
  boolean fragment of `DType`.
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
use theory::lia::{LIA, DType, CmpOp, LinearOp};

// Pointwise less-than on scalars: Int(1,1), Int(1,1) -> Bool(1,1).
let i = DType::Int(1, 1);
let b = DType::Bool(1, 1);
assert!(LIA::Cmp(CmpOp::Lt).check::<DType>(&[i, i], &[b]).is_ok());

// ReLU preserves shape and stays in the integer fragment.
let m = DType::Int(3, 4);
assert!(LIA::Linear(LinearOp::ReLU).check::<DType>(&[m], &[m]).is_ok());
assert!(LIA::Linear(LinearOp::ReLU).check::<DType>(&[b], &[b]).is_err());
```
*/

use crate::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DType {
    Int(usize, usize),
    Bool(bool::Bool),
}

impl DType {
    pub fn is_bool(&self) -> bool {
        matches!(self, DType::Bool(_))
    }

    pub fn is_int(&self) -> bool {
        matches!(self, DType::Int(_, _))
    }

    pub fn shape(&self) -> (usize, usize) {
        match self {
            DType::Bool(b) => b.shape(),
            DType::Int(i, j) => (*i, *j),
        }
    }
}

// -- From DType to its subtypes --
impl From<int::DType> for DType {
    fn from(t: int::DType) -> Self {
        let int::DType::Int(i, j) = t;
        DType::Int(i, j)
    }
}

impl From<bool::Bool> for DType {
    fn from(b: bool::Bool) -> Self {
        DType::Bool(b)
    }
}

// -- From subtypes to DType --
impl TryFrom<DType> for int::DType {
    type Error = ();
    fn try_from(lia_t: DType) -> Result<int::DType, Self::Error> {
        match lia_t {
            DType::Int(i, j) => Ok(int::DType::Int(i, j)),
            _ => Err(()),
        }
    }
}

impl TryFrom<DType> for bool::Bool {
    type Error = ();

    fn try_from(lia_t: DType) -> Result<bool::Bool, Self::Error> {
        match lia_t {
            DType::Bool(b) => Ok(b),
            _ => Err(()),
        }
    }
}

impl<'a> TryFrom<&'a DType> for &'a bool::Bool {
    type Error = ();

    fn try_from(lia_t: &'a DType) -> Result<&'a bool::Bool, Self::Error> {
        match lia_t {
            DType::Bool(b) => Ok(b),
            _ => Err(()),
        }
    }
}

// XXX: this seems a bit hacky..
impl<'a, E> TryFrom<Result<&'a bool::Bool, E>> for &'a bool::Bool {
    type Error = E;

    fn try_from(lia_t: Result<&'a bool::Bool, E>) -> Result<&'a bool::Bool, Self::Error> {
        lia_t
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CmpOp {
    Le,
    Lt,
    Ge,
    Gt,
    Eq,
    Ne,
}

#[derive(Clone, PartialEq, Debug)]
pub enum LinearOp {
    // A*x + B where `A` and `B` are constants
    Linear(Vec<Vec<i64>>, Vec<Vec<i64>>),
    ReLU,
    Argmax,
    Min,
    Max,
}

#[derive(Clone, PartialEq, Debug)]
pub enum FlowOp {
    Ite,
    Id, // this could probably be in the top-level enum directly..
}

#[derive(Clone, PartialEq, Debug)]
pub enum LIA {
    Const(Vec<Vec<i64>>),
    Bool(bool::Prop),
    Linear(LinearOp),
    Cmp(CmpOp),
    Flow(FlowOp),
}

impl Theory for LIA {
    type DType = DType;

    fn type_check<'a, R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<&'a DType>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        match self {
            LIA::Const(cm) => {
                let mut read = read.into_iter();
                let mut write = write.into_iter();
                if read.next().is_some() {
                    return Err("Const: cannot read values".into());
                }
                let dtype = write_nxt(&mut write, 0)?;
                match dtype {
                    DType::Int(i, j) => int::check_init_dims(cm, *i, *j)?,
                    DType::Bool(_) => {
                        return Err("Const must be integer matrix, not boolean".into());
                    }
                }
                if write.next().is_some() {
                    return Err("Const: returns more than one value".into());
                }
                Ok(())
            }

            LIA::Bool(op) => {
                let r = read.into_iter().map(|d: D| {
                    d.try_into()
                        .map_err(|_| ())
                        .and_then(|d: &DType| d.try_into())
                        .map_err(|_| "Type was supposed to be Bool")
                });

                let w = write.into_iter().map(|d: D| {
                    d.try_into()
                        .map_err(|_| ())
                        .and_then(|d: &DType| d.try_into())
                        .map_err(|_| "Type was supposed to be Bool")
                });

                op.type_check(r, w)
            }
            LIA::Cmp(op) => {
                let mut read = read.into_iter();
                let mut write = write.into_iter();
                match op {
                    CmpOp::Le | CmpOp::Lt | CmpOp::Eq | CmpOp::Ne | CmpOp::Ge | CmpOp::Gt => {
                        if *read_nxt(&mut read, 0)? != *read_nxt(&mut read, 1)? {
                            return Err(format!(
                                "{:?}: input values must have the same type",
                                self
                            ));
                        }
                        let DType::Bool(bool::Bool(1, 1)) = write_nxt(&mut write, 0)? else {
                            return Err(format!(
                                "{:?}: input and output values must have the same type",
                                self
                            ));
                        };

                        Ok(())
                    }
                }
            }
            LIA::Linear(op) => {
                let mut read = read.into_iter();
                let mut write = write.into_iter();
                match op {
                    LinearOp::Linear(a, b) => {
                        let (r1, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                            return Err(format!("{:?}: must read exactly one value", self));
                        };
                        let (w1, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                            return Err(format!("{:?}: must write exactly one value", self));
                        };

                        // check A and B constants
                        let a_rows = a.len();
                        if a_rows == 0 {
                            return Err(format!("{:?}: `A` is empty", self));
                        }
                        let a_cols = a[0].len();
                        if a.iter().any(|row| row.len() != a_cols) {
                            return Err(format!(
                                "{:?}: `A` has invalid dimensions, rows have different lengths",
                                self
                            ));
                        }

                        let b_rows = b.len();
                        let mut b_cols: usize = 0;
                        if b_rows != 0 {
                            b_cols = b[0].len();
                            if b.iter().any(|row| row.len() != b_cols) {
                                return Err(format!(
                                    "{:?}: `A` has invalid dimensions, rows have different lengths",
                                    self
                                ));
                            }

                            if b_rows != 1 && b_cols != 1 {
                                return Err(format!(
                                    "{:?}: `B` has to be a vector, got matrix {}x{}",
                                    self, b_rows, b_cols
                                ));
                            }
                        }

                        match (r1, w1) {
                            (DType::Int(d1, d2), DType::Int(d3, d4)) => {
                                if *d2 != a_rows {
                                    return Err(format!(
                                        "{:?}: mismatch in inner dimensions of `A` and `x`: A has {}x{}, x has {}x{}",
                                        self, d1, d2, a_rows, a_rows
                                    ));
                                }
                                // `A*x` is a a_rows x d2 matrix, `B` has to have these dimensions (if non-empty)
                                if b_rows > 0 && (a_rows != b_rows || *d2 != b_cols) {
                                    return Err(format!(
                                        "{:?}: A*x has dimension {}x{} while B has {}x{}",
                                        self, a_rows, d2, b_rows, b_cols
                                    ));
                                }
                                if a_rows != *d3 || *d2 != *d4 {
                                    return Err(format!(
                                        "{:?}: bad output matrix dimensions, expected {}x{} but got {}x{}",
                                        self, a_rows, d2, d3, d4
                                    ));
                                }
                            }
                            // TODO: should we allow also boolean matrices?
                            _ => {
                                return Err(format!(
                                    "{:?}: input and output must be int matrices",
                                    self
                                ));
                            }
                        }
                        Ok(())
                    }
                    LinearOp::ReLU => {
                        let (r1, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                            return Err(format!("{:?}: must read exactly one value", self));
                        };
                        let (w1, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                            return Err(format!("{:?}: must write exactly one value", self));
                        };

                        if *r1 != *w1 {
                            return Err(format!(
                                "{:?}: input and output must have the same type",
                                self
                            ));
                        }

                        if !matches!(w1, DType::Int(_, _)) {
                            return Err(format!(
                                "{:?}: input and output values must be int matrices",
                                self
                            ));
                        }
                        Ok(())
                    }

                    LinearOp::Argmax | LinearOp::Min | LinearOp::Max => {
                        let (r1, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                            return Err(format!("{:?}: must read exactly one value", self));
                        };
                        let (w1, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                            return Err(format!("{:?}: must write exactly one value", self));
                        };

                        match w1 {
                            DType::Int(i, j) => {
                                // FIXME: we should fix which dimension is 1..
                                if *i == 1 || *j == 1 {
                                    return Ok(());
                                }
                                Err(format!(
                                    "{:?}: output must be a vector, got matrix {}x{}",
                                    self, i, j
                                ))
                            }
                            _ => Err(format!("{:?}: output must be integer matrix", self)),
                        }
                    }
                }
            }
            LIA::Flow(op) => {
                let mut read = read.into_iter();
                let mut write = write.into_iter();
                match op {
                    FlowOp::Id => {
                        let (r1, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                            return Err(format!("{:?}: must read exactly one value", self));
                        };
                        let (w1, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                            return Err(format!("{:?}: must write exactly one value", self));
                        };

                        if *r1 != *w1 {
                            return Err(format!(
                                "{:?}: input and output must have the same type",
                                self
                            ));
                        }
                    }
                    FlowOp::Ite => {
                        let (r1, r2, r3, None) = (
                            read_nxt(&mut read, 0)?,
                            read_nxt(&mut read, 1)?,
                            read_nxt(&mut read, 2)?,
                            read.next(),
                        ) else {
                            return Err(format!("{:?}: must read exactly three values", self));
                        };
                        let (w1, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                            return Err(format!("{:?}: must write exactly one value", self));
                        };

                        if *r2 != *r3 {
                            return Err(format!(
                                "{:?}: 2nd and 3rd inputs must have the same type",
                                self
                            ));
                        }

                        if *w1 != *r2 {
                            return Err(format!(
                                "{:?}: inputs and outputmust have the same type",
                                self
                            ));
                        }

                        if *r1 != DType::Bool(bool::Bool(1, 1)) {
                            return Err(format!(
                                "{:?}: input and output values must have the same type",
                                self
                            ));
                        }
                    }
                }

                Ok(())
            }
        }
    }
}
