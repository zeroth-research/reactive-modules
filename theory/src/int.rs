/*!
# Int numbers

Defines the theory [`Int`] of matrices of signed integers (`i64`).

A [`DType`] value describes the *type* of a term as a matrix shape
`Int(rows, cols)`. The operations in [`Int`] are:

- [`Int::Const`] — an inline integer matrix literal whose shape must
  match the declared write type.
- [`Int::Id`] — unary, shape-preserving.
- [`Int::Add`], [`Int::Mul`] — elementwise binary; both inputs and the
  output share the same shape.
- [`Int::MatMul`] — matrix multiplication: `(m,k) × (k,n) → (m,n)`.

`Int` implements [`Theory`], so [`Theory::check`] type-checks a term by
validating the read/write argument shapes against the operation.

## Examples

```
use theory::Theory;
use theory::int::{Int, ArithInt};

let a = Int(2, 3);
let b = Int(3, 4);
let c = Int(2, 4);

// Matrix multiply: (2x3) * (3x4) -> (2x4).
assert!(ArithInt::MatMul.check::<DType>(&[a, b], &[c]).is_ok());

// Elementwise `Add` requires matching shapes.
let m = Int(2, 3);
assert!(ArithInt::Add.check::<DType>(&[m, m], &[m]).is_ok());
assert!(ArithInt::Add.check::<DType>(&[a, b], &[c]).is_err());
```
*/

use crate::*;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Int(pub usize, pub usize);

impl Int {
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Int(i, j) => (*i, *j),
        }
    }
}

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 1 {
            write!(f, "Int({})", self.1)
        } else {
            write!(f, "Int({}, {})", self.0, self.1)
        }
    }
}

#[derive(Clone, PartialEq, Debug, Eq)]
pub enum ArithInt {
    // TODO: use bigint?
    Const(Vec<Vec<i64>>),
    Add,
    Mul,
    Sub,
    Div,
    Mod,
    Neg,
    MatMul,
    Abs,
}

impl fmt::Display for ArithInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            ArithInt::Add => write!(f, "Add"),
            ArithInt::Mul => write!(f, "Mul"),
            ArithInt::Sub => write!(f, "Sub"),
            ArithInt::Div => write!(f, "Div"),
            ArithInt::Mod => write!(f, "Mod"),
            ArithInt::Neg => write!(f, "Neg"),
            ArithInt::Abs => write!(f, "Abs"),
            ArithInt::MatMul => write!(f, "MatMul"),
            ArithInt::Const(cm) => fmt_matrix(cm, f),
        }
    }
}

pub(crate) fn check_init_dims(cm: &[Vec<i64>], i: usize, j: usize) -> Result<(), String> {
    if cm.len() != i {
        return Err(format!(
            "Const: Initializer has a wrong number of rows (has {}, expected {})",
            cm.len(),
            i
        ));
    }
    if cm.iter().any(|row| row.len() != j) {
        return Err(format!(
            "Const: some column of initializer has wrong dimension, expected {}",
            j
        ));
    }
    Ok(())
}

impl Theory for ArithInt {
    type DType = Int;

    fn type_check<'a, R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<&'a Int>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        let mut read = read.into_iter();
        let mut write = write.into_iter();
        match self {
            ArithInt::Const(cm) => {
                if read.next().is_some() {
                    return Err("Const: cannot read values".into());
                }

                let Int(i, j) = write_nxt(&mut write, 0)?;

                check_init_dims(cm, *i, *j)?;

                if write.next().is_some() {
                    return Err("Const: returns more than one value".into());
                }
                Ok(())
            }

            ArithInt::Neg | ArithInt::Abs => {
                let (r, w) = (read_nxt(&mut read, 0)?, write_nxt(&mut write, 0)?);
                if r != w {
                    return Err(format!(
                        "{:?}: input and output type must be the same",
                        self
                    ));
                }
                if read.next().is_some() {
                    return Err(format!("{:?}: must read a single value (reads more)", self));
                }
                if write.next().is_some() {
                    return Err(format!(
                        "{:?}: must write a single value (writes more)",
                        self
                    ));
                }
                Ok(())
            }

            ArithInt::Add | ArithInt::Mul | ArithInt::Sub | ArithInt::Mod | ArithInt::Div => {
                let w1 = write_nxt(&mut write, 0)?;
                let (r1, r2, None) = (
                    read_nxt(&mut read, 0)?,
                    read_nxt(&mut read, 1)?,
                    read.next(),
                ) else {
                    return Err(format!("{:?}: must read exactly two values", self));
                };
                if r1 != r2 {
                    return Err(format!("{:?}: input values must have the same type", self));
                }
                if w1 != r1 {
                    return Err(format!(
                        "{:?}: input and output values must have the same type",
                        self
                    ));
                }
                Ok(())
            }

            ArithInt::MatMul => {
                let (r1, r2, None) = (
                    read_nxt(&mut read, 0)?,
                    read_nxt(&mut read, 1)?,
                    read.next(),
                ) else {
                    return Err(format!("{:?}: must read exactly two values", self));
                };
                let (w1, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                    return Err(format!("{:?}: must write exactly one value", self));
                };

                let (d1, d2) = r1.shape();
                let (d3, d4) = r2.shape();
                let (d5, d6) = w1.shape();

                if d2 != d3 {
                    return Err(format!(
                        "{:?}: mismatch in inner dimensions of input matrices: {} != {}",
                        self, d2, d3
                    ));
                }
                if d1 != d5 {
                    return Err(format!(
                        "{:?}: mismatch in first input and output dimensions: {} != {}",
                        self, d1, d5
                    ));
                }

                if d4 != d6 {
                    return Err(format!(
                        "{:?}: mismatch in second input and output dimensions: {} != {}",
                        self, d4, d6
                    ));
                }
                Ok(())
            }
        }
    }
}
