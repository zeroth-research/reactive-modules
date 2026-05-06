/*!
# Real numbers

Defines the theory [`Real`] of matrices of signed integers (`i64`).

A [`DType`] value describes the *type* of a term as a matrix shape
`Real(rows, cols)`. The operations in [`Real`] are:

- [`Real::Const`] — an inline integer matrix literal whose shape must
  match the declared write type.
- [`Real::Id`] — unary, shape-preserving.
- [`Real::Add`], [`Real::Mul`] — elementwise binary; both inputs and the
  output share the same shape.
- [`Real::MatMul`] — matrix multiplication: `(m,k) × (k,n) → (m,n)`.

`Real` implements [`Theory`], so [`Theory::check`] type-checks a term by
validating the read/write argument shapes against the operation.

## Examples

```
use theory::Theory;
use theory::int::{Real, ArithReal};

let a = Real(2, 3);
let b = Real(3, 4);
let c = Real(2, 4);

// Matrix multiply: (2x3) * (3x4) -> (2x4).
assert!(ArithReal::MatMul.check::<DType>(&[a, b], &[c]).is_ok());

// Elementwise `Add` requires matching shapes.
let m = Real(2, 3);
assert!(ArithReal::Add.check::<DType>(&[m, m], &[m]).is_ok());
assert!(ArithReal::Add.check::<DType>(&[a, b], &[c]).is_err());
```
*/

use crate::*;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Real(pub usize, pub usize);

impl Real {
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Real(i, j) => (*i, *j),
        }
    }
}

impl fmt::Display for Real {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 1 {
            write!(f, "Real({})", self.1)
        } else {
            write!(f, "Real({}, {})", self.0, self.1)
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum ArithReal {
    Const(Vec<Vec<f64>>),
    Add,
    Mul,
    Sub,
    Div,
    Mod,
    Neg,
    MatMul,
    Abs,
    // transcendental functions
    Sin,
    Cos,
}

impl fmt::Display for ArithReal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            ArithReal::Add => write!(f, "Add"),
            ArithReal::Mul => write!(f, "Mul"),
            ArithReal::Sub => write!(f, "Sub"),
            ArithReal::Div => write!(f, "Div"),
            ArithReal::Mod => write!(f, "Mod"),
            ArithReal::Neg => write!(f, "Neg"),
            ArithReal::Abs => write!(f, "Abs"),
            ArithReal::MatMul => write!(f, "MatMul"),
            ArithReal::Sin => write!(f, "Sin"),
            ArithReal::Cos => write!(f, "Cos"),
            ArithReal::Const(cm) => fmt_matrix(cm, f),
        }
    }
}

pub(crate) fn check_init_dims(cm: &[Vec<f64>], i: usize, j: usize) -> Result<(), String> {
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

impl Theory for ArithReal {
    type DType = Real;

    fn type_check<'a, R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<&'a Real>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        let mut read = read.into_iter();
        let mut write = write.into_iter();
        match self {
            ArithReal::Const(cm) => {
                if read.next().is_some() {
                    return Err("Const: cannot read values".into());
                }

                let Real(i, j) = write_nxt(&mut write, 0)?;

                check_init_dims(cm, *i, *j)?;

                if write.next().is_some() {
                    return Err("Const: returns more than one value".into());
                }
                Ok(())
            }

            ArithReal::Neg | ArithReal::Abs | ArithReal::Sin | ArithReal::Cos => {
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

            ArithReal::Add | ArithReal::Mul | ArithReal::Sub | ArithReal::Mod | ArithReal::Div => {
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

            ArithReal::MatMul => {
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
