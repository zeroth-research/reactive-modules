/*!
# Float numbers

Defines the theory [`Float`] of matrices of signed integers (`i64`).

A [`DType`] value describes the *type* of a term as a matrix shape
`Float(rows, cols)`. The operations in [`Float`] are:

- [`Float::Const`] — an inline integer matrix literal whose shape must
  match the declared write type.
- [`Float::Id`] — unary, shape-preserving.
- [`Float::Add`], [`Float::Mul`] — elementwise binary; both inputs and the
  output share the same shape.
- [`Float::MatMul`] — matrix multiplication: `(m,k) × (k,n) → (m,n)`.

`Float` implements [`Theory`], so [`Theory::check`] type-checks a term by
validating the read/write argument shapes against the operation.

## Examples

```
use theory::Theory;
use theory::int::{Float, ArithFloat};

let a = Float(2, 3);
let b = Float(3, 4);
let c = Float(2, 4);

// Matrix multiply: (2x3) * (3x4) -> (2x4).
assert!(ArithFloat::MatMul.check::<DType>(&[a, b], &[c]).is_ok());

// Elementwise `Add` requires matching shapes.
let m = Float(2, 3);
assert!(ArithFloat::Add.check::<DType>(&[m, m], &[m]).is_ok());
assert!(ArithFloat::Add.check::<DType>(&[a, b], &[c]).is_err());
```
*/

use crate::*;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Float(pub usize, pub usize);

impl Float {
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Float(i, j) => (*i, *j),
        }
    }
}

impl MatrixType for Float {
    fn shape(&self) -> (usize, usize) {
        (self.0, self.1)
    }
}

impl fmt::Display for Float {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 1 {
            write!(f, "Float({})", self.1)
        } else {
            write!(f, "Float({}, {})", self.0, self.1)
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum ArithFloat {
    Const(Vec<Vec<f64>>),
    Add,
    Mul,
    Sub,
    Div,
    Mod,
    Neg,
    MatMul,
    Abs,
}

impl fmt::Display for ArithFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            ArithFloat::Add => write!(f, "Add"),
            ArithFloat::Mul => write!(f, "Mul"),
            ArithFloat::Sub => write!(f, "Sub"),
            ArithFloat::Div => write!(f, "Div"),
            ArithFloat::Mod => write!(f, "Mod"),
            ArithFloat::Neg => write!(f, "Neg"),
            ArithFloat::Abs => write!(f, "Abs"),
            ArithFloat::MatMul => write!(f, "MatMul"),
            ArithFloat::Const(cm) => fmt_matrix(cm, f),
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

impl Theory for ArithFloat {
    type DType = Float;

    fn type_check<'a, R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<&'a Float>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        let mut read = read.into_iter();
        let mut write = write.into_iter();
        match self {
            ArithFloat::Const(cm) => {
                if read.next().is_some() {
                    return Err("Const: cannot read values".into());
                }

                let Float(i, j) = write_nxt(&mut write, 0)?;

                check_init_dims(cm, *i, *j)?;

                if write.next().is_some() {
                    return Err("Const: returns more than one value".into());
                }
                Ok(())
            }

            ArithFloat::Neg | ArithFloat::Abs => {
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

            ArithFloat::Add
            | ArithFloat::Mul
            | ArithFloat::Sub
            | ArithFloat::Mod
            | ArithFloat::Div => {
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

            ArithFloat::MatMul => {
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
