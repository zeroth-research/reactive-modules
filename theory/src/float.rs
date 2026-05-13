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
    Op(Arith),
}

impl fmt::Display for ArithFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            ArithFloat::Const(cm) => fmt_matrix(cm, f),
            ArithFloat::Op(op) => op.fmt(f),
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

impl From<Arith> for ArithFloat {
    fn from(a: Arith) -> Self {
        ArithFloat::Op(a)
    }
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

            ArithFloat::Op(op) => op.type_check(read, write),
        }
    }
}
