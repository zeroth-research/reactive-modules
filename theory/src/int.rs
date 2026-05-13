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

impl MatrixType for Int {
    fn shape(&self) -> (usize, usize) {
        (self.0, self.1)
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
    Op(Arith),
}

impl fmt::Display for ArithInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            ArithInt::Const(cm) => fmt_matrix(cm, f),
            ArithInt::Op(op) => op.fmt(f),
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

impl From<Arith> for ArithInt {
    fn from(a: Arith) -> Self {
        ArithInt::Op(a)
    }
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
            ArithInt::Op(op) => op.type_check(read, write),
        }
    }
}
