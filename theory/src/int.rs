/*!
# Int numbers

Defines the theory [`Int`] of matrices of signed integers (`i64`).

A [`IntDType`] value describes the *type* of a term as a matrix shape
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
use theory::int::{Int, IntDType};

let a = IntDType::Int(2, 3);
let b = IntDType::Int(3, 4);
let c = IntDType::Int(2, 4);

// Matrix multiply: (2x3) * (3x4) -> (2x4).
assert!(Int::MatMul.check::<IntDType>(&[a, b], &[c]).is_ok());

// Elementwise `Add` requires matching shapes.
let m = IntDType::Int(2, 3);
assert!(Int::Add.check::<IntDType>(&[m, m], &[m]).is_ok());
assert!(Int::Add.check::<IntDType>(&[a, b], &[c]).is_err());
```
*/

use crate::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum IntDType {
    Int(usize, usize),
}

#[derive(Clone, PartialEq, Debug)]
pub enum Int {
    // TODO: use bigint?
    Const(Vec<Vec<i64>>),
    Add,
    Mul,
    MatMul,
    Id,
}

impl Theory for Int {
    type DType = IntDType;

    fn _check(&self, read: &[Self::DType], write: &[Self::DType]) -> Result<(), String> {
        match self {
            Int::Const(cm) => {
                if read.len() > 0 {
                    return Err("Const: cannot read values".into());
                }
                if write.len() != 1 {
                    return Err("Const: must return a single value".into());
                }
                match write[0] {
                    IntDType::Int(i, j) => {
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
                }
            }
            Int::Id => {
                if read.len() != 1 {
                    return Err(format!(
                        "{:?}: must read a single value, got {}",
                        self,
                        read.len()
                    ));
                }
                if write.len() != 1 {
                    return Err(format!(
                        "{:?}: must write a single value, got {}",
                        self,
                        write.len()
                    ));
                }
                if write[0] != read[0] {
                    return Err(format!(
                        "{:?}: input and output type must be the same",
                        self
                    ));
                }
                Ok(())
            }
            Int::Add | Int::Mul => {
                if read.len() != 2 {
                    return Err(format!(
                        "{:?}: must read two values, got {}",
                        self,
                        read.len()
                    ));
                }
                if write.len() != 1 {
                    return Err(format!(
                        "{:?}: must write a single value, got {}",
                        self,
                        write.len()
                    ));
                }
                if read[0] != read[1] {
                    return Err(format!("{:?}: input values must have the same type", self));
                }
                if write[0] != read[1] {
                    return Err(format!(
                        "{:?}: input and output values must have the same type",
                        self
                    ));
                }
                Ok(())
            }
            Int::MatMul => {
                if read.len() != 2 {
                    return Err(format!(
                        "{:?}: must read two values, got {}",
                        self,
                        read.len()
                    ));
                }
                if write.len() != 1 {
                    return Err(format!(
                        "{:?}: must write a single value, got {}",
                        self,
                        write.len()
                    ));
                }
                match (&read[0], &read[1], &write[0]) {
                    (IntDType::Int(d1, d2), IntDType::Int(d3, d4), IntDType::Int(d5, d6)) => {
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
                    }
                }
                Ok(())
            }
        }
    }
}
