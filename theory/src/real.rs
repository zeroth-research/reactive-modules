/*!
# Real numbers

Defines the theory [`Real`] of matrices of real numbers (represented as
`f64`).

A [`RealDType`] value describes the *type* of a term as a matrix shape
`Real(rows, cols)`. The operations in [`Real`] are:

- [`Real::Const`] — an inline real matrix literal whose shape must match
  the declared write type.
- [`Real::Id`] — unary, shape-preserving.
- [`Real::Add`], [`Real::Mul`] — elementwise binary; both inputs and the output share
  the same shape.
- [`Real::MatMul`] — matrix multiplication: `(m,k) × (k,n) → (m,n)`.

`Real` implements [`Theory`], so [`Theory::check`] type-checks a term by
validating the read/write argument shapes against the operation.

## Examples

```
use theory::Theory;
use theory::real::{Real, RealDType};

let a = RealDType::Real(2, 3);
let b = RealDType::Real(3, 4);
let c = RealDType::Real(2, 4);

// Matrix multiply: (2x3) * (3x4) -> (2x4).
assert!(Real::MatMul.check::<RealDType>(&[a, b], &[c]).is_ok());

// Elementwise `Add` requires matching shapes.
let m = RealDType::Real(2, 3);
assert!(Real::Add.check::<RealDType>(&[m, m], &[m]).is_ok());
assert!(Real::Add.check::<RealDType>(&[a, b], &[c]).is_err());
```
*/

use crate::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum RealDType {
    Real(usize, usize),
}

#[derive(Clone, PartialEq, Debug)]
pub enum Real {
    // TODO: use String or rationals?
    Const(Vec<Vec<f64>>),
    Add,
    Mul,
    MatMul,
    Neg,
    Id,
}

impl Theory for Real {
    type DType = RealDType;

    fn _check(&self, read: &[Self::DType], write: &[Self::DType]) -> Result<(), String> {
        match self {
            Real::Const(cm) => {
                if read.len() > 0 {
                    return Err("Const: cannot read values".into());
                }
                if write.len() != 1 {
                    return Err("Const: must return a single value".into());
                }
                match write[0] {
                    RealDType::Real(i, j) => {
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
            Real::Neg | Real::Id => {
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
            Real::Add | Real::Mul => {
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
            Real::MatMul => {
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
                    (RealDType::Real(d1, d2), RealDType::Real(d3, d4), RealDType::Real(d5, d6)) => {
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
