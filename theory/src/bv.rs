/*!
# Bit-vectors of fixed length

Defines the theory [`BV<N>`] of matrices of `N`-bit bit-vectors, where
the width `N` is fixed at compile time via a const generic.

A [`BVDType<N>`] value describes the *type* of a term as a matrix shape
`BV(rows, cols)` whose elements are `N`-bit bit-vectors. The operations
in [`BV`] are:

- [`BV::Const`] — an inline bit-vector matrix literal whose shape must
  match the declared write type.
- [`BV::Id`], [`BV::Not`] — unary, shape-preserving.
- [`BV::And`], [`BV::Or`], [`BV::Xor`], [`BV::Add`], [`BV::Mul`] —
  elementwise binary; both inputs and the output share the same shape.
- [`BV::MatMul`] — matrix multiplication: `(m,k) × (k,n) → (m,n)`, with
  the inner dimensions required to agree.

`BV<N>` implements [`Theory`], so [`Theory::check`] type-checks terms by
validating read/write argument shapes against the operation.

## Examples

```
use theory::Theory;
use theory::bv::{BV, BVDType};

// 8-bit bit-vectors.
let a = BVDType::<8>::BV(2, 3);
let b = BVDType::<8>::BV(3, 4);
let c = BVDType::<8>::BV(2, 4);

// Matrix multiply: (2x3) * (3x4) -> (2x4).
assert!(BV::<8>::MatMul.check::<BVDType<8>>(&[a, b], &[c]).is_ok());

// Elementwise `Add` requires matching shapes.
let m = BVDType::<8>::BV(2, 3);
assert!(BV::<8>::Add.check::<BVDType<8>>(&[m, m], &[m]).is_ok());
assert!(BV::<8>::Add.check::<BVDType<8>>(&[a, b], &[c]).is_err());
```
*/

use crate::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BVDType<const N: usize> {
    // matrix of bitvectors of length `N`
    BV(usize, usize),
}

/// TODO: write "formal" semantics
#[derive(Clone, PartialEq, Debug)]
pub enum BV<const N: usize> {
    // TODO: use bitarray, this works only for `N <= 64`
    Const(Vec<Vec<usize>>),
    Add,
    Mul,
    MatMul,
    And,
    Or,
    Xor,
    Not,
}

impl<const N: usize> Theory for BV<N> {
    type DType = BVDType<N>;

    fn _check(&self, read: &[Self::DType], write: &[Self::DType]) -> Result<(), String> {
        match self {
            BV::Const(cm) => {
                if read.len() > 0 {
                    return Err("Const: cannot read values".into());
                }
                if write.len() != 1 {
                    return Err("Const: must return a single value".into());
                }
                match &write[0] {
                    BVDType::BV(i, j) => {
                        if cm.len() != *i {
                            return Err(format!(
                                "Const: Initializer has a wrong number of rows (has {}, expected {})",
                                cm.len(),
                                *i
                            ));
                        }
                        if cm.iter().any(|row| row.len() != *j) {
                            return Err(format!(
                                "Const: some column of initializer has wrong dimension, expected {}",
                                *j
                            ));
                        }
                        Ok(())
                    }
                }
            }
            BV::Not => {
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
            BV::Xor | BV::And | BV::Or | BV::Add | BV::Mul => {
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
            BV::MatMul => {
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
                    (BVDType::BV(d1, d2), BVDType::BV(d3, d4), BVDType::BV(d5, d6)) => {
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
