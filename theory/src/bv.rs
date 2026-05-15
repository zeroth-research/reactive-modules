/*!
# Bit-vectors of fixed length

Defines the theory [`BV<N>`] of matrices of `N`-bit bit-vectors, where
the width `N` is fixed at compile time via a const generic.

A [`BVDType<N>`] value describes the *type* of a term as a matrix shape
`BV(rows, cols)` whose elements are `N`-bit bit-vectors. The operations
in [`BV`] are:

- [`BV::Const`] — an inline bit-vector matrix literal whose shape must
  match the declared write type.
- [`BV::Not`] — unary, shape-preserving.
- [`BV::And`], [`BV::Or`], [`BV::Xor`], [`BV::Add`], [`BV::Mul`] —
  elementwise binary; both inputs and the output share the same shape.
- [`BV::MatMul`] — matrix multiplication: `(m,k) × (k,n) → (m,n)`, with
  the inner dimensions required to agree.

`BV<N>` implements [`Theory`], so [`Theory::check`] type-checks terms by
validating read/write argument shapes against the operation.

## Examples

```
use theory::Theory;
use theory::bv::{BV, DType};

// 8-bit bit-vectors.
let a = DType::BVU(8, 2, 3);
let b = DType::BVU(8, 3, 4);
let c = DType::BVU(8, 2, 4);

// Matrix multiply: (2x3) * (3x4) -> (2x4).
assert!(BV::MatMul.check(&[a, b], &[c]).is_ok());

// Elementwise `Add` requires matching shapes.
let m = DType::BVU(8, 2, 3);
assert!(BV::Add.check(&[m, m], &[m]).is_ok());
assert!(BV::Add.check(&[a, b], &[c]).is_err());
```
*/

use crate::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DType {
    // matrix of unsigned bitvectors determnined by (bitvector-length, # rows, # cols)
    BVU(usize, usize, usize),
    // matrix of signed bitvectors determnined by (bitvector-length, # rows, # cols)
    BVS(usize, usize, usize),
}

impl DType {
    pub fn is_signed(&self) -> bool {
        matches!(self, DType::BVS(_, _, _))
    }

    pub fn shape(&self) -> (usize, usize) {
        match self {
            DType::BVU(_, i, j) => (*i, *j),
            DType::BVS(_, i, j) => (*i, *j),
        }
    }

    pub fn bw(&self) -> usize {
        match self {
            DType::BVU(bw, _, _) => *bw,
            DType::BVS(bw, _, _) => *bw,
        }
    }
}

/// TODO: write "formal" semantics
#[derive(Clone, PartialEq, Debug)]
pub enum BV {
    // TODO: use bitarray, this works only for `N <= 64`
    Const(Vec<Vec<usize>>),
    Add,
    Mul,
    MatMul,
    And,
    Or,
    Xor,
    Not,
    // TODO: add cast from to BVU/BVS
}

fn check_init_dims(cm: &[Vec<usize>], bw: usize, i: usize, j: usize) -> Result<(), String> {
    if bw > 64 {
        return Err("Support at most 64-bit atm.".into());
    }
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

impl Theory for BV {
    type DType = DType;

    fn check<'a, R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<&'a DType>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        let mut read = read.into_iter();
        let mut write = write.into_iter();

        match self {
            BV::Const(cm) => {
                if read.next().is_some() {
                    return Err("Const: cannot read values".into());
                }
                let dtype = write_nxt(&mut write, 0)?;
                match dtype {
                    DType::BVU(bw, i, j) => check_init_dims(cm, *bw, *i, *j)?,
                    DType::BVS(bw, i, j) => check_init_dims(cm, *bw, *i, *j)?,
                }
                if write.next().is_some() {
                    return Err("Const: returns more than one value".into());
                }
                Ok(())
            }

            BV::Not => {
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
            BV::And | BV::Or | BV::Xor | BV::Add | BV::Mul => {
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

            BV::MatMul => {
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

                if r1.bw() != r2.bw() {
                    return Err(format!(
                        "{:?}: 1st and 2nd inputs have different bitwidth: {} != {}",
                        self,
                        r1.bw(),
                        r2.bw()
                    ));
                }

                if r1.bw() != w1.bw() {
                    return Err(format!(
                        "{:?}: inputs and output have different bitwidth: {} != {}",
                        self,
                        r1.bw(),
                        w1.bw()
                    ));
                }

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
