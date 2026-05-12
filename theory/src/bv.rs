/*!
# Bit-vectors of fixed length

Defines the theory [`BVTheory`] of matrices over bit-vectors.
The operations in [`BVTheory`] are:

- [`BV::Const`] — an inline bit-vector matrix literal whose shape must
  match the declared write type.
- [`BV::Not`] — unary, shape-preserving.
- [`BV::And`], [`BV::Or`], [`BV::Xor`], [`BV::Add`], [`BV::Mul`] —
  elementwise binary; both inputs and the output share the same shape.
- [`BV::MatMul`] — matrix multiplication: `(m,k) × (k,n) → (m,n)`, with
  the inner dimensions required to agree.

## Examples

```
use theory::Theory;
use theory::bv::{BV, BVTheory};

// 8-bit unsigned bit-vector matrices.
let a = BV::U(8, 2, 3);
let b = BV::U(8, 3, 4);
let c = BV::U(8, 2, 4);

// Matrix multiply: (2x3) * (3x4) -> (2x4).
assert!(BV::MatMul.type_check::<BV>(&[a, b], &[c]).is_ok());

// Elementwise `Add` requires matching shapes.
let m = BV::U(8, 2, 3);
assert!(BV::Add.check::<BV>(&[m, m], &[m]).is_ok());
assert!(BV::Add.check::<BV>(&[a, b], &[c]).is_err());
```
*/

use crate::*;
use std::fmt;

// TODO: factor out Mat<T>(usize, usize)? But then we need PhantomData...

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum BV {
    // matrix of unsigned bitvectors determnined by (bitvector-length, # rows, # cols)
    U(usize, usize, usize),
    // matrix of signed bitvectors determnined by (bitvector-length, # rows, # cols)
    S(usize, usize, usize),
}

impl BV {
    pub fn is_signed(&self) -> bool {
        matches!(self, BV::S(_, _, _))
    }

    pub fn bw(&self) -> usize {
        match self {
            BV::U(bw, _, _) => *bw,
            BV::S(bw, _, _) => *bw,
        }
    }
}

impl fmt::Display for BV {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BV::U(bw, r, c) => write!(f, "UWord({bw}, {r}, {c})"),
            BV::S(bw, r, c) => write!(f, "SWord({bw}, {r}, {c})"),
        }
    }
}

impl fmt::Display for BVTheory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BVTheory::Const(_) => write!(f, "Const"),
            BVTheory::Add => write!(f, "Add"),
            BVTheory::Sub => write!(f, "Sub"),
            BVTheory::Mul => write!(f, "Mul"),
            BVTheory::Div => write!(f, "Div"),
            BVTheory::Mod => write!(f, "Mod"),
            BVTheory::MatMul => write!(f, "MatMul"),
            BVTheory::Transpose => write!(f, "Transpose"),
            BVTheory::And => write!(f, "And"),
            BVTheory::Or => write!(f, "Or"),
            BVTheory::Xor => write!(f, "Xor"),
            BVTheory::Not => write!(f, "Not"),
            BVTheory::Select(h, l) => write!(f, "Select({h}, {l})"),
            BVTheory::Extend(w) => write!(f, "Extend({w})"),
            BVTheory::ToUnsigned() => write!(f, "ToUnsigned"),
            BVTheory::ToSigned() => write!(f, "ToSigned"),
        }
    }
}

impl MatrixType for BV {
    fn shape(&self) -> (usize, usize) {
        match self {
            BV::U(_, i, j) => (*i, *j),
            BV::S(_, i, j) => (*i, *j),
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum BVTheory {
    // TODO: use bitarray, this works only for `N <= 64`
    Const(Vec<Vec<usize>>),
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    MatMul,
    Transpose,
    And,
    Or,
    Xor,
    Not,
    // TODO: add cast from to U/S
    Select(u32, u32),
    Extend(u32),
    ToUnsigned(),
    ToSigned(),
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

impl Theory for BVTheory {
    type DType = BV;

    fn type_check<'a, R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<&'a BV>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        let mut read = read.into_iter();
        let mut write = write.into_iter();

        match self {
            BVTheory::Const(cm) => {
                if read.next().is_some() {
                    return Err("Const: cannot read values".into());
                }
                let dtype = write_nxt(&mut write, 0)?;
                match dtype {
                    BV::U(bw, i, j) => check_init_dims(cm, *bw, *i, *j)?,
                    BV::S(bw, i, j) => check_init_dims(cm, *bw, *i, *j)?,
                }
                if write.next().is_some() {
                    return Err("Const: returns more than one value".into());
                }
                Ok(())
            }

            BVTheory::Not => {
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
            BVTheory::And
            | BVTheory::Or
            | BVTheory::Xor
            | BVTheory::Add
            | BVTheory::Sub
            | BVTheory::Mul
            | BVTheory::Div
            | BVTheory::Mod => {
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

            BVTheory::MatMul => {
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
            BVTheory::Transpose => {
                let (r, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                    return Err("Transpose: must read exactly one value".into());
                };
                let (w, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                    return Err("Transpose: must write exactly one value".into());
                };
                let (rm, rn) = r.shape();
                let (wm, wn) = w.shape();
                if wm != rn || wn != rm {
                    return Err(format!(
                        "Transpose: output shape ({wm}, {wn}) must be ({rn}, {rm})"
                    ));
                }
                Ok(())
            }
            BVTheory::ToUnsigned() => {
                let (r, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                    return Err("ToUnsigned: must read exactly one value".into());
                };
                let (w, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                    return Err("ToUnsigned: must write exactly one value".into());
                };
                let expected = BV::U(r.bw(), r.shape().0, r.shape().1);
                if *w != expected {
                    return Err(format!(
                        "ToUnsigned: output must be {:?}, got {:?}",
                        expected, w
                    ));
                }
                Ok(())
            }
            BVTheory::ToSigned() => {
                let (r, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                    return Err("ToSigned: must read exactly one value".into());
                };
                let (w, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                    return Err("ToSigned: must write exactly one value".into());
                };
                let expected = BV::S(r.bw(), r.shape().0, r.shape().1);
                if *w != expected {
                    return Err(format!(
                        "ToSigned: output must be {:?}, got {:?}",
                        expected, w
                    ));
                }
                Ok(())
            }
            BVTheory::Select(high, low) => {
                let (r, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                    return Err("Select: must read exactly one value".into());
                };
                let (w, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                    return Err("Select: must write exactly one value".into());
                };
                let (high, low) = (*high as usize, *low as usize);
                if high < low {
                    return Err(format!("Select: high ({}) < low ({})", high, low));
                }
                if high >= r.bw() {
                    return Err(format!(
                        "Select: high ({}) >= input bitwidth ({})",
                        high,
                        r.bw()
                    ));
                }
                let (rows, cols) = r.shape();
                let expected = BV::U(high - low + 1, rows, cols);
                if *w != expected {
                    return Err(format!(
                        "Select: output must be {:?}, got {:?}",
                        expected, w
                    ));
                }
                Ok(())
            }
            BVTheory::Extend(width) => {
                let (r, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                    return Err("Extend: must read exactly one value".into());
                };
                let (w, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                    return Err("Extend: must write exactly one value".into());
                };
                let width = *width as usize;
                let (rows, cols) = r.shape();
                let out_bw = r.bw() + width;
                let expected = match r {
                    BV::U(_, _, _) => BV::U(out_bw, rows, cols),
                    BV::S(_, _, _) => BV::S(out_bw, rows, cols),
                };
                if *w != expected {
                    return Err(format!(
                        "Extend: output must be {:?}, got {:?}",
                        expected, w
                    ));
                }
                Ok(())
            }
        }
    }
}
