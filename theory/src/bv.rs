/*!
# Bit-vectors of fixed length

Defines the theory [`BV<N>`] of matrices of `N`-bit bit-vectors, where
the width `N` is fixed at compile time via a const generic.

A [`BVType<N>`] value describes the *type* of a term as a matrix shape
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
* use theory::Theory;
* use theory::bv::{BV, Sort};
*
*  // 8-bit bit-vectors.
*  let a = Type::BV(8, [2, 3]);
*  let b = Type::BV(8, [3, 4]);
*  let c = Type::BV(8, [2, 4]);
*
*  // Matrix multiply: (2x3) * (3x4) -> (2x4).
*  assert!(BV::MatMul.check([a, b], [c]).is_ok());
*
*  // Elementwise `Add` requires matching shapes.
*  let m = Type::BV(8, [2, 3]);
*  assert!(BV::Add.check([m, m], [m]).is_ok());
*  assert!(BV::Add.check([a, b], [c]).is_err());
*  ```
*/

use crate::*;
use pyo3::prelude::*;

use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Sort {
    // matrix of bitvectors determined by (bitvector-length, [# rows, # cols])
    BV(usize, [usize; 2]),
}

impl Sort {
    pub fn shape(&self) -> &[usize; 2] {
        match self {
            Sort::BV(_, shape) => shape,
        }
    }

    pub fn bw(&self) -> usize {
        match self {
            Sort::BV(bw, _) => *bw,
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::BV(bw, [i, j]) => write!(f, "BV<{bw}>({i}, {j})"),
        }
    }
}

/// Theory of bitvector matrices. Operations on bitvectors follow the SMT-LIB2 semantics.
#[derive(Clone, Debug)]
#[pyclass(frozen)]
pub enum BV {
    /// Create constant BV matrix from a given 2-D tensor
    Const(crate::PyTensor),
    /// Element-wise addition of two BV matrices (arithmetic modulo `2^bitwidth`)
    Add(),
    /// Element-wise subtraction of two BV matrices (arithmetic modulo `2^bitwidth`)
    Sub(),
    /// Element-wise multiplication of two BV matrices (arithmetic modulo `2^bitwidth`)
    Mul(),
    /// Element-wise **unsigned** division of two BV matrices (arithmetic modulo `2^bitwidth`)
    UDiv(),
    /// Element-wise **signed** division of two BV matrices . Arithmetic modulo `2^bitwidth`.
    /// Edge cases follow SMT-LIB2 specification (e.g., `-128 / -1 = -128` on 8 bits and
    /// `x / 0 = 2^(bitwidth - 1)`)
    SDiv(),
    /// Element-wise **unsigned** modulo of two BV matrices
    UMod(),
    /// Element-wise **signed** modulo of two BV matrices
    SMod(),
    /// Element-wise negation of two BV matrices (arithmetic modulo `2^bitwidth`).
    /// Edge case follows the SMT-LIB2 specification (e.g., `neg(-128) = -128` on 8 bits)
    Neg(),
    /// XXX: remove from here and move the modelling to Python.
    /// XXX What to do on `Abs(BV<n>_MIN)`? Should the result be `BV<n>_MAX`?
    /// (so that `Abs` always returns positive value?)
    /// Element-wise absolute value. Modeled as `Abs(x) = Ite(x < 0, Neg(x), x)`
    Abs(),
    /// Two bitvector matrices multiplication. Modeled through multiplication and sums
    /// as usual.
    MatMul(),
    /// Element-wise bit-wise "and"
    And(),
    /// Element-wise bit-wise "or"
    Or(),
    /// Element-wise bit-wise "xor"
    Xor(),
    /// Element-wise bit-wise "not" (bit inversion)
    Not(),
    /// Element-wise **unsigned** less-or-equal comparison (result is a matrix of BV<1> values)
    ULe(),
    /// Element-wise **unsigned** less-than comparison (result is a matrix of BV<1> values)
    ULt(),
    /// Element-wise **unsigned** greater-or-equal comparison (result is a matrix of BV<1> values)
    UGe(),
    /// Element-wise **unsigned** greater-than comparison (result is a matrix of BV<1> values)
    UGt(),
    /// Element-wise **signed** less-or-equal comparison (result is a matrix of BV<1> values)
    SLe(),
    /// Element-wise **signed** less-than comparison (result is a matrix of BV<1> values)
    SLt(),
    /// Element-wise **signed** greater-or-equal comparison (result is a matrix of BV<1> values)
    SGe(),
    /// Element-wise **signed** greater-than comparison (result is a matrix of BV<1> values)
    SGt(),
    /// Element-wise equality comparison (result is a matrix of BV<1> values)
    Eq(),
    /// Element-wise non-equality comparison (result is a matrix of BV<1> values)
    Ne(),
    /// If-then-else construct, condition is `BV<1>`
    Ite(),
    /// Identity on bitvector matrices
    Id(),
    /// Element-wise BV<n> to BV<1> via "non-zero comparison" (`x != 0`). Corresponds to SMV ToBool,
    /// FIXME: to be removed and replaced by `Ne x 0`
    BVToBool(),
    /// Element-wise extract bits `[high..=low]` (inclusive)
    /// TODO: rename to `extract`, matching MathSAT
    BitSelect { high: usize, low: usize },
    /// Element-wise zero-extend by `extra` bits
    Extend { extra: usize },
    /// Uninterpreted symbol
    Uninterpreted(String),
}

impl fmt::Display for BV {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BV::Const(cm) => write!(f, "{}", cm),
            BV::And() => write!(f, "And"),
            BV::Or() => write!(f, "Or"),
            BV::Xor() => write!(f, "Xor"),
            BV::Not() => write!(f, "Not"),
            BV::SLe() => write!(f, "SLe"),
            BV::SLt() => write!(f, "SLt"),
            BV::SGe() => write!(f, "SGe"),
            BV::SGt() => write!(f, "SGt"),
            BV::ULe() => write!(f, "ULe"),
            BV::ULt() => write!(f, "ULt"),
            BV::UGe() => write!(f, "UGe"),
            BV::UGt() => write!(f, "UGt"),
            BV::Eq() => write!(f, "Eq"),
            BV::Ne() => write!(f, "Ne"),
            BV::Add() => write!(f, "Add"),
            BV::Sub() => write!(f, "Sub"),
            BV::Neg() => write!(f, "Neg"),
            BV::Abs() => write!(f, "Abs"),
            BV::Mul() => write!(f, "Mul"),
            BV::UDiv() => write!(f, "UDiv"),
            BV::SDiv() => write!(f, "SDiv"),
            BV::UMod() => write!(f, "UMod"),
            BV::SMod() => write!(f, "SMod"),
            BV::MatMul() => write!(f, "MatMul"),
            BV::Ite() => write!(f, "Ite"),
            BV::Id() => write!(f, "Id"),
            BV::BVToBool() => write!(f, "BVToBool"),
            BV::BitSelect { high, low } => write!(f, "BitSelect[{high}:{low}]"),
            BV::Extend { extra } => write!(f, "Extend(+{extra})"),
            BV::Uninterpreted(name) => write!(f, "Uninterpreted({name})"),
        }
    }
}

fn check_init_dims(cm: &crate::PyTensor, bw: usize, i: usize, j: usize) -> Result<(), String> {
    if bw > 64 {
        return Err("Support at most 64-bit atm.".into());
    }
    let size = cm.size();
    if size.len() != 2 {
        return Err(format!(
            "Const: initializer must be a 2D tensor, got {}D",
            size.len()
        ));
    }
    if size[0] as usize != i {
        return Err(format!(
            "Const: Initializer has a wrong number of rows (has {}, expected {})",
            size[0], i
        ));
    }
    if size[1] as usize != j {
        return Err(format!(
            "Const: some column of initializer has wrong dimension, expected {}",
            j
        ));
    }
    if bw < 64 {
        let max = (1i64 << bw) - 1;
        let min_val = cm.min().int64_value(&[]);
        let max_val = cm.max().int64_value(&[]);
        if min_val < 0 || max_val > max {
            return Err(format!("Const: tensor values do not fit in {bw} bits"));
        }
    }
    Ok(())
}

impl Theory for BV {
    type Sort = Sort;
    const NAME: &'static str = "BV";

    fn check<R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<Sort> + fmt::Display,
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
                let dtype = write_nxt(&mut write, 0, "BV")?;
                match dtype {
                    Sort::BV(bw, [i, j]) => check_init_dims(cm, bw, i, j)?,
                }
                if write.next().is_some() {
                    return Err("Const: returns more than one value".into());
                }
                Ok(())
            }

            // TODO: for `Abs`, what if input is the signed min of BV<N>? It's absolute value
            // does not fit into BV<N>
            BV::Not() | BV::Id() | BV::Neg() | BV::Abs() => {
                let (r, w) = (
                    read_nxt(&mut read, 0, "BV")?,
                    write_nxt(&mut write, 0, "BV")?,
                );
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
            BV::SLe()
            | BV::SLt()
            | BV::SGe()
            | BV::SGt()
            | BV::ULe()
            | BV::ULt()
            | BV::UGe()
            | BV::UGt()
            | BV::Eq()
            | BV::Ne() => {
                let (r1, r2, None) = (
                    read_nxt(&mut read, 0, "BV")?,
                    read_nxt(&mut read, 1, "BV")?,
                    read.next(),
                ) else {
                    return Err(format!("{self}: must read exactly two values"));
                };
                let (w1, None) = (write_nxt(&mut write, 0, "BV")?, write.next()) else {
                    return Err(format!("{self}: must write exactly one value"));
                };
                if r1 != r2 {
                    return Err(format!("{self}: inputs must have the same type"));
                }
                let [rows, cols] = r1.shape();
                if w1 != Sort::BV(1, [*rows, *cols]) {
                    return Err(format!(
                        "{self}: output must be BV<1>({rows}, {cols}), got {w1}"
                    ));
                }
                Ok(())
            }
            BV::UDiv() | BV::UMod() => {
                let (r1, r2, None) = (
                    read_nxt(&mut read, 0, "BV")?,
                    read_nxt(&mut read, 1, "BV")?,
                    read.next(),
                ) else {
                    return Err(format!("{self}: must read exactly two values"));
                };
                let (w1, None) = (write_nxt(&mut write, 0, "BV")?, write.next()) else {
                    return Err(format!("{self}: must write exactly one value"));
                };
                if !matches!(r1, Sort::BV(..)) {
                    return Err(format!("{self}: inputs must be unsigned, got {r1}"));
                }
                if r1 != r2 {
                    return Err(format!("{self}: input values must have the same type"));
                }
                if w1 != r1 {
                    return Err(format!(
                        "{self}: input and output values must have the same type"
                    ));
                }
                Ok(())
            }
            BV::SDiv() | BV::SMod() => {
                let (r1, r2, None) = (
                    read_nxt(&mut read, 0, "BV")?,
                    read_nxt(&mut read, 1, "BV")?,
                    read.next(),
                ) else {
                    return Err(format!("{self}: must read exactly two values"));
                };
                let (w1, None) = (write_nxt(&mut write, 0, "BV")?, write.next()) else {
                    return Err(format!("{self}: must write exactly one value"));
                };
                if !matches!(r1, Sort::BV(..)) {
                    return Err(format!("{self}: inputs must be signed, got {r1}"));
                }
                if r1 != r2 {
                    return Err(format!("{self}: input values must have the same type"));
                }
                if w1 != r1 {
                    return Err(format!(
                        "{self}: input and output values must have the same type"
                    ));
                }
                Ok(())
            }
            BV::And() | BV::Or() | BV::Xor() | BV::Add() | BV::Sub() | BV::Mul() => {
                let w1 = write_nxt(&mut write, 0, "BV")?;
                let (r1, r2, None) = (
                    read_nxt(&mut read, 0, "BV")?,
                    read_nxt(&mut read, 1, "BV")?,
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

            BV::Ite() => {
                let w1 = write_nxt(&mut write, 0, "BV")?;
                let (r1, r2, r3, None) = (
                    read_nxt(&mut read, 0, "BV")?,
                    read_nxt(&mut read, 1, "BV")?,
                    read_nxt(&mut read, 2, "BV")?,
                    read.next(),
                ) else {
                    return Err(format!("{:?}: must read exactly two values", self));
                };
                if r1.bw() != 1 {
                    return Err(format!(
                        "{:?}: first argument must have one bit exactly",
                        self
                    ));
                }
                if r2 != r3 {
                    return Err(format!("{:?}: arms must have the same type", self));
                }
                if w1 != r2 {
                    return Err(format!(
                        "{:?}: input and output values must have the same type",
                        self
                    ));
                }
                Ok(())
            }

            BV::MatMul() => {
                let (r1, r2, None) = (
                    read_nxt(&mut read, 0, "BV")?,
                    read_nxt(&mut read, 1, "BV")?,
                    read.next(),
                ) else {
                    return Err(format!("{:?}: must read exactly two values", self));
                };
                let (w1, None) = (write_nxt(&mut write, 0, "BV")?, write.next()) else {
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

                let [d1, d2] = r1.shape();
                let [d3, d4] = r2.shape();
                let [d5, d6] = w1.shape();

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
            BV::BVToBool() => {
                let (r1, None) = (read_nxt(&mut read, 0, "BV")?, read.next()) else {
                    return Err(format!("{self}: must read exactly one value"));
                };
                let (w1, None) = (write_nxt(&mut write, 0, "BV")?, write.next()) else {
                    return Err(format!("{self}: must write exactly one value"));
                };
                let [rows, cols] = r1.shape();
                if w1 != Sort::BV(1, [*rows, *cols]) {
                    return Err(format!(
                        "{self}: output must be BV<1>({rows}, {cols}), got {w1}"
                    ));
                }
                Ok(())
            }
            BV::BitSelect { high, low } => {
                let (r1, None) = (read_nxt(&mut read, 0, "BV")?, read.next()) else {
                    return Err(format!("{self}: must read exactly one value"));
                };
                let (w1, None) = (write_nxt(&mut write, 0, "BV")?, write.next()) else {
                    return Err(format!("{self}: must write exactly one value"));
                };
                if high < low {
                    return Err(format!("{self}: high ({high}) must be >= low ({low})"));
                }
                if *high >= r1.bw() {
                    return Err(format!(
                        "{self}: high ({high}) is out of range for input width {}",
                        r1.bw()
                    ));
                }
                let out_bw = high - low + 1;
                let [rows, cols] = r1.shape();
                if w1 != Sort::BV(out_bw, [*rows, *cols]) {
                    return Err(format!(
                        "{self}: output must be BV<{out_bw}>({rows}, {cols}), got {w1}"
                    ));
                }
                Ok(())
            }
            BV::Extend { extra } => {
                let (r1, None) = (read_nxt(&mut read, 0, "BV")?, read.next()) else {
                    return Err(format!("{self}: must read exactly one value"));
                };
                let (w1, None) = (write_nxt(&mut write, 0, "BV")?, write.next()) else {
                    return Err(format!("{self}: must write exactly one value"));
                };
                let out_bw = r1.bw() + extra;
                let [rows, cols] = r1.shape();
                if w1 != Sort::BV(out_bw, [*rows, *cols]) {
                    return Err(format!(
                        "{self}: output must be BV<{out_bw}>({rows}, {cols}), got {w1}"
                    ));
                }
                Ok(())
            }
            BV::Uninterpreted(_) => {
                // uninterpreted has either one read or one write
                if read.next().is_some() {
                    if read.next().is_some() {
                        return Err(format!("{:?}: expected exactly one read, got more", self));
                    }
                    if write.next().is_some() {
                        return Err(format!(
                            "{:?}: expected exactly one read, got also write",
                            self
                        ));
                    }
                    return Ok(());
                }
                if write.next().is_some() {
                    if write.next().is_some() {
                        return Err(format!("{:?}: expected exactly one write, got more", self));
                    }
                    if read.next().is_some() {
                        return Err(format!(
                            "{:?}: expected exactly one write, got also read",
                            self
                        ));
                    }
                    return Ok(());
                }
                return Err(format!(
                    "{:?}: expected exactly one write or one read, got none",
                    self
                ));
            }
        }
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    fn bv(bw: usize, r: usize, c: usize) -> Sort {
        Sort::BV(bw, [r, c])
    }

    // --- Type helpers ---

    #[test]
    fn type_shape_and_bw() {
        let t = bv(8, 2, 3);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.bw(), 8);

        let t = bv(16, 1, 4);
        assert_eq!(t.shape(), &[1, 4]);
        assert_eq!(t.bw(), 16);
    }

    // --- Const ---

    #[test]
    fn const_ok() {
        let t = bv(8, 2, 2);
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0i64, 1], [2, 3]]).into();
        assert!(BV::Const(cm).check(vec![], [t]).is_ok());
    }

    #[test]
    fn const_value_overflow_fails() {
        let t = bv(8, 1, 1);
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[256i64]]).into();
        assert!(BV::Const(cm).check(vec![], [t]).is_err());
    }

    #[test]
    fn const_value_max_fits() {
        let t = bv(8, 1, 1);
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[255i64]]).into();
        assert!(BV::Const(cm).check(vec![], [t]).is_ok());
    }

    #[test]
    fn const_wrong_row_count_fails() {
        let t = bv(8, 2, 2);
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0i64, 1]]).into();
        assert!(BV::Const(cm).check(vec![], [t]).is_err());
    }

    #[test]
    fn const_wrong_col_count_fails() {
        let t = bv(8, 1, 2);
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0i64]]).into();
        assert!(BV::Const(cm).check(vec![], [t]).is_err());
    }

    #[test]
    fn const_with_read_fails() {
        let t = bv(8, 1, 1);
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0i64]]).into();
        assert!(BV::Const(cm).check([t], [t]).is_err());
    }

    // --- Not / Id ---

    #[test]
    fn not_ok() {
        let t = bv(8, 2, 3);
        assert!(BV::Not().check([t], [t]).is_ok());
    }

    #[test]
    fn not_type_mismatch_fails() {
        assert!(BV::Not().check([bv(8, 1, 1)], [bv(8, 2, 2)]).is_err());
    }

    #[test]
    fn id_ok() {
        let t = bv(32, 4, 4);
        assert!(BV::Id().check([t], [t]).is_ok());
    }

    // --- Binary elementwise (Add, Mul, And, Or, Xor) ---

    #[test]
    fn add_ok() {
        let t = bv(8, 3, 3);
        assert!(BV::Add().check([t, t], [t]).is_ok());
    }

    #[test]
    fn add_shape_mismatch_fails() {
        assert!(
            BV::Add()
                .check([bv(8, 1, 2), bv(8, 2, 1)], [bv(8, 1, 2)])
                .is_err()
        );
    }

    #[test]
    fn mul_ok() {
        let t = bv(16, 1, 1);
        assert!(BV::Mul().check([t, t], [t]).is_ok());
    }

    #[test]
    fn and_ok() {
        let t = bv(1, 2, 2);
        assert!(BV::And().check([t, t], [t]).is_ok());
    }

    #[test]
    fn or_ok() {
        let t = bv(1, 2, 2);
        assert!(BV::Or().check([t, t], [t]).is_ok());
    }

    #[test]
    fn xor_ok() {
        let t = bv(4, 1, 1);
        assert!(BV::Xor().check([t, t], [t]).is_ok());
    }

    // --- Comparisons ---

    #[test]
    fn lt_ok() {
        let t = bv(8, 2, 3);
        let out = bv(1, 2, 3);
        assert!(BV::ULt().check([t, t], [out]).is_ok());
        assert!(BV::SLt().check([t, t], [out]).is_ok());
    }

    #[test]
    fn le_ok() {
        let t = bv(8, 1, 1);
        let out = bv(1, 1, 1);
        assert!(BV::ULe().check([t, t], [out]).is_ok());
        assert!(BV::SLe().check([t, t], [out]).is_ok());
    }

    #[test]
    fn eq_ok() {
        let t = bv(32, 1, 1);
        let out = bv(1, 1, 1);
        assert!(BV::Eq().check([t, t], [out]).is_ok());
    }

    #[test]
    fn ne_ok() {
        let t = bv(8, 1, 1);
        let out = bv(1, 1, 1);
        assert!(BV::Ne().check([t, t], [out]).is_ok());
    }

    #[test]
    fn cmp_wrong_output_type_fails() {
        let t = bv(8, 2, 3);
        // output must be U(1, rows, cols), not U(8, ...)
        assert!(BV::ULt().check([t, t], [t]).is_err());
        assert!(BV::SLt().check([t, t], [t]).is_err());
    }

    #[test]
    fn cmp_input_mismatch_fails() {
        let out = bv(1, 1, 1);
        assert!(BV::Eq().check([bv(8, 1, 1), bv(16, 1, 1)], [out]).is_err());
    }

    // --- UDiv / SDiv ---

    #[test]
    fn udiv_ok() {
        let t = bv(8, 1, 1);
        assert!(BV::UDiv().check([t, t], [t]).is_ok());
    }

    #[test]
    fn sdiv_ok() {
        let t = bv(8, 1, 1);
        assert!(BV::SDiv().check([t, t], [t]).is_ok());
    }

    // --- MatMul ---

    #[test]
    fn matmul_ok() {
        let a = bv(8, 2, 3);
        let b = bv(8, 3, 4);
        let c = bv(8, 2, 4);
        assert!(BV::MatMul().check([a, b], [c]).is_ok());
    }

    #[test]
    fn matmul_inner_dim_mismatch_fails() {
        let a = bv(8, 2, 3);
        let b = bv(8, 4, 4);
        let c = bv(8, 2, 4);
        assert!(BV::MatMul().check([a, b], [c]).is_err());
    }

    #[test]
    fn matmul_bw_mismatch_fails() {
        let a = bv(8, 2, 3);
        let b = bv(16, 3, 4);
        let c = bv(8, 2, 4);
        assert!(BV::MatMul().check([a, b], [c]).is_err());
    }

    // --- Ite ---

    #[test]
    fn ite_ok() {
        let cond = bv(1, 1, 1);
        let t = bv(8, 1, 1);
        assert!(BV::Ite().check([cond, t, t], [t]).is_ok());
    }

    #[test]
    fn ite_bad_condition_bw_fails() {
        let cond = bv(8, 1, 1);
        let t = bv(8, 1, 1);
        assert!(BV::Ite().check([cond, t, t], [t]).is_err());
    }

    // --- Sub / Neg / UMod / SMod ---

    #[test]
    fn sub_ok() {
        let t = bv(8, 2, 3);
        assert!(BV::Sub().check([t, t], [t]).is_ok());
    }

    #[test]
    fn neg_ok() {
        let t = bv(8, 2, 3);
        assert!(BV::Neg().check([t], [t]).is_ok());
    }

    #[test]
    fn neg_type_mismatch_fails() {
        assert!(BV::Neg().check([bv(8, 1, 1)], [bv(8, 2, 2)]).is_err());
    }

    #[test]
    fn umod_ok() {
        let t = bv(8, 1, 1);
        assert!(BV::UMod().check([t, t], [t]).is_ok());
    }

    #[test]
    fn smod_ok() {
        let t = bv(8, 1, 1);
        assert!(BV::SMod().check([t, t], [t]).is_ok());
    }

    // --- BVToBool / BoolToBV (SMV bool(...) / word1(...)) ---

    #[test]
    fn bv_to_bool_ok() {
        // BV<8>(2,3) -> BV<1>(2,3)
        assert!(BV::BVToBool().check([bv(8, 2, 3)], [bv(1, 2, 3)]).is_ok());
    }

    #[test]
    fn bv_to_bool_identity_width_ok() {
        // BV<1> -> BV<1> is also fine (idempotent on already-bool input).
        assert!(BV::BVToBool().check([bv(1, 1, 1)], [bv(1, 1, 1)]).is_ok());
    }

    #[test]
    fn bv_to_bool_wrong_out_width_fails() {
        assert!(BV::BVToBool().check([bv(8, 1, 1)], [bv(8, 1, 1)]).is_err());
    }

    #[test]
    fn bv_to_bool_shape_mismatch_fails() {
        assert!(BV::BVToBool().check([bv(8, 2, 3)], [bv(1, 1, 1)]).is_err());
    }

    // --- BitSelect / Extend ---

    #[test]
    fn bit_select_ok() {
        // BV<16>[11:4] -> BV<8>
        assert!(
            BV::BitSelect { high: 11, low: 4 }
                .check([bv(16, 1, 1)], [bv(8, 1, 1)])
                .is_ok()
        );
    }

    #[test]
    fn bit_select_single_bit_ok() {
        assert!(
            BV::BitSelect { high: 3, low: 3 }
                .check([bv(8, 2, 2)], [bv(1, 2, 2)])
                .is_ok()
        );
    }

    #[test]
    fn bit_select_out_of_range_fails() {
        assert!(
            BV::BitSelect { high: 8, low: 0 }
                .check([bv(8, 1, 1)], [bv(9, 1, 1)])
                .is_err()
        );
    }

    #[test]
    fn bit_select_wrong_out_width_fails() {
        assert!(
            BV::BitSelect { high: 7, low: 0 }
                .check([bv(16, 1, 1)], [bv(7, 1, 1)])
                .is_err()
        );
    }

    #[test]
    fn extend_ok() {
        assert!(
            BV::Extend { extra: 8 }
                .check([bv(8, 1, 1)], [bv(16, 1, 1)])
                .is_ok()
        );
    }

    #[test]
    fn extend_wrong_out_width_fails() {
        assert!(
            BV::Extend { extra: 4 }
                .check([bv(8, 1, 1)], [bv(16, 1, 1)])
                .is_err()
        );
    }

    #[test]
    fn ite_arm_mismatch_fails() {
        let cond = bv(1, 1, 1);
        assert!(
            BV::Ite()
                .check([cond, bv(8, 1, 1), bv(16, 1, 1)], [bv(8, 1, 1)])
                .is_err()
        );
    }
}
