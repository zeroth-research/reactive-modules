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
use theory::Theory;
use theory::bv::{BV, Type};

 // 8-bit bit-vectors.
 let a = Type::UWord(8, [2, 3]);
 let b = Type::UWord(8, [3, 4]);
 let c = Type::UWord(8, [2, 4]);

 // Matrix multiply: (2x3) * (3x4) -> (2x4).
 assert!(BV::MatMul.check([a, b], [c]).is_ok());

 // Elementwise `Add` requires matching shapes.
 let m = Type::UWord(8, [2, 3]);
 assert!(BV::Add.check([m, m], [m]).is_ok());
 assert!(BV::Add.check([a, b], [c]).is_err());
 ```
*/

use crate::*;

use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Type {
    // matrix of unsigned bitvectors determined by (bitvector-length, # rows, # cols)
    UWord(usize, [usize; 2]),
    // matrix of signed bitvectors determined by (bitvector-length, # rows, # cols)
    SWord(usize, [usize; 2]),
}

impl Type {
    pub fn is_signed(&self) -> bool {
        matches!(self, Type::SWord(_, _))
    }

    pub fn shape(&self) -> &[usize; 2] {
        match self {
            Type::UWord(_, shape) => shape,
            Type::SWord(_, shape) => shape,
        }
    }

    pub fn bw(&self) -> usize {
        match self {
            Type::UWord(bw, _) => *bw,
            Type::SWord(bw, _) => *bw,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::UWord(bw, [i, j]) => write!(f, "uBV<{bw}>({i}, {j})"),
            Type::SWord(bw, [i, j]) => write!(f, "sBV<{bw}>({i}, {j})"),
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
    UDiv,
    SDiv,
    MatMul,
    And,
    Or,
    Xor,
    Not,
    Le,
    Lt,
    Ge,
    Gt,
    Eq,
    Ne,
    Ite,
    Id,
}

impl fmt::Display for BV {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BV::Const(cm) => fmt_matrix(cm, f),
            BV::And => write!(f, "And"),
            BV::Or => write!(f, "Or"),
            BV::Xor => write!(f, "Xor"),
            BV::Not => write!(f, "Not"),
            BV::Le => write!(f, "Le"),
            BV::Lt => write!(f, "Lt"),
            BV::Ge => write!(f, "Ge"),
            BV::Gt => write!(f, "Gt"),
            BV::Eq => write!(f, "Eq"),
            BV::Ne => write!(f, "Ne"),
            BV::Add => write!(f, "Add"),
            BV::Mul => write!(f, "Mul"),
            BV::UDiv => write!(f, "UDiv"),
            BV::SDiv => write!(f, "SDiv"),
            BV::MatMul => write!(f, "MatMul"),
            BV::Ite => write!(f, "Ite"),
            BV::Id => write!(f, "Id"),
        }
    }
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
    if bw < 64 {
        let max = 1usize << bw;
        for (r, row) in cm.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                if v >= max {
                    return Err(format!(
                        "Const: value {v} at [{r},{c}] does not fit in {bw} bits"
                    ));
                }
            }
        }
    }
    Ok(())
}

impl Theory for BV {
    type DType = Type;

    fn check<R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<Type>,
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
                    Type::UWord(bw, [i, j]) => check_init_dims(cm, bw, i, j)?,
                    Type::SWord(bw, [i, j]) => check_init_dims(cm, bw, i, j)?,
                }
                if write.next().is_some() {
                    return Err("Const: returns more than one value".into());
                }
                Ok(())
            }

            BV::Not | BV::Id => {
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
            BV::Le | BV::Lt | BV::Ge | BV::Gt | BV::Eq | BV::Ne => {
                let (r1, r2, None) = (
                    read_nxt(&mut read, 0)?,
                    read_nxt(&mut read, 1)?,
                    read.next(),
                ) else {
                    return Err(format!("{self}: must read exactly two values"));
                };
                let (w1, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                    return Err(format!("{self}: must write exactly one value"));
                };
                if r1 != r2 {
                    return Err(format!("{self}: inputs must have the same type"));
                }
                let [rows, cols] = r1.shape();
                if w1 != Type::UWord(1, [*rows, *cols]) {
                    return Err(format!(
                        "{self}: output must be U(1, {rows}, {cols}), got {w1}"
                    ));
                }
                Ok(())
            }
            BV::UDiv => {
                let (r1, r2, None) = (
                    read_nxt(&mut read, 0)?,
                    read_nxt(&mut read, 1)?,
                    read.next(),
                ) else {
                    return Err(format!("{self}: must read exactly two values"));
                };
                let (w1, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                    return Err(format!("{self}: must write exactly one value"));
                };
                if !matches!(r1, Type::UWord(..)) {
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
            BV::SDiv => {
                let (r1, r2, None) = (
                    read_nxt(&mut read, 0)?,
                    read_nxt(&mut read, 1)?,
                    read.next(),
                ) else {
                    return Err(format!("{self}: must read exactly two values"));
                };
                let (w1, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                    return Err(format!("{self}: must write exactly one value"));
                };
                if !matches!(r1, Type::SWord(..)) {
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

            BV::Ite => {
                let w1 = write_nxt(&mut write, 0)?;
                let (r1, r2, r3, None) = (
                    read_nxt(&mut read, 0)?,
                    read_nxt(&mut read, 1)?,
                    read_nxt(&mut read, 2)?,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(bw: usize, r: usize, c: usize) -> Type {
        Type::UWord(bw, [r, c])
    }

    fn s(bw: usize, r: usize, c: usize) -> Type {
        Type::SWord(bw, [r, c])
    }

    // --- Type helpers ---

    #[test]
    fn type_shape_and_bw() {
        let t = u(8, 2, 3);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.bw(), 8);
        assert!(!t.is_signed());

        let t = s(16, 1, 4);
        assert_eq!(t.shape(), &[1, 4]);
        assert_eq!(t.bw(), 16);
        assert!(t.is_signed());
    }

    // --- Const ---

    #[test]
    fn const_ok() {
        let t = u(8, 2, 2);
        let cm = vec![vec![0usize, 1], vec![2, 3]];
        assert!(BV::Const(cm).check(vec![], [t.clone()]).is_ok());
    }

    #[test]
    fn const_value_overflow_fails() {
        let t = u(8, 1, 1);
        let cm = vec![vec![256usize]];
        assert!(BV::Const(cm).check(vec![], [t.clone()]).is_err());
    }

    #[test]
    fn const_value_max_fits() {
        let t = u(8, 1, 1);
        let cm = vec![vec![255usize]];
        assert!(BV::Const(cm).check(vec![], [t.clone()]).is_ok());
    }

    #[test]
    fn const_wrong_row_count_fails() {
        let t = u(8, 2, 2);
        let cm = vec![vec![0usize, 1]];
        assert!(BV::Const(cm).check(vec![], [t.clone()]).is_err());
    }

    #[test]
    fn const_wrong_col_count_fails() {
        let t = u(8, 1, 2);
        let cm = vec![vec![0usize]];
        assert!(BV::Const(cm).check(vec![], [t.clone()]).is_err());
    }

    #[test]
    fn const_with_read_fails() {
        let t = u(8, 1, 1);
        let cm = vec![vec![0usize]];
        assert!(BV::Const(cm).check([t.clone()], [t.clone()]).is_err());
    }

    // --- Not / Id ---

    #[test]
    fn not_ok() {
        let t = u(8, 2, 3);
        assert!(BV::Not.check([t.clone()], [t.clone()]).is_ok());
    }

    #[test]
    fn not_type_mismatch_fails() {
        assert!(BV::Not.check([u(8, 1, 1)], [u(8, 2, 2)]).is_err());
    }

    #[test]
    fn id_ok() {
        let t = s(32, 4, 4);
        assert!(BV::Id.check([t.clone()], [t.clone()]).is_ok());
    }

    // --- Binary elementwise (Add, Mul, And, Or, Xor) ---

    #[test]
    fn add_ok() {
        let t = u(8, 3, 3);
        assert!(BV::Add.check([t.clone(), t.clone()], [t.clone()]).is_ok());
    }

    #[test]
    fn add_shape_mismatch_fails() {
        assert!(BV::Add.check([u(8, 1, 2), u(8, 2, 1)], [u(8, 1, 2)]).is_err());
    }

    #[test]
    fn mul_ok() {
        let t = u(16, 1, 1);
        assert!(BV::Mul.check([t.clone(), t.clone()], [t.clone()]).is_ok());
    }

    #[test]
    fn and_ok() {
        let t = u(1, 2, 2);
        assert!(BV::And.check([t.clone(), t.clone()], [t.clone()]).is_ok());
    }

    #[test]
    fn or_ok() {
        let t = u(1, 2, 2);
        assert!(BV::Or.check([t.clone(), t.clone()], [t.clone()]).is_ok());
    }

    #[test]
    fn xor_ok() {
        let t = u(4, 1, 1);
        assert!(BV::Xor.check([t.clone(), t.clone()], [t.clone()]).is_ok());
    }

    // --- Comparisons ---

    #[test]
    fn lt_ok() {
        let t = u(8, 2, 3);
        let out = u(1, 2, 3);
        assert!(BV::Lt.check([t.clone(), t.clone()], [out.clone()]).is_ok());
    }

    #[test]
    fn le_ok() {
        let t = u(8, 1, 1);
        let out = u(1, 1, 1);
        assert!(BV::Le.check([t.clone(), t.clone()], [out.clone()]).is_ok());
    }

    #[test]
    fn eq_ok() {
        let t = s(32, 1, 1);
        let out = u(1, 1, 1);
        assert!(BV::Eq.check([t.clone(), t.clone()], [out.clone()]).is_ok());
    }

    #[test]
    fn ne_ok() {
        let t = u(8, 1, 1);
        let out = u(1, 1, 1);
        assert!(BV::Ne.check([t.clone(), t.clone()], [out.clone()]).is_ok());
    }

    #[test]
    fn cmp_wrong_output_type_fails() {
        let t = u(8, 2, 3);
        // output must be U(1, rows, cols), not U(8, ...)
        assert!(BV::Lt.check([t.clone(), t.clone()], [t.clone()]).is_err());
    }

    #[test]
    fn cmp_input_mismatch_fails() {
        let out = u(1, 1, 1);
        assert!(BV::Eq.check([u(8, 1, 1), u(16, 1, 1)], [out.clone()]).is_err());
    }

    // --- UDiv / SDiv ---

    #[test]
    fn udiv_ok() {
        let t = u(8, 1, 1);
        assert!(BV::UDiv.check([t.clone(), t.clone()], [t.clone()]).is_ok());
    }

    #[test]
    fn udiv_signed_input_fails() {
        let t = s(8, 1, 1);
        assert!(BV::UDiv.check([t.clone(), t.clone()], [t.clone()]).is_err());
    }

    #[test]
    fn sdiv_ok() {
        let t = s(8, 1, 1);
        assert!(BV::SDiv.check([t.clone(), t.clone()], [t.clone()]).is_ok());
    }

    #[test]
    fn sdiv_unsigned_input_fails() {
        let t = u(8, 1, 1);
        assert!(BV::SDiv.check([t.clone(), t.clone()], [t.clone()]).is_err());
    }

    // --- MatMul ---

    #[test]
    fn matmul_ok() {
        let a = u(8, 2, 3);
        let b = u(8, 3, 4);
        let c = u(8, 2, 4);
        assert!(BV::MatMul.check([a.clone(), b.clone()], [c.clone()]).is_ok());
    }

    #[test]
    fn matmul_inner_dim_mismatch_fails() {
        let a = u(8, 2, 3);
        let b = u(8, 4, 4);
        let c = u(8, 2, 4);
        assert!(BV::MatMul.check([a.clone(), b.clone()], [c.clone()]).is_err());
    }

    #[test]
    fn matmul_bw_mismatch_fails() {
        let a = u(8, 2, 3);
        let b = u(16, 3, 4);
        let c = u(8, 2, 4);
        assert!(BV::MatMul.check([a.clone(), b.clone()], [c.clone()]).is_err());
    }

    // --- Ite ---

    #[test]
    fn ite_ok() {
        let cond = u(1, 1, 1);
        let t = u(8, 1, 1);
        assert!(BV::Ite.check([cond.clone(), t.clone(), t.clone()], [t.clone()]).is_ok());
    }

    #[test]
    fn ite_bad_condition_bw_fails() {
        let cond = u(8, 1, 1);
        let t = u(8, 1, 1);
        assert!(BV::Ite.check([cond.clone(), t.clone(), t.clone()], [t.clone()]).is_err());
    }

    #[test]
    fn ite_arm_mismatch_fails() {
        let cond = u(1, 1, 1);
        assert!(BV::Ite.check([cond.clone(), u(8, 1, 1), u(16, 1, 1)], [u(8, 1, 1)]).is_err());
    }
}
