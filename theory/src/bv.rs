/*!
# Bit-vectors

Defines the theory [`BV`] of matrices of bit-vectors.

A [`Sort`] value is either `BV<N>(rows, cols)`, a matrix of `N`-bit bit-vectors,
or `Zero(rows, cols)`, the sort of their (zero) derivatives. The operations
in [`BV`] are:

- [`BV::Const`] — a bit-vector matrix literal whose values must fit into
  the width of the written matrix.
- [`BV::Not`], [`BV::Neg`], [`BV::Abs`] — unary, pointwise.
- [`BV::And`], [`BV::Or`], [`BV::Xor`], [`BV::Add`], [`BV::Sub`], [`BV::Mul`],
  [`BV::UDiv`], [`BV::SDiv`], [`BV::UMod`], [`BV::SMod`] — binary, pointwise;
  both inputs and the output have the same sort.
- [`BV::ULe`], [`BV::ULt`], [`BV::UGe`], [`BV::UGt`], [`BV::SLe`], [`BV::SLt`],
  [`BV::SGe`], [`BV::SGt`], [`BV::Eq`], [`BV::Ne`] — pointwise comparisons
  producing a matrix of bits `BV<1>`.
- [`BV::MatMul`] — matrix multiplication: `(m,k) × (k,n) → (m,n)`.
- [`BV::BitSelect`], [`BV::Extend`] — pointwise extraction of bits and zero-extension.
- [`BV::Ite`] — if-then-else: reads a bit guard and two same-typed branches.

`BV` implements [`Signature`] over [`Sort`], with matrices of scalars
`BV<N>(1, 1)` of any width `N`, written as the symbolic sort `'t`, e.g.,
the signature of [`BV::Add`] is `(Mat('t, a, b), Mat('t, a, b)) -> Mat('t, a, b)`,
and of [`BV::ULt`] is `(Mat('t, a, b), Mat('t, a, b)) -> Mat(BV<1>(1, 1), a, b)`.

`BV` implements [`Theory`]; [`Theory::check`] typechecks the sorts of the
read and write wires against the signature of the operation over the
bit-vectors of the width of the first bit-vector wire.

## Examples

```
use theory::Theory;
use theory::bv::{BV, Sort};

// 8-bit bit-vectors.
let a = Sort::BV(8, [2, 3]);
let b = Sort::BV(8, [3, 4]);
let c = Sort::BV(8, [2, 4]);

let ok = |s| Ok::<_, String>(s);

// Matrix multiply: (2x3) * (3x4) -> (2x4).
assert!(BV::MatMul().check([a, b].map(ok), [c].map(ok)).is_ok());

// Elementwise Add requires matching shapes.
let m = Sort::BV(8, [2, 3]);
assert!(BV::Add().check([m, m].map(ok), [m].map(ok)).is_ok());
assert!(BV::Add().check([a, b].map(ok), [c].map(ok)).is_err());
```
*/

use crate::signature::{Mat, MatSort, Signature, SignatureElem, TypeVar, VecSignature, sig};
use crate::sort::{MultiSort, TangentMultiSort};
use crate::*;
#[cfg(feature = "pyo3")]
use pyo3::pyclass;

use crate::bv::BV::{Havoc, Id};
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Sort {
    // matrix of bitvectors determined by (bitvector-length, [# rows, # cols])
    BV(usize, [usize; 2]),
    // the tangent of `BV` (it has only the zero derivative)
    Zero([usize; 2]),
}

impl Sort {
    pub fn shape(&self) -> &[usize; 2] {
        match self {
            Sort::BV(_, shape) | Sort::Zero(shape) => shape,
        }
    }

    /// The bit-width, `0` for the derivatives (zero-width bit-vectors have the only value 0)
    pub fn bw(&self) -> usize {
        match self {
            Sort::BV(bw, _) => *bw,
            Sort::Zero(_) => 0,
        }
    }

    /// Is this sort a tangent (i.e., a sort of derivatives)?
    pub fn is_tangent(&self) -> bool {
        matches!(self, Sort::Zero(..))
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::BV(bw, [i, j]) => write!(f, "BV<{bw}>({i}, {j})"),
            Sort::Zero([i, j]) => write!(f, "Zero({i}, {j})"),
        }
    }
}

impl MultiSort for Sort {
    const NAME: &'static str = "BVSort";
}

impl TangentMultiSort for Sort {
    fn T(&self) -> Self {
        Sort::Zero(*self.shape())
    }
}

/// Theory of bitvector matrices. Operations on bitvectors follow the SMT-LIB2 semantics.
#[derive(Clone, Debug, strum::Display)]
#[cfg_attr(feature = "pyo3", pyclass(frozen))]
pub enum BV {
    /// Create constant BV matrix from a given 2-D tensor
    #[strum(to_string = "{0}")]
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
    #[strum(to_string = "BitSelect[{high}:{low}]")]
    BitSelect {
        high: usize,
        low: usize,
    },
    /// Element-wise zero-extend by `extra` bits
    #[strum(to_string = "Extend(+{extra})")]
    Extend {
        extra: usize,
    },
    /// Uninterpreted symbol
    #[strum(to_string = "Uninterpreted({0})")]
    Uninterpreted(String),
    Havoc(usize, [usize; 2]),
}

impl Sequential for BV {
    fn skip(_range: &Self::Sort) -> Self {
        Id()
    }
}

impl Combinatorial for BV {
    fn havoc(range: &Self::Sort) -> Self {
        // a derivative cannot be havoced: the check of the term with `Zero` fails
        Havoc(range.bw(), *range.shape())
    }
}

impl Theory for BV {
    type Sort = Sort;
    const NAME: &'static str = "BV";

    fn check<R, W, E: fmt::Display>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<Sort, E>>,
        W: IntoIterator<Item = Result<Sort, E>>,
    {
        let check = || -> Result<(), String> {
            let (read, write) = (wire_sorts(read)?, wire_sorts(write)?);
            // the operation works on the bit-vectors of the width of the first bit-vector
            // wire (if there is none, the wires are derivatives that no width matches)
            let bw = read
                .iter()
                .chain(&write)
                .find(|s| !s.is_tangent())
                .map_or(0, Sort::bw);
            let mut sig = signature(self, Some(bw))?;
            // `Uninterpreted` either writes or reads a single value, but its signature
            // has the write
            if matches!(self, BV::Uninterpreted(_)) && !read.is_empty() {
                std::mem::swap(&mut sig.dom, &mut sig.cod);
            }
            typecheck_wires(sig, &read, &write, sort_elem)
        };
        check().map_err(|e| format!("{self:?}: {e}"))
    }
}

type Elem = SignatureElem<Sort>;

/// The signatures over the bit-vectors of any width. Matrices in the signatures are
/// matrices of scalars: `BV(n, [i, j])` is `Mat(BV(n, [1, 1]), i, j)`, and the
/// symbolic sort `'t` stands for `BV(n, [1, 1])` of any width `n`. The width of
/// the result of [`BV::Extend`] (`n + extra`) is not expressible, it is another `'u`.
///
/// # Panics
///
/// If the operation is ill-formed: the tensor of [`BV::Const`] is not 2D, or
/// `high < low` in [`BV::BitSelect`] ([`Theory::check`] reports these as errors).
impl Signature<Sort> for BV {
    fn dom(&self) -> impl Iterator<Item = Elem> {
        expect_signature(self).dom.into_iter()
    }

    fn cod(&self) -> impl Iterator<Item = Elem> {
        expect_signature(self).cod.into_iter()
    }

    fn dom_arity(&self) -> usize {
        expect_signature(self).dom.len()
    }

    fn cod_arity(&self) -> usize {
        expect_signature(self).cod.len()
    }
}

fn expect_signature(op: &BV) -> VecSignature<Sort> {
    signature(op, None).unwrap_or_else(|e| panic!("{op:?}: {e}"))
}

/// The signature of `op` over the bit-vectors of the width `bw`, or of any width if `bw`
/// is `None`. It is an error if `op` is ill-formed (for the width `bw`).
fn signature(op: &BV, bw: Option<usize>) -> Result<VecSignature<Sort>, String> {
    // the sorts of the elements of matrices: bit-vectors of the width `bw`, and bits
    let bv = |bw| MatSort::S(Sort::BV(bw, [1, 1]));
    let (elem, bit) = (bw.map_or(MatSort::from('t'), bv), bv(1));
    // a matrix with symbolic dimensions, the same for all (pointwise) arguments
    let mat = |s: MatSort<Sort>| Mat(s, 'a', 'b');
    Ok(match op {
        BV::Const(cm) => {
            let [rows, cols] = shape_2d(cm)?;
            if let Some(bw) = bw {
                check_fits(cm, bw)?;
            }
            sig!(() -> Mat(elem, rows, cols))
        }
        // TODO: for `Abs`, what if input is the signed min of BV<N>? It's absolute value
        // does not fit into BV<N>
        BV::Not() | BV::Neg() | BV::Abs() => sig!(mat(elem) -> mat(elem)),
        BV::Add()
        | BV::Sub()
        | BV::Mul()
        | BV::UDiv()
        | BV::SDiv()
        | BV::UMod()
        | BV::SMod()
        | BV::And()
        | BV::Or()
        | BV::Xor() => sig!((mat(elem), mat(elem)) -> mat(elem)),
        BV::ULe()
        | BV::ULt()
        | BV::UGe()
        | BV::UGt()
        | BV::SLe()
        | BV::SLt()
        | BV::SGe()
        | BV::SGt()
        | BV::Eq()
        | BV::Ne() => sig!((mat(elem), mat(elem)) -> mat(bit)),
        BV::MatMul() => sig!((Mat(elem, 'a', 'b'), Mat(elem, 'b', 'c')) -> Mat(elem, 'a', 'c')),
        BV::BVToBool() => sig!(mat(elem) -> mat(bit)),
        BV::BitSelect { high, low } => {
            if high < low {
                return Err(format!("high ({high}) must be >= low ({low})"));
            }
            if let Some(bw) = bw
                && *high >= bw
            {
                return Err(format!("high ({high}) is out of range for width {bw}"));
            }
            sig!(mat(elem) -> mat(bv(high - low + 1)))
        }
        BV::Extend { extra } => {
            let extended = bw.map_or(MatSort::from('u'), |bw| bv(bw + extra));
            sig!(mat(elem) -> mat(extended))
        }
        BV::Ite() => sig!((Mat(bit, 1, 1), TypeVar('a'), TypeVar('a')) -> TypeVar('a')),
        BV::Id() => sig!(TypeVar('a') -> TypeVar('a')),
        // `Uninterpreted` either writes or reads a single value, but a signature
        // has a fixed arity: we pick the write (used by the builder for untraced calls)
        BV::Uninterpreted(_) => sig!(() -> TypeVar('a')),
        BV::Havoc(width, shape) => sig!(() -> sort_elem(Sort::BV(*width, *shape))),
    })
}

/// The signature element of `sort`: a matrix of scalars
fn sort_elem(sort: Sort) -> Elem {
    match sort {
        Sort::BV(bw, [i, j]) => Mat(Sort::BV(bw, [1, 1]), i, j),
        Sort::Zero([i, j]) => Mat(Sort::Zero([1, 1]), i, j),
    }
}

/// Check that the values of the tensor `cm` fit into (unsigned) bit-vectors of the width `bw`
fn check_fits(cm: &crate::PyTensor, bw: usize) -> Result<(), String> {
    if bw > 64 {
        return Err("Support at most 64-bit atm.".into());
    }
    if bw < 64 {
        let max = (1i64 << bw) - 1;
        let min_val = cm.min().int64_value(&[]);
        let max_val = cm.max().int64_value(&[]);
        if min_val < 0 || max_val > max {
            return Err(format!("tensor values do not fit in {bw} bits"));
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    fn bv(bw: usize, r: usize, c: usize) -> Sort {
        Sort::BV(bw, [r, c])
    }

    pub fn ok<S>(s: S) -> Result<S, String> {
        Ok(s)
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
        assert!(BV::Const(cm).check([], [t].map(ok)).is_ok());
    }

    #[test]
    fn const_value_overflow_fails() {
        let t = bv(8, 1, 1);
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[256i64]]).into();
        assert!(BV::Const(cm).check(vec![], [t].map(ok)).is_err());
    }

    #[test]
    fn const_value_max_fits() {
        let t = bv(8, 1, 1);
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[255i64]]).into();
        assert!(BV::Const(cm).check(vec![], [t].map(ok)).is_ok());
    }

    #[test]
    fn const_wrong_row_count_fails() {
        let t = bv(8, 2, 2);
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0i64, 1]]).into();
        assert!(BV::Const(cm).check(vec![], [t].map(ok)).is_err());
    }

    #[test]
    fn const_wrong_col_count_fails() {
        let t = bv(8, 1, 2);
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0i64]]).into();
        assert!(BV::Const(cm).check(vec![], [t].map(ok)).is_err());
    }

    #[test]
    fn const_with_read_fails() {
        let t = bv(8, 1, 1);
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0i64]]).into();
        assert!(BV::Const(cm).check([t].map(ok), [t].map(ok)).is_err());
    }

    // --- Not / Id ---

    #[test]
    fn not_ok() {
        let t = bv(8, 2, 3);
        assert!(BV::Not().check([t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn not_type_mismatch_fails() {
        assert!(
            BV::Not()
                .check([bv(8, 1, 1)].map(ok), [bv(8, 2, 2)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn id_ok() {
        let t = bv(32, 4, 4);
        assert!(BV::Id().check([t].map(ok), [t].map(ok)).is_ok());
    }

    // --- Binary elementwise (Add, Mul, And, Or, Xor) ---

    #[test]
    fn add_ok() {
        let t = bv(8, 3, 3);
        assert!(BV::Add().check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn add_shape_mismatch_fails() {
        assert!(
            BV::Add()
                .check([bv(8, 1, 2), bv(8, 2, 1)].map(ok), [bv(8, 1, 2)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn mul_ok() {
        let t = bv(16, 1, 1);
        assert!(BV::Mul().check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn and_ok() {
        let t = bv(1, 2, 2);
        assert!(BV::And().check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn or_ok() {
        let t = bv(1, 2, 2);
        assert!(BV::Or().check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn xor_ok() {
        let t = bv(4, 1, 1);
        assert!(BV::Xor().check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    // --- Comparisons ---

    #[test]
    fn lt_ok() {
        let t = bv(8, 2, 3);
        let out = bv(1, 2, 3);
        assert!(BV::ULt().check([t, t].map(ok), [out].map(ok)).is_ok());
        assert!(BV::SLt().check([t, t].map(ok), [out].map(ok)).is_ok());
    }

    #[test]
    fn le_ok() {
        let t = bv(8, 1, 1);
        let out = bv(1, 1, 1);
        assert!(BV::ULe().check([t, t].map(ok), [out].map(ok)).is_ok());
        assert!(BV::SLe().check([t, t].map(ok), [out].map(ok)).is_ok());
    }

    #[test]
    fn eq_ok() {
        let t = bv(32, 1, 1);
        let out = bv(1, 1, 1);
        assert!(BV::Eq().check([t, t].map(ok), [out].map(ok)).is_ok());
    }

    #[test]
    fn ne_ok() {
        let t = bv(8, 1, 1);
        let out = bv(1, 1, 1);
        assert!(BV::Ne().check([t, t].map(ok), [out].map(ok)).is_ok());
    }

    #[test]
    fn cmp_wrong_output_type_fails() {
        let t = bv(8, 2, 3);
        // output must be U(1, rows, cols), not U(8, ...)
        assert!(BV::ULt().check([t, t].map(ok), [t].map(ok)).is_err());
        assert!(BV::SLt().check([t, t].map(ok), [t].map(ok)).is_err());
    }

    #[test]
    fn cmp_input_mismatch_fails() {
        let out = bv(1, 1, 1);
        assert!(
            BV::Eq()
                .check([bv(8, 1, 1), bv(16, 1, 1)].map(ok), [out].map(ok))
                .is_err()
        );
    }

    // --- UDiv / SDiv ---

    #[test]
    fn udiv_ok() {
        let t = bv(8, 1, 1);
        assert!(BV::UDiv().check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn sdiv_ok() {
        let t = bv(8, 1, 1);
        assert!(BV::SDiv().check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    // --- MatMul ---

    #[test]
    fn matmul_ok() {
        let a = bv(8, 2, 3);
        let b = bv(8, 3, 4);
        let c = bv(8, 2, 4);
        assert!(BV::MatMul().check([a, b].map(ok), [c].map(ok)).is_ok());
    }

    #[test]
    fn matmul_inner_dim_mismatch_fails() {
        let a = bv(8, 2, 3);
        let b = bv(8, 4, 4);
        let c = bv(8, 2, 4);
        assert!(BV::MatMul().check([a, b].map(ok), [c].map(ok)).is_err());
    }

    #[test]
    fn matmul_bw_mismatch_fails() {
        let a = bv(8, 2, 3);
        let b = bv(16, 3, 4);
        let c = bv(8, 2, 4);
        assert!(BV::MatMul().check([a, b].map(ok), [c].map(ok)).is_err());
    }

    // --- Ite ---

    #[test]
    fn ite_ok() {
        let cond = bv(1, 1, 1);
        let t = bv(8, 1, 1);
        assert!(BV::Ite().check([cond, t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn ite_bad_condition_bw_fails() {
        let cond = bv(8, 1, 1);
        let t = bv(8, 1, 1);
        assert!(BV::Ite().check([cond, t, t].map(ok), [t].map(ok)).is_err());
    }

    // --- Sub / Neg / UMod / SMod ---

    #[test]
    fn sub_ok() {
        let t = bv(8, 2, 3);
        assert!(BV::Sub().check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn neg_ok() {
        let t = bv(8, 2, 3);
        assert!(BV::Neg().check([t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn neg_type_mismatch_fails() {
        assert!(
            BV::Neg()
                .check([bv(8, 1, 1)].map(ok), [bv(8, 2, 2)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn umod_ok() {
        let t = bv(8, 1, 1);
        assert!(BV::UMod().check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn smod_ok() {
        let t = bv(8, 1, 1);
        assert!(BV::SMod().check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    // --- BVToBool / BoolToBV (SMV bool(...) / word1(...)) ---

    #[test]
    fn bv_to_bool_ok() {
        // BV<8>(2,3) -> BV<1>(2,3)
        assert!(
            BV::BVToBool()
                .check([bv(8, 2, 3)].map(ok), [bv(1, 2, 3)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn bv_to_bool_identity_width_ok() {
        // BV<1> -> BV<1> is also fine (idempotent on already-bool input).
        assert!(
            BV::BVToBool()
                .check([bv(1, 1, 1)].map(ok), [bv(1, 1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn bv_to_bool_wrong_out_width_fails() {
        assert!(
            BV::BVToBool()
                .check([bv(8, 1, 1)].map(ok), [bv(8, 1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn bv_to_bool_shape_mismatch_fails() {
        assert!(
            BV::BVToBool()
                .check([bv(8, 2, 3)].map(ok), [bv(1, 1, 1)].map(ok))
                .is_err()
        );
    }

    // --- BitSelect / Extend ---

    #[test]
    fn bit_select_ok() {
        // BV<16>[11:4] -> BV<8>
        assert!(
            BV::BitSelect { high: 11, low: 4 }
                .check([bv(16, 1, 1)].map(ok), [bv(8, 1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn bit_select_single_bit_ok() {
        assert!(
            BV::BitSelect { high: 3, low: 3 }
                .check([bv(8, 2, 2)].map(ok), [bv(1, 2, 2)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn bit_select_out_of_range_fails() {
        assert!(
            BV::BitSelect { high: 8, low: 0 }
                .check([bv(8, 1, 1)].map(ok), [bv(9, 1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn bit_select_wrong_out_width_fails() {
        assert!(
            BV::BitSelect { high: 7, low: 0 }
                .check([bv(16, 1, 1)].map(ok), [bv(7, 1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn extend_ok() {
        assert!(
            BV::Extend { extra: 8 }
                .check([bv(8, 1, 1)].map(ok), [bv(16, 1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn extend_wrong_out_width_fails() {
        assert!(
            BV::Extend { extra: 4 }
                .check([bv(8, 1, 1)].map(ok), [bv(16, 1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn ite_arm_mismatch_fails() {
        let cond = bv(1, 1, 1);
        assert!(
            BV::Ite()
                .check(
                    [cond, bv(8, 1, 1), bv(16, 1, 1)].map(ok),
                    [bv(8, 1, 1)].map(ok)
                )
                .is_err()
        );
    }

    #[test]
    fn ite_matrix_guard_fails() {
        // the guard is a single bit, as in LIA and LRA
        let t = bv(8, 2, 2);
        assert!(
            BV::Ite()
                .check([bv(1, 2, 2), t, t].map(ok), [t].map(ok))
                .is_err()
        );
    }

    // --- Signatures ---

    // Formats the signature of `op` as `(dom) -> (cod)`
    fn sig(op: BV) -> String {
        let join = |elems: Vec<String>| elems.join(", ");
        let dom = join(op.dom().map(|e| e.to_string()).collect());
        let cod = join(op.cod().map(|e| e.to_string()).collect());
        format!("({dom}) -> ({cod})")
    }

    fn int_tensor(rows: i64, cols: i64) -> crate::PyTensor {
        tch::Tensor::zeros([rows, cols], (tch::Kind::Int64, tch::Device::Cpu)).into()
    }

    #[test]
    fn signature_ops() {
        let m = "Mat('t, a, b)";
        for op in [
            BV::Add(),
            BV::Sub(),
            BV::Mul(),
            BV::UDiv(),
            BV::SDiv(),
            BV::UMod(),
            BV::SMod(),
            BV::And(),
            BV::Or(),
            BV::Xor(),
        ] {
            assert_eq!(sig(op), format!("({m}, {m}) -> ({m})"));
        }
        for op in [BV::Not(), BV::Neg(), BV::Abs()] {
            assert_eq!(sig(op), format!("({m}) -> ({m})"));
        }
        for op in [
            BV::ULe(),
            BV::ULt(),
            BV::UGe(),
            BV::UGt(),
            BV::SLe(),
            BV::SLt(),
            BV::SGe(),
            BV::SGt(),
            BV::Eq(),
            BV::Ne(),
        ] {
            assert_eq!(sig(op), format!("({m}, {m}) -> (Mat(BV<1>(1, 1), a, b))"));
        }
        assert_eq!(
            sig(BV::MatMul()),
            "(Mat('t, a, b), Mat('t, b, c)) -> (Mat('t, a, c))"
        );
        assert_eq!(
            sig(BV::BVToBool()),
            format!("({m}) -> (Mat(BV<1>(1, 1), a, b))")
        );
        assert_eq!(
            sig(BV::BitSelect { high: 11, low: 4 }),
            format!("({m}) -> (Mat(BV<8>(1, 1), a, b))")
        );
        // the width of the result is not expressible
        assert_eq!(
            sig(BV::Extend { extra: 8 }),
            format!("({m}) -> (Mat('u, a, b))")
        );
        assert_eq!(sig(BV::Ite()), "(Mat(BV<1>(1, 1), 1, 1), 'a, 'a) -> ('a)");
        assert_eq!(sig(BV::Id()), "('a) -> ('a)");
        assert_eq!(sig(BV::Uninterpreted("f".into())), "() -> ('a)");
        assert_eq!(sig(BV::Havoc(8, [2, 3])), "() -> (Mat(BV<8>(1, 1), 2, 3))");
        // the width is given by the written wire
        assert_eq!(sig(BV::Const(int_tensor(2, 3))), "() -> (Mat('t, 2, 3))");
    }

    #[test]
    #[should_panic(expected = "high (3) must be >= low (4)")]
    fn signature_bit_select_ill_formed() {
        let _ = sig(BV::BitSelect { high: 3, low: 4 });
    }

    #[test]
    fn signature_arity() {
        assert_eq!((BV::Ite().dom_arity(), BV::Ite().cod_arity()), (3, 1));
        assert_eq!((BV::MatMul().dom_arity(), BV::MatMul().cod_arity()), (2, 1));
        let havoc = BV::Havoc(8, [1, 1]);
        assert_eq!((havoc.dom_arity(), havoc.cod_arity()), (0, 1));
    }

    #[test]
    fn signature_typecheck() {
        use crate::signature::typecheck;
        assert_eq!(typecheck(BV::Add(), BV::Xor()), Ok(()));
        // `'a -> 'a` is instantiated by `Mat('t, a, b) -> Mat('t, a, b)`
        assert_eq!(typecheck(BV::Id(), BV::Not()), Ok(()));
        // `'t` is instantiated by `BV<8>(1, 1)`
        assert_eq!(
            typecheck(BV::Havoc(8, [2, 3]), BV::Const(int_tensor(2, 3))),
            Ok(())
        );
        assert_eq!(
            typecheck(BV::Havoc(8, [2, 3]), BV::Havoc(16, [2, 3])),
            Err(
                "Failed unifying `() -> Mat(BV<8>(1, 1), 2, 3)` with `() -> Mat(BV<16>(1, 1), 2, 3)` \
                 on `Mat(BV<8>(1, 1), 2, 3)` and `Mat(BV<16>(1, 1), 2, 3)`"
                    .to_string()
            )
        );
        assert!(typecheck(BV::BitSelect { high: 7, low: 0 }, BV::BVToBool()).is_err());
    }

    #[test]
    fn check_errors() {
        assert_eq!(
            BV::Add().check([bv(8, 1, 1), bv(16, 1, 1)].map(ok), [bv(8, 1, 1)].map(ok)),
            Err(
                "Add: Failed unifying `(Mat(BV<8>(1, 1), a, b), Mat(BV<8>(1, 1), a, b)) -> Mat(BV<8>(1, 1), a, b)` \
                 with `(Mat(BV<8>(1, 1), 1, 1), Mat(BV<16>(1, 1), 1, 1)) -> Mat(BV<8>(1, 1), 1, 1)` \
                 on `Mat(BV<8>(1, 1), 1, 1)` and `Mat(BV<16>(1, 1), 1, 1)`"
                    .to_string()
            )
        );
        let t = bv(8, 1, 1);
        assert_eq!(
            BV::Not().check([t, t].map(ok), [t].map(ok)),
            Err("Not: Signatures domains have different arity (1 != 2)".to_string())
        );
        // errors of wires are propagated
        assert_eq!(
            BV::Not().check([Err("bad wire")], [t].map(Ok)),
            Err("Not: bad wire".to_string())
        );
    }

    #[test]
    fn check_ill_formed_ops() {
        // ill-formed ops are reported, not panicking
        let t: crate::PyTensor = tch::Tensor::from_slice(&[0i64, 1]).into();
        let res = BV::Const(t).check([].map(ok), [bv(8, 1, 2)].map(ok));
        assert!(res.unwrap_err().ends_with("expected a 2D tensor, got 1D"));
        let res = BV::Const(int_tensor(1, 1)).check([].map(ok), [bv(65, 1, 1)].map(ok));
        assert!(res.unwrap_err().ends_with("Support at most 64-bit atm."));
        let res =
            BV::BitSelect { high: 3, low: 4 }.check([bv(8, 1, 1)].map(ok), [bv(1, 1, 1)].map(ok));
        assert!(res.unwrap_err().ends_with("high (3) must be >= low (4)"));
        let res =
            BV::BitSelect { high: 8, low: 0 }.check([bv(8, 1, 1)].map(ok), [bv(9, 1, 1)].map(ok));
        assert!(
            res.unwrap_err()
                .ends_with("high (8) is out of range for width 8")
        );
    }

    #[test]
    fn check_uninterpreted() {
        let f = || BV::Uninterpreted("f".into());
        let t = bv(8, 2, 3);
        // either a single write, or a single read of any sort
        assert!(f().check([].map(ok), [t].map(ok)).is_ok());
        assert!(f().check([t].map(ok), [].map(ok)).is_ok());
        assert!(f().check([bv(1, 1, 1)].map(ok), [].map(ok)).is_ok());
        assert!(f().check([t].map(ok), [t].map(ok)).is_err());
        assert!(f().check([t, t].map(ok), [].map(ok)).is_err());
        assert!(f().check([].map(ok), [t, t].map(ok)).is_err());
        assert!(f().check([].map(ok), [].map(ok)).is_err());
    }

    #[test]
    fn check_derivatives() {
        // the derivatives of bit-vectors are zero
        let z = bv(8, 2, 3).T();
        assert_eq!(z, Sort::Zero([2, 3]));
        assert_eq!(z.T(), z);
        // operations on values do not apply to derivatives
        let t = bv(8, 2, 3);
        assert!(BV::Add().check([z, z].map(ok), [z].map(ok)).is_err());
        assert!(BV::Add().check([t, z].map(ok), [t].map(ok)).is_err());
        assert!(BV::Not().check([z].map(ok), [t].map(ok)).is_err());
        assert!(BV::Havoc(8, [2, 3]).check([].map(ok), [z].map(ok)).is_err());
        assert!(
            BV::Const(int_tensor(2, 3))
                .check([].map(ok), [z].map(ok))
                .is_err()
        );
        assert!(BV::Id().check([t].map(ok), [z].map(ok)).is_err());
    }
}
