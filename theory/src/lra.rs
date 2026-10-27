/*!
# Linear real arithmetic

Defines the theory [`LRA`] of linear real arithmetic over matrices,
mixing real and boolean matrices in a single signature.

A [`Sort`] value is either `Real(rows, cols)` or `Bool(rows, cols)`, or
the sort of their derivatives: `TReal(rows, cols; d)` of the degree `d >= 1`
for reals and `Zero(rows, cols)` for booleans (see [`TangentMultiSort::T`]).
The operations in [`LRA`] are:

- [`LRA::Real`], [`LRA::Bool`] — real and boolean matrix literals.
- [`LRA::And`], [`LRA::Or`], [`LRA::Xor`], [`LRA::Not`]
  — boolean operations.
- [`LRA::Le`], [`LRA::Lt`], [`LRA::Ge`], [`LRA::Gt`], [`LRA::Eq`], [`LRA::Ne`]
  — pointwise real comparisons producing a boolean matrix.
- [`LRA::Ite`] — if-then-else: reads a boolean guard and two same-typed branches.
- [`LRA::Linear`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and
  `B` constant real matrices of compatible shapes.
- [`LRA::ReLU`] — the shape-preserving rectified-linear map on real matrices.
- [`LRA::RealZerograd`], [`LRA::BoolZerograd`] — the zero derivatives.

`LRA` implements [`Signature`] over [`Sort`], with matrices of scalars
`Real(1, 1)` and `Bool(1, 1)`, e.g., the signature of [`LRA::ReLU`]
is `Mat(Real(1, 1), a, b) -> Mat(Real(1, 1), a, b)`. These are the signatures
over values: the arithmetic operations also work on the derivatives of reals
(of the same degree), e.g., `ReLU` also has the signature
`Mat(TReal(1, 1; 1), a, b) -> Mat(TReal(1, 1; 1), a, b)`.

`LRA` implements [`Theory`]; [`Theory::check`] typechecks the sorts of the
read and write wires against the signature of the operation (over values, or
over the derivatives of the degree of the first real wire).

## Examples

```
use theory::Theory;
use theory::lra::{LRA, Sort};

// Wires carry a sort (derivatives carry the tangent sorts).
let ok = |s| Ok::<_, String>(s);

// Pointwise less-than on scalars: Real(1,1), Real(1,1) -> Bool(1,1).
let i = Sort::Real([1, 1]);
let b = Sort::Bool([1, 1]);
assert!(LRA::Lt().check([i, i].map(ok), [b].map(ok)).is_ok());

// ReLU preserves shape and stays in the real fragment.
let m = Sort::Real([3, 4]);
assert!(LRA::ReLU().check([m].map(ok), [m].map(ok)).is_ok());
assert!(LRA::ReLU().check([b].map(ok), [b].map(ok)).is_err());
```
*/

use crate::signature::{Mat, Signature, SignatureElem, TypeVar, VecSignature};
use crate::sort::{MultiSort, TangentMultiSort};
use crate::*;
#[cfg(feature = "pyo3")]
use pyo3::pyclass;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Sort {
    Real([usize; 2]),
    Bool([usize; 2]),
    // the tangent of `Real` of the given degree (>= 1), i.e., its derivative
    TReal([usize; 2], u8),
    // the tangent of `Bool` (it has only the zero derivative)
    Zero([usize; 2]),
}

impl Sort {
    pub fn is_bool(&self) -> bool {
        matches!(self, Sort::Bool(..))
    }

    pub fn is_real(&self) -> bool {
        matches!(self, Sort::Real(..))
    }

    /// Is this sort a tangent (i.e., a sort of derivatives)?
    pub fn is_tangent(&self) -> bool {
        matches!(self, Sort::TReal(..) | Sort::Zero(..))
    }

    pub fn shape(&self) -> &[usize; 2] {
        match self {
            Sort::Bool(shape) | Sort::Real(shape) | Sort::TReal(shape, _) | Sort::Zero(shape) => {
                shape
            }
        }
    }

    /// The degree of a real sort (`0`) or of its tangent, `None` for other sorts
    fn real_degree(&self) -> Option<u8> {
        match self {
            Sort::Real(_) => Some(0),
            Sort::TReal(_, degree) => Some(*degree),
            _ => None,
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Real([i, j]) => write!(f, "Real({i}, {j})"),
            Sort::Bool([i, j]) => write!(f, "Bool({i}, {j})"),
            Sort::TReal([i, j], d) => write!(f, "TReal({i}, {j}; {d})"),
            Sort::Zero([i, j]) => write!(f, "Zero({i}, {j})"),
        }
    }
}

impl MultiSort for Sort {
    const NAME: &'static str = "LRASort";
}

impl TangentMultiSort for Sort {
    fn T(&self) -> Self {
        match self {
            Sort::Real(shape) => Sort::TReal(*shape, 1),
            Sort::TReal(shape, d) => Sort::TReal(*shape, d + 1),
            Sort::Bool(shape) | Sort::Zero(shape) => Sort::Zero(*shape),
        }
    }
}

#[derive(Clone, Debug, strum::Display)]
#[cfg_attr(feature = "pyo3", pyclass(frozen))]
pub enum LRA {
    // real and boolean matrix literals
    #[strum(to_string = "{0}")]
    Real(crate::PyTensor),
    #[strum(to_string = "{0}")]
    Bool(crate::PyTensor),
    // boolean operations
    And(),
    Or(),
    Xor(),
    Not(),
    // real comparisons
    Le(),
    Lt(),
    Ge(),
    Gt(),
    Eq(),
    Ne(),
    // linear / matrix operations
    // A*x + B where `A` and `B` are constants
    #[strum(to_string = "Linear({0}, {1})")]
    Linear(crate::PyTensor, crate::PyTensor),
    Add(),
    Sub(),
    // XXX: should these be in RLA?
    ReLU(),
    Argmax(),
    Min(),
    Max(),
    // matrix operations
    Transpose(),
    // control flow
    Ite(),
    Id(),
    #[strum(to_string = "Uninterpreted({0})")]
    Uninterpreted(String),
    BoolZerograd([usize; 2]), // unstable
    RealZerograd([usize; 2]), // unstable
    AnyBool([usize; 2]),
    AnyReal([usize; 2]),
}

impl Sequential for LRA {
    fn skip(_range: &Self::Sort) -> Self {
        LRA::Id()
    }
}

impl Combinatorial for LRA {
    fn havoc(range: &Self::Sort) -> Self {
        // a derivative cannot be havoced: the check of the term with a tangent fails
        match range {
            Sort::Bool(shape) | Sort::Zero(shape) => LRA::AnyBool(*shape),
            Sort::Real(shape) | Sort::TReal(shape, _) => LRA::AnyReal(*shape),
        }
    }
}

impl Differential for LRA {
    fn zero(range: &Self::Sort) -> Self {
        // the range is the sort of the derivative; values have no zero derivative,
        // so the check of the term with them fails
        match range {
            Sort::Bool(shape) | Sort::Zero(shape) => LRA::BoolZerograd(*shape),
            Sort::Real(shape) | Sort::TReal(shape, _) => LRA::RealZerograd(*shape),
        }
    }
}

impl Theory for LRA {
    type Sort = Sort;
    const NAME: &'static str = "LRA";

    fn check<R, W, E: fmt::Display>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<Sort, E>>,
        W: IntoIterator<Item = Result<Sort, E>>,
    {
        let check = || -> Result<(), String> {
            let (read, write) = (wire_sorts(read)?, wire_sorts(write)?);
            // the operation works on values, or on derivatives (of the first real wire)
            let degree = read
                .iter()
                .chain(&write)
                .find_map(Sort::real_degree)
                .unwrap_or(0);
            let mut sig = signature(self, degree)?;
            // `Uninterpreted` either writes or reads a single value, but its signature
            // has the write
            if matches!(self, LRA::Uninterpreted(_)) && !read.is_empty() {
                std::mem::swap(&mut sig.dom, &mut sig.cod);
            }
            typecheck_wires(sig, &read, &write, sort_elem)
        };
        check().map_err(|e| format!("{self:?}: {e}"))
    }
}

type Elem = SignatureElem<Sort>;

/// The signatures over values. Matrices in the signatures are matrices of scalars:
/// `Real([i, j])` is `Mat(Real([1, 1]), i, j)`, `TReal([i, j], d)` is
/// `Mat(TReal([1, 1], d), i, j)`, and similarly for `Bool` and `Zero`.
///
/// # Panics
///
/// If the operation is ill-formed: its tensors are not 2D, or `B` of [`LRA::Linear`]
/// is not a column vector ([`Theory::check`] reports these as errors).
impl Signature<Sort> for LRA {
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

fn expect_signature(op: &LRA) -> VecSignature<Sort> {
    signature(op, 0).unwrap_or_else(|e| panic!("{op:?}: {e}"))
}

/// The signature of `op` over values (`degree` 0), or over the derivatives of reals of
/// the given `degree`, e.g., `Add` adds them, and `Lt` compares them (into values).
/// It is an error if `op` is ill-formed.
fn signature(op: &LRA, degree: u8) -> Result<VecSignature<Sort>, String> {
    // the sorts of the elements of matrices: values, and reals of the `degree`
    let (real0, bool) = (Sort::Real([1, 1]), Sort::Bool([1, 1]));
    let real = (0..degree).fold(real0, |s, _| s.T());
    // a matrix with symbolic dimensions, the same for all (pointwise) arguments
    let m = |s: Sort| Mat(s, 'a', 'b');
    let (dom, cod) = match op {
        LRA::Real(cm) => {
            let [rows, cols] = shape_2d(cm)?;
            (vec![], vec![Mat(real0, rows, cols)])
        }
        LRA::Bool(cm) => {
            let [rows, cols] = shape_2d(cm)?;
            (vec![], vec![Mat(bool, rows, cols)])
        }
        LRA::And() | LRA::Or() | LRA::Xor() => (vec![m(bool), m(bool)], vec![m(bool)]),
        LRA::Not() => (vec![m(bool)], vec![m(bool)]),
        LRA::Le() | LRA::Lt() | LRA::Ge() | LRA::Gt() | LRA::Eq() | LRA::Ne() => {
            (vec![m(real), m(real)], vec![m(bool)])
        }
        LRA::Linear(a, b) => {
            // Y = A·X + B  where X=[in,batch], A=[out,in], B=[out,1] (or empty), Y=[out,batch]
            let [out, inp] = shape_2d(a)?;
            if b.numel() != 0 && shape_2d(b)? != [out, 1] {
                return Err(format!("`B` must be a column vector [{out}x1]"));
            }
            (vec![Mat(real, inp, 'b')], vec![Mat(real, out, 'b')])
        }
        // `Min` and `Max` are pointwise, as in the evaluator
        LRA::Add() | LRA::Sub() | LRA::Min() | LRA::Max() => {
            (vec![m(real), m(real)], vec![m(real)])
        }
        LRA::ReLU() => (vec![m(real)], vec![m(real)]),
        // the index into the flattened matrix, as in the evaluator and the Python builder
        LRA::Argmax() => (vec![m(real)], vec![Mat(real, 1, 1)]),
        // of any sort, as in the evaluator
        LRA::Transpose() => (vec![Mat('t', 'a', 'b')], vec![Mat('t', 'b', 'a')]),
        LRA::Ite() => (
            vec![Mat(bool, 1, 1), TypeVar('a'), TypeVar('a')],
            vec![TypeVar('a')],
        ),
        LRA::Id() => (vec![TypeVar('a')], vec![TypeVar('a')]),
        // `Uninterpreted` either writes or reads a single value, but a signature
        // has a fixed arity: we pick the write (used by the builder for untraced calls)
        LRA::Uninterpreted(_) => (vec![], vec![TypeVar('a')]),
        // the zero derivatives
        LRA::BoolZerograd(shape) => (vec![], vec![sort_elem(Sort::Bool(*shape).T())]),
        LRA::RealZerograd(shape) => (vec![], vec![sort_elem(Sort::Real(*shape).T())]),
        LRA::AnyBool(shape) => (vec![], vec![sort_elem(Sort::Bool(*shape))]),
        LRA::AnyReal(shape) => (vec![], vec![sort_elem(Sort::Real(*shape))]),
    };
    Ok(VecSignature { dom, cod })
}

/// The signature element of `sort`: a matrix of scalars
fn sort_elem(sort: Sort) -> Elem {
    match sort {
        Sort::Real([i, j]) => Mat(Sort::Real([1, 1]), i, j),
        Sort::Bool([i, j]) => Mat(Sort::Bool([1, 1]), i, j),
        Sort::TReal([i, j], degree) => Mat(Sort::TReal([1, 1], degree), i, j),
        Sort::Zero([i, j]) => Mat(Sort::Zero([1, 1]), i, j),
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    fn real(r: usize, c: usize) -> Sort {
        Sort::Real([r, c])
    }

    fn bool_t(r: usize, c: usize) -> Sort {
        Sort::Bool([r, c])
    }

    fn ok(s: Sort) -> Result<Sort, String> {
        Ok(s)
    }

    #[test]
    fn type_kind_and_shape() {
        assert!(real(2, 3).is_real() && !real(2, 3).is_bool());
        assert_eq!(real(2, 3).shape(), &[2, 3]);
        assert!(bool_t(1, 1).is_bool() && !bool_t(1, 1).is_real());
    }

    #[test]
    fn const_real_ok() {
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0.0f64, 1.0], [2.0, 3.0]]).into();
        assert!(
            LRA::Real(cm)
                .check([].map(ok), [real(2, 2)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn const_real_covector_write_fails() {
        // a real constant writes a value (0-form), never a derivative:
        // a covector (1-form) write wire must be rejected
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0.0f64]]).into();
        assert!(
            LRA::Real(cm)
                .check([].map(ok), [real(1, 1).T()].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_real_bool_write_fails() {
        assert!(
            LRA::Real(tch::Tensor::from_slice2(&[[0.0f64]]).into())
                .check([].map(ok), [bool_t(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_real_wrong_rows_fails() {
        assert!(
            LRA::Real(tch::Tensor::from_slice2(&[[0.0f64]]).into())
                .check([].map(ok), [real(2, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_real_with_read_fails() {
        let t = real(1, 1);
        assert!(
            LRA::Real(tch::Tensor::from_slice2(&[[0.0f64]]).into())
                .check([t].map(ok), [t].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_bool_ok() {
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[true, false], [false, true]]).into();
        assert!(
            LRA::Bool(cm)
                .check([].map(ok), [bool_t(2, 2)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn const_bool_real_write_fails() {
        assert!(
            LRA::Bool(tch::Tensor::from_slice2(&[[true]]).into())
                .check([].map(ok), [real(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn not_ok() {
        let b = bool_t(2, 3);
        assert!(LRA::Not().check([b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn not_real_input_fails() {
        let t = real(1, 1);
        assert!(LRA::Not().check([t].map(ok), [t].map(ok)).is_err());
    }

    #[test]
    fn and_ok() {
        let b = bool_t(2, 2);
        assert!(LRA::And().check([b, b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn or_ok() {
        let b = bool_t(1, 1);
        assert!(LRA::Or().check([b, b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn xor_ok() {
        let b = bool_t(3, 1);
        assert!(LRA::Xor().check([b, b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn and_real_output_fails() {
        let b = bool_t(1, 1);
        assert!(
            LRA::And()
                .check([b, b].map(ok), [real(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn and_type_mismatch_fails() {
        assert!(
            LRA::And()
                .check([bool_t(1, 1), bool_t(1, 2)].map(ok), [bool_t(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn lt_ok() {
        assert!(
            LRA::Lt()
                .check([real(1, 1), real(1, 1)].map(ok), [bool_t(1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn le_ok() {
        assert!(
            LRA::Le()
                .check([real(2, 3), real(2, 3)].map(ok), [bool_t(2, 3)].map(ok))
                .is_ok()
        );

        assert!(
            LRA::Le()
                .check([real(3, 3), real(2, 3)].map(ok), [bool_t(2, 3)].map(ok))
                .is_err()
        );

        assert!(
            LRA::Le()
                .check([real(2, 3), real(2, 3)].map(ok), [bool_t(3, 3)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn eq_ok() {
        assert!(
            LRA::Eq()
                .check([real(2, 2), real(2, 2)].map(ok), [bool_t(2, 2)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn cmp_non_bool_output_fails() {
        let t = real(1, 1);
        assert!(LRA::Lt().check([t, t].map(ok), [t].map(ok)).is_err());
    }

    #[test]
    fn cmp_input_mismatch_fails() {
        assert!(
            LRA::Eq()
                .check([real(1, 1), real(1, 2)].map(ok), [bool_t(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn add_ok() {
        let t = real(3, 4);
        assert!(LRA::Add().check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn add_shape_mismatch_fails() {
        assert!(
            LRA::Add()
                .check([real(1, 2), real(2, 1)].map(ok), [real(1, 2)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn add_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LRA::Add().check([b, b].map(ok), [b].map(ok)).is_err());
    }

    #[test]
    fn relu_ok() {
        let t = real(3, 4);
        assert!(LRA::ReLU().check([t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn relu_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LRA::ReLU().check([b].map(ok), [b].map(ok)).is_err());
    }

    #[test]
    fn argmax_ok() {
        assert!(
            LRA::Argmax()
                .check([real(3, 4)].map(ok), [real(1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn argmax_matrix_output_fails() {
        assert!(
            LRA::Argmax()
                .check([real(3, 4)].map(ok), [real(3, 4)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn min_ok() {
        // pointwise
        let t = real(4, 1);
        assert!(LRA::Min().check([t, t].map(ok), [t].map(ok)).is_ok());
        assert!(LRA::Max().check([t].map(ok), [real(1, 1)].map(ok)).is_err());
    }

    #[test]
    fn linear_ok() {
        // A=[2,3] maps 3 features to 2; X=[3,4] is 3 features × 4 batch items.
        // Convention: Y = A·X  →  Y=[2,4].
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Double, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([0, 0], (tch::Kind::Double, tch::Device::Cpu)).into();
        assert!(
            LRA::Linear(a, b)
                .check([real(3, 4)].map(ok), [real(2, 4)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn linear_with_bias_ok() {
        // A=[2,3], b=[2,1] column bias, X=[3,1] single sample → Y=[2,1].
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Double, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([2, 1], (tch::Kind::Double, tch::Device::Cpu)).into();
        assert!(
            LRA::Linear(a, b)
                .check([real(3, 1)].map(ok), [real(2, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn linear_dim_mismatch_fails() {
        // A=[2,3] but X has 4 rows — inner dimension mismatch.
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Double, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([0, 0], (tch::Kind::Double, tch::Device::Cpu)).into();
        assert!(
            LRA::Linear(a, b)
                .check([real(4, 1)].map(ok), [real(2, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn transpose_ok() {
        assert!(
            LRA::Transpose()
                .check([real(3, 4)].map(ok), [real(4, 3)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn transpose_wrong_shape_fails() {
        assert!(
            LRA::Transpose()
                .check([real(3, 4)].map(ok), [real(3, 4)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn ite_ok() {
        let t = real(3, 4);
        assert!(
            LRA::Ite()
                .check([bool_t(1, 1), t, t].map(ok), [t].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn ite_non_bool_guard_fails() {
        let t = real(1, 1);
        assert!(LRA::Ite().check([t, t, t].map(ok), [t].map(ok)).is_err());
    }

    #[test]
    fn ite_arm_mismatch_fails() {
        assert!(
            LRA::Ite()
                .check(
                    [bool_t(1, 1), real(1, 1), real(1, 2)].map(ok),
                    [real(1, 1)].map(ok)
                )
                .is_err()
        );
    }

    #[test]
    fn id_ok() {
        let t = real(4, 4);
        assert!(LRA::Id().check([t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn id_type_mismatch_fails() {
        assert!(
            LRA::Id()
                .check([real(1, 1)].map(ok), [real(2, 2)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn id_arity_mismatch_fails() {
        let t = real(1, 1);
        assert!(LRA::Id().check([t, t].map(ok), [t, t].map(ok)).is_err());
        assert!(LRA::Id().check([t, t].map(ok), [t].map(ok)).is_err());
        assert!(LRA::Id().check([t].map(ok), [t, t].map(ok)).is_err());
    }

    #[test]
    fn havoc_ok() {
        assert!(
            LRA::AnyReal([2, 1])
                .check([].map(ok), [real(2, 1)].map(ok))
                .is_ok()
        );
        assert!(
            LRA::AnyBool([1, 1])
                .check([].map(ok), [bool_t(1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn havoc_read_fails() {
        let t = real(1, 1);
        assert!(
            LRA::AnyReal([1, 1])
                .check([t].map(ok), [t].map(ok))
                .is_err()
        );
    }

    #[test]
    fn havoc_arity_mismatch_fails() {
        assert!(
            LRA::AnyReal([2, 1])
                .check([].map(ok), [real(2, 1), bool_t(1, 1)].map(ok))
                .is_err()
        );
        assert!(LRA::AnyReal([1, 1]).check([].map(ok), [].map(ok)).is_err());
        // the write sort must match the op's declared range
        assert!(
            LRA::AnyReal([2, 1])
                .check([].map(ok), [real(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn zero_ok() {
        // the zero derivative is written to the derivative (a tangent), not to a value
        let zero = |op: LRA, s: Sort| op.check([].map(ok), [s].map(ok));
        assert!(zero(LRA::RealZerograd([2, 1]), real(2, 1).T()).is_ok());
        assert!(zero(LRA::RealZerograd([2, 1]), real(2, 1)).is_err());
        assert!(zero(LRA::RealZerograd([2, 1]), real(2, 1).T().T()).is_err());
        assert!(zero(LRA::BoolZerograd([2, 1]), bool_t(2, 1).T()).is_ok());
        assert!(zero(LRA::BoolZerograd([2, 1]), bool_t(2, 1)).is_err());
        // `zero` is given the sort of the derivative
        assert!(matches!(
            LRA::zero(&real(2, 1).T()),
            LRA::RealZerograd([2, 1])
        ));
        assert!(matches!(
            LRA::zero(&bool_t(2, 1).T()),
            LRA::BoolZerograd([2, 1])
        ));
    }

    #[test]
    fn derivatives() {
        let (r, b) = (real(2, 1), bool_t(2, 1));
        assert_eq!(r.T(), Sort::TReal([2, 1], 1));
        assert_eq!(r.T().T(), Sort::TReal([2, 1], 2));
        assert_eq!(b.T(), Sort::Zero([2, 1]));
        assert_eq!(b.T().T(), Sort::Zero([2, 1]));
        // arithmetic works on derivatives of the same degree
        let dr = r.T();
        assert!(LRA::Add().check([dr, dr].map(ok), [dr].map(ok)).is_ok());
        assert!(LRA::ReLU().check([dr].map(ok), [dr].map(ok)).is_ok());
        assert!(LRA::Id().check([dr].map(ok), [dr].map(ok)).is_ok());
        assert!(LRA::Lt().check([dr, dr].map(ok), [b].map(ok)).is_ok());
        assert!(LRA::Add().check([dr, r].map(ok), [dr].map(ok)).is_err());
        assert!(
            LRA::Add()
                .check([dr, dr].map(ok), [dr.T()].map(ok))
                .is_err()
        );
        // `dx = x` mixes a derivative and a value
        assert!(LRA::Id().check([r].map(ok), [dr].map(ok)).is_err());
        // boolean operations and guards are on values only
        assert!(LRA::Not().check([b.T()].map(ok), [b.T()].map(ok)).is_err());
        let g = bool_t(1, 1);
        assert!(
            LRA::Ite()
                .check([g.T(), dr, dr].map(ok), [dr].map(ok))
                .is_err()
        );
        assert!(LRA::Ite().check([g, dr, dr].map(ok), [dr].map(ok)).is_ok());
    }

    #[test]
    fn zero_arity_mismatch_fails() {
        let t = real(1, 1);
        assert!(
            LRA::RealZerograd([1, 1])
                .check([t].map(ok), [t].map(ok))
                .is_err()
        );
        assert!(
            LRA::RealZerograd([2, 1])
                .check([].map(ok), [real(2, 1), real(1, 1)].map(ok))
                .is_err()
        );
        assert!(
            LRA::RealZerograd([1, 1])
                .check([].map(ok), [].map(ok))
                .is_err()
        );
    }

    fn real_tensor(rows: i64, cols: i64) -> crate::PyTensor {
        tch::Tensor::zeros([rows, cols], (tch::Kind::Float, tch::Device::Cpu)).into()
    }

    // Formats the signature of `op` as `(dom) -> (cod)`
    fn sig(op: LRA) -> String {
        let join = |elems: Vec<String>| elems.join(", ");
        let dom = join(op.dom().map(|e| e.to_string()).collect());
        let cod = join(op.cod().map(|e| e.to_string()).collect());
        format!("({dom}) -> ({cod})")
    }

    #[test]
    fn signatures() {
        let r = "Mat(Real(1, 1), a, b)";
        assert_eq!(sig(LRA::Add()), format!("({r}, {r}) -> ({r})"));
        assert_eq!(
            sig(LRA::Lt()),
            format!("({r}, {r}) -> (Mat(Bool(1, 1), a, b))")
        );
        assert_eq!(
            sig(LRA::Linear(real_tensor(2, 3), real_tensor(0, 0))),
            "(Mat(Real(1, 1), 3, b)) -> (Mat(Real(1, 1), 2, b))"
        );
        assert_eq!(
            sig(LRA::RealZerograd([2, 3])),
            "() -> (Mat(TReal(1, 1; 1), 2, 3))"
        );
        assert_eq!(
            sig(LRA::BoolZerograd([2, 3])),
            "() -> (Mat(Zero(1, 1), 2, 3))"
        );
        assert_eq!(sig(LRA::AnyReal([2, 3])), "() -> (Mat(Real(1, 1), 2, 3))");
        assert_eq!(sig(LRA::Transpose()), "(Mat('t, a, b)) -> (Mat('t, b, a))");
        // over derivatives
        let dr = "Mat(TReal(1, 1; 1), a, b)";
        let sig1 = |op| {
            let sig = signature(&op, 1).unwrap();
            (
                sig.dom.len(),
                sig.cod.iter().map(|e| e.to_string()).collect::<Vec<_>>(),
            )
        };
        assert_eq!(sig1(LRA::Add()), (2, vec![dr.to_string()]));
        assert_eq!(
            sig1(LRA::Lt()),
            (2, vec!["Mat(Bool(1, 1), a, b)".to_string()])
        );
    }

    #[test]
    fn check_errors() {
        assert_eq!(
            LRA::Add().check([real(1, 1), real(1, 2)].map(ok), [real(1, 1)].map(ok)),
            Err(
                "Add: Failed unifying `(Mat(Real(1, 1), a, b), Mat(Real(1, 1), a, b)) -> Mat(Real(1, 1), a, b)` \
                 with `(Mat(Real(1, 1), 1, 1), Mat(Real(1, 1), 1, 2)) -> Mat(Real(1, 1), 1, 1)` \
                 on `Mat(Real(1, 1), 1, 1)` and `Mat(Real(1, 1), 1, 2)`"
                    .to_string()
            )
        );
        // over derivatives, the signature is over the tangents
        let dr = real(1, 1).T();
        assert_eq!(
            LRA::Add().check([dr, real(1, 1)].map(ok), [dr].map(ok)),
            Err(
                "Add: Failed unifying `(Mat(TReal(1, 1; 1), a, b), Mat(TReal(1, 1; 1), a, b)) -> Mat(TReal(1, 1; 1), a, b)` \
                 with `(Mat(TReal(1, 1; 1), 1, 1), Mat(Real(1, 1), 1, 1)) -> Mat(TReal(1, 1; 1), 1, 1)` \
                 on `Mat(TReal(1, 1; 1), 1, 1)` and `Mat(Real(1, 1), 1, 1)`"
                    .to_string()
            )
        );
        // errors of wires are propagated
        let b = bool_t(1, 1);
        assert_eq!(
            LRA::Not().check([Err("bad wire")], [b].map(Ok)),
            Err("Not: bad wire".to_string())
        );
    }

    #[test]
    fn check_ill_formed_ops() {
        // ill-formed tensors are reported, not panicking
        let t: crate::PyTensor = tch::Tensor::from_slice(&[0.0f64, 1.0]).into();
        let res = LRA::Real(t).check([].map(ok), [real(1, 2)].map(ok));
        assert!(res.unwrap_err().ends_with("expected a 2D tensor, got 1D"));
        let res = LRA::Linear(real_tensor(2, 3), real_tensor(2, 2))
            .check([real(3, 1)].map(ok), [real(2, 1)].map(ok));
        assert!(
            res.unwrap_err()
                .ends_with("`B` must be a column vector [2x1]")
        );
    }

    #[test]
    fn check_uninterpreted() {
        let f = || LRA::Uninterpreted("f".into());
        let t = real(2, 3);
        // either a single write, or a single read of any sort
        assert!(f().check([].map(ok), [t].map(ok)).is_ok());
        assert!(f().check([t].map(ok), [].map(ok)).is_ok());
        assert!(f().check([t.T()].map(ok), [].map(ok)).is_ok());
        assert!(f().check([t].map(ok), [t].map(ok)).is_err());
        assert!(f().check([t, t].map(ok), [].map(ok)).is_err());
        assert!(f().check([].map(ok), [].map(ok)).is_err());
    }

    #[test]
    fn linear_derivatives() {
        // `2 * d(t)` folds to a Linear over derivatives
        let lin = || LRA::Linear(real_tensor(2, 3), real_tensor(2, 1));
        let (x, y) = (real(3, 1), real(2, 1));
        assert!(lin().check([x.T()].map(ok), [y.T()].map(ok)).is_ok());
        assert!(lin().check([x.T()].map(ok), [y].map(ok)).is_err());
        assert!(lin().check([x].map(ok), [y.T()].map(ok)).is_err());
        assert!(
            lin()
                .check([x.T().T()].map(ok), [y.T().T()].map(ok))
                .is_ok()
        );
    }
}
