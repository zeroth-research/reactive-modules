/*!
# Linear integer arithmetic

Defines the theory [`LIA`] of linear integer arithmetic over matrices,
mixing integer and boolean matrices in a single signature.

A [`Sort`] value is either `Int(rows, cols)` or `Bool(rows, cols)`,
or `Zero(rows, cols)`, the sort of their (zero) derivatives.
`Type` converts to and from [`int::IntType`] and [`bool::PropType`]
so that integer and propositional terms embed directly into `LIA`. The
operations in [`LIA`] are:

- [`LIA::Int`], [`LIA::Bool`] — integer and boolean matrix literals.
- [`LIA::And`], [`LIA::Or`], [`LIA::Xor`], [`LIA::Not`]
  — boolean operations on the boolean fragment of `Type`.
- [`LIA::Le`], [`LIA::Lt`], [`LIA::Ge`], [`LIA::Gt`], [`LIA::Eq`], [`LIA::Ne`]
  — pointwise integer comparisons producing a boolean matrix.
- [`LIA::Ite`] — if-then-else: reads a boolean guard and two same-typed branches.
- [`LIA::Linear`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and
  `B` constant integer matrices of compatible shapes.
- [`LIA::ReLU`] — the shape-preserving rectified-linear map on integer matrices.

`LIA` implements [`Signature`] over [`Sort`], with matrices of scalars
`Int(1, 1)` and `Bool(1, 1)`, e.g., the signature of [`LIA::ReLU`]
is `Mat(Int(1, 1), a, b) -> Mat(Int(1, 1), a, b)`, and of [`LIA::Transpose`]
(of any sort `'t`) is `Mat('t, a, b) -> Mat('t, b, a)`.

`LIA` implements [`Theory`]; [`Theory::check`] typechecks the sorts of the
read and write wires against the signature of the operation.

## Examples

```
use theory::Theory;
use theory::lia::{LIA, Sort};

// Wires carry a sort.
let ok = |s| Ok::<_, String>(s);

// Pointwise less-than on scalars: Int(1,1), Int(1,1) -> Bool(1,1).
let i = Sort::Int([1, 1]);
let b = Sort::Bool([1, 1]);
assert!(LIA::Lt().check([i, i].map(ok), [b].map(ok)).is_ok());

// ReLU preserves shape and stays in the integer fragment.
let m = Sort::Int([3, 4]);
assert!(LIA::ReLU().check([m].map(ok), [m].map(ok)).is_ok());
assert!(LIA::ReLU().check([b].map(ok), [b].map(ok)).is_err());
```
*/

use crate::signature::{Mat, Signature, SignatureElem, TypeVar, VecSignature, sig};
use crate::sort::{MultiSort, TangentMultiSort};
use crate::*;
#[cfg(feature = "pyo3")]
use pyo3::pyclass;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Sort {
    Int([usize; 2]),
    Bool([usize; 2]),
    // the tangent of `Int` and `Bool` (they have only the zero derivative)
    Zero([usize; 2]),
}

impl Sort {
    pub fn is_bool(&self) -> bool {
        matches!(self, Sort::Bool(..))
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Sort::Int(..))
    }

    pub fn shape(&self) -> &[usize; 2] {
        match self {
            Sort::Bool(shape) | Sort::Int(shape) | Sort::Zero(shape) => shape,
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Int([i, j]) => write!(f, "Int({i}, {j})"),
            Sort::Bool([i, j]) => write!(f, "Bool({i}, {j})"),
            Sort::Zero([i, j]) => write!(f, "Zero({i}, {j})"),
        }
    }
}

impl MultiSort for Sort {
    const NAME: &'static str = "LIASort";
}

impl TangentMultiSort for Sort {
    fn T(&self) -> Self {
        Sort::Zero(*self.shape())
    }
}

#[derive(Clone, Debug, strum::Display)]
#[cfg_attr(feature = "pyo3", pyclass(frozen))]
pub enum LIA {
    // integer and boolean matrix literals
    #[strum(to_string = "{0}")]
    Int(crate::PyTensor),
    #[strum(to_string = "{0}")]
    Bool(crate::PyTensor),
    // boolean operations
    And(),
    Or(),
    Xor(),
    Not(),
    // integer comparisons
    Le(),
    Lt(),
    Ge(),
    Gt(),
    Eq(),
    Ne(),
    // linear / matrix operations
    // A*x + B where `A` and `B` are constants
    Linear(crate::PyTensor, crate::PyTensor),
    Add(),
    Sub(),
    // XXX: should these be in LIA?
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
    AnyInt([usize; 2]),
    AnyBool([usize; 2]),
}

impl Sequential for LIA {
    fn skip(_range: &Self::Sort) -> Self {
        LIA::Id()
    }
}

impl Combinatorial for LIA {
    fn havoc(range: &Sort) -> Self {
        // a derivative cannot be havoced: the check of the term with `Zero` fails
        match range {
            Sort::Int(shape) | Sort::Zero(shape) => LIA::AnyInt(*shape),
            Sort::Bool(shape) => LIA::AnyBool(*shape),
        }
    }
}

impl Theory for LIA {
    type Sort = Sort;
    const NAME: &'static str = "LIA";

    fn check<R, W, E: fmt::Display>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<Sort, E>>,
        W: IntoIterator<Item = Result<Sort, E>>,
    {
        let check = || -> Result<(), String> {
            let (read, write) = (wire_sorts(read)?, wire_sorts(write)?);
            let mut sig = signature(self)?;
            // `Uninterpreted` either writes or reads a single value, but its signature
            // has the write
            if matches!(self, LIA::Uninterpreted(_)) && !read.is_empty() {
                std::mem::swap(&mut sig.dom, &mut sig.cod);
            }
            typecheck_wires(sig, &read, &write, sort_elem)
        };
        check().map_err(|e| format!("{self:?}: {e}"))
    }
}

type Elem = SignatureElem<Sort>;

/// Matrices in the signatures are matrices of scalars: `Int([i, j])`
/// is `Mat(Int([1, 1]), i, j)` and `Bool([i, j])` is `Mat(Bool([1, 1]), i, j)`.
///
/// # Panics
///
/// If the operation is ill-formed: its tensors are not 2D, or `B` of [`LIA::Linear`]
/// is not a column vector ([`Theory::check`] reports these as errors).
impl Signature<Sort> for LIA {
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

fn expect_signature(op: &LIA) -> VecSignature<Sort> {
    signature(op).unwrap_or_else(|e| panic!("{op:?}: {e}"))
}

/// The signature of `op`, or an error if `op` is ill-formed
fn signature(op: &LIA) -> Result<VecSignature<Sort>, String> {
    // the sorts of the elements of matrices
    let (int, bool) = (Sort::Int([1, 1]), Sort::Bool([1, 1]));
    // a matrix with symbolic dimensions, the same for all (pointwise) arguments
    let m = |s: Sort| Mat(s, 'a', 'b');
    Ok(match op {
        LIA::Int(cm) => {
            let [rows, cols] = shape_2d(cm)?;
            sig!(() -> Mat(int, rows, cols))
        }
        LIA::Bool(cm) => {
            let [rows, cols] = shape_2d(cm)?;
            sig!(() -> Mat(bool, rows, cols))
        }
        LIA::And() | LIA::Or() | LIA::Xor() => sig!((m(bool), m(bool)) -> m(bool)),
        LIA::Not() => sig!(m(bool) -> m(bool)),
        LIA::Le() | LIA::Lt() | LIA::Ge() | LIA::Gt() | LIA::Eq() | LIA::Ne() => {
            sig!((m(int), m(int)) -> m(bool))
        }
        LIA::Linear(a, b) => {
            // Y = A·X + B  where X=[in,batch], A=[out,in], B=[out,1] (or empty), Y=[out,batch]
            let [out, inp] = shape_2d(a)?;
            if b.numel() != 0 && shape_2d(b)? != [out, 1] {
                return Err(format!("`B` must be a column vector [{out}x1]"));
            }
            sig!(Mat(int, inp, 'b') -> Mat(int, out, 'b'))
        }
        // `Min` and `Max` are pointwise, as in the evaluator
        LIA::Add() | LIA::Sub() | LIA::Min() | LIA::Max() => sig!((m(int), m(int)) -> m(int)),
        LIA::ReLU() => sig!(m(int) -> m(int)),
        // the index into the flattened matrix, as in the evaluator and the Python builder
        LIA::Argmax() => sig!(m(int) -> Mat(int, 1, 1)),
        // of any sort, as in the evaluator
        LIA::Transpose() => sig!(Mat('t', 'a', 'b') -> Mat('t', 'b', 'a')),
        LIA::Ite() => sig!((Mat(bool, 1, 1), TypeVar('a'), TypeVar('a')) -> TypeVar('a')),
        LIA::Id() => sig!(TypeVar('a') -> TypeVar('a')),
        // `Uninterpreted` either writes or reads a single value, but a signature
        // has a fixed arity: we pick the write (used by the builder for untraced calls)
        LIA::Uninterpreted(_) => sig!(() -> TypeVar('a')),
        LIA::AnyInt(shape) => sig!(() -> sort_elem(Sort::Int(*shape))),
        LIA::AnyBool(shape) => sig!(() -> sort_elem(Sort::Bool(*shape))),
    })
}

/// The signature element of `sort`: a matrix of scalars
fn sort_elem(sort: Sort) -> Elem {
    match sort {
        Sort::Int([i, j]) => Mat(Sort::Int([1, 1]), i, j),
        Sort::Bool([i, j]) => Mat(Sort::Bool([1, 1]), i, j),
        Sort::Zero([i, j]) => Mat(Sort::Zero([1, 1]), i, j),
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    fn int(r: usize, c: usize) -> Sort {
        Sort::Int([r, c])
    }

    fn bool_t(r: usize, c: usize) -> Sort {
        Sort::Bool([r, c])
    }

    fn ok(s: Sort) -> Result<Sort, String> {
        Ok(s)
    }

    #[test]
    fn type_kind_and_shape() {
        assert!(int(2, 3).is_int() && !int(2, 3).is_bool());
        assert_eq!(int(2, 3).shape(), &[2, 3]);
        assert!(bool_t(1, 1).is_bool() && !bool_t(1, 1).is_int());
    }

    #[test]
    fn const_int_ok() {
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0i64, 1], [2, 3]]).into();
        assert!(LIA::Int(cm).check([].map(ok), [int(2, 2)].map(ok)).is_ok());
    }

    #[test]
    fn const_int_bool_write_fails() {
        assert!(
            LIA::Int(tch::Tensor::from_slice2(&[[0i64]]).into())
                .check([].map(ok), [bool_t(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_int_wrong_rows_fails() {
        assert!(
            LIA::Int(tch::Tensor::from_slice2(&[[0i64]]).into())
                .check([].map(ok), [int(2, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_int_with_read_fails() {
        let t = int(1, 1);
        assert!(
            LIA::Int(tch::Tensor::from_slice2(&[[0i64]]).into())
                .check([t].map(ok), [t].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_bool_ok() {
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[true, false], [false, true]]).into();
        assert!(
            LIA::Bool(cm)
                .check([].map(ok), [bool_t(2, 2)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn const_bool_int_write_fails() {
        assert!(
            LIA::Bool(tch::Tensor::from_slice2(&[[true]]).into())
                .check([].map(ok), [int(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn not_ok() {
        let b = bool_t(2, 3);
        assert!(LIA::Not().check([b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn not_int_input_fails() {
        let t = int(1, 1);
        assert!(LIA::Not().check([t].map(ok), [t].map(ok)).is_err());
    }

    #[test]
    fn and_ok() {
        let b = bool_t(2, 2);
        assert!(LIA::And().check([b, b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn or_ok() {
        let b = bool_t(1, 1);
        assert!(LIA::Or().check([b, b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn xor_ok() {
        let b = bool_t(3, 1);
        assert!(LIA::Xor().check([b, b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn and_int_output_fails() {
        let b = bool_t(1, 1);
        assert!(
            LIA::And()
                .check([b, b].map(ok), [int(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn and_type_mismatch_fails() {
        assert!(
            LIA::And()
                .check([bool_t(1, 1), bool_t(1, 2)].map(ok), [bool_t(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn lt_ok() {
        assert!(
            LIA::Lt()
                .check([int(1, 1), int(1, 1)].map(ok), [bool_t(1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn le_ok() {
        let res = LIA::Le().check([int(2, 3), int(2, 3)].map(ok), [bool_t(2, 3)].map(ok));
        assert!(res.is_ok(), "result: {:?}", res);
    }

    #[test]
    fn eq_ok() {
        assert!(
            LIA::Eq()
                .check([int(3, 2), int(3, 2)].map(ok), [bool_t(3, 2)].map(ok))
                .is_ok()
        );
        assert!(
            LIA::Eq()
                .check([int(3, 3), int(3, 2)].map(ok), [bool_t(3, 2)].map(ok))
                .is_err()
        );
        assert!(
            LIA::Eq()
                .check([int(3, 2), int(3, 2)].map(ok), [bool_t(3, 3)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn cmp_non_bool_output_fails() {
        let t = int(1, 1);
        assert!(LIA::Lt().check([t, t].map(ok), [t].map(ok)).is_err());
    }

    #[test]
    fn cmp_input_mismatch_fails() {
        assert!(
            LIA::Eq()
                .check([int(1, 1), int(1, 2)].map(ok), [bool_t(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn add_ok() {
        let t = int(3, 4);
        assert!(LIA::Add().check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn add_shape_mismatch_fails() {
        assert!(
            LIA::Add()
                .check([int(1, 2), int(2, 1)].map(ok), [int(1, 2)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn add_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LIA::Add().check([b, b].map(ok), [b].map(ok)).is_err());
    }

    #[test]
    fn relu_ok() {
        let t = int(3, 4);
        assert!(LIA::ReLU().check([t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn relu_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LIA::ReLU().check([b].map(ok), [b].map(ok)).is_err());
    }

    #[test]
    fn argmax_ok() {
        assert!(
            LIA::Argmax()
                .check([int(3, 4)].map(ok), [int(1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn argmax_matrix_output_fails() {
        assert!(
            LIA::Argmax()
                .check([int(3, 4)].map(ok), [int(3, 4)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn min_ok() {
        // pointwise
        let t = int(4, 1);
        assert!(LIA::Min().check([t, t].map(ok), [t].map(ok)).is_ok());
        assert!(LIA::Max().check([t].map(ok), [int(1, 1)].map(ok)).is_err());
    }

    #[test]
    fn linear_ok() {
        // A=[2,3] maps 3 features to 2; X=[3,4] is 3 features × 4 batch items.
        // Convention: Y = A·X  →  Y=[2,4].
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Int64, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([0, 0], (tch::Kind::Int64, tch::Device::Cpu)).into();
        assert!(
            LIA::Linear(a, b)
                .check([int(3, 4)].map(ok), [int(2, 4)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn linear_with_bias_ok() {
        // A=[2,3], b=[2,1] column bias, X=[3,1] single sample → Y=[2,1].
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Int64, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([2, 1], (tch::Kind::Int64, tch::Device::Cpu)).into();
        assert!(
            LIA::Linear(a, b)
                .check([int(3, 1)].map(ok), [int(2, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn linear_dim_mismatch_fails() {
        // A=[2,3] but X has 4 rows — inner dimension mismatch.
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Int64, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([0, 0], (tch::Kind::Int64, tch::Device::Cpu)).into();
        assert!(
            LIA::Linear(a, b)
                .check([int(4, 1)].map(ok), [int(2, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn transpose_ok() {
        assert!(
            LIA::Transpose()
                .check([int(3, 4)].map(ok), [int(4, 3)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn transpose_wrong_shape_fails() {
        assert!(
            LIA::Transpose()
                .check([int(3, 4)].map(ok), [int(3, 4)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn ite_ok() {
        let t = int(3, 4);
        assert!(
            LIA::Ite()
                .check([bool_t(1, 1), t, t].map(ok), [t].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn ite_non_bool_guard_fails() {
        let t = int(1, 1);
        assert!(LIA::Ite().check([t, t, t].map(ok), [t].map(ok)).is_err());
    }

    #[test]
    fn ite_arm_mismatch_fails() {
        assert!(
            LIA::Ite()
                .check(
                    [bool_t(1, 1), int(1, 1), int(1, 2)].map(ok),
                    [int(1, 1)].map(ok)
                )
                .is_err()
        );
    }

    #[test]
    fn id_ok() {
        let t = int(4, 4);
        assert!(LIA::Id().check([t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn id_type_mismatch_fails() {
        assert!(
            LIA::Id()
                .check([int(1, 1)].map(ok), [int(2, 2)].map(ok))
                .is_err()
        );
    }

    // Formats the signature of `op` as `(dom) -> (cod)`
    fn sig(op: LIA) -> String {
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
        let bin = |s| format!("(Mat({s}, a, b), Mat({s}, a, b)) -> (Mat({s}, a, b))");
        for op in [LIA::And(), LIA::Or(), LIA::Xor()] {
            assert_eq!(sig(op), bin("Bool(1, 1)"));
        }
        for op in [LIA::Add(), LIA::Sub(), LIA::Min(), LIA::Max()] {
            assert_eq!(sig(op), bin("Int(1, 1)"));
        }
        for op in [
            LIA::Le(),
            LIA::Lt(),
            LIA::Ge(),
            LIA::Gt(),
            LIA::Eq(),
            LIA::Ne(),
        ] {
            assert_eq!(
                sig(op),
                "(Mat(Int(1, 1), a, b), Mat(Int(1, 1), a, b)) -> (Mat(Bool(1, 1), a, b))"
            );
        }
        assert_eq!(
            sig(LIA::Not()),
            "(Mat(Bool(1, 1), a, b)) -> (Mat(Bool(1, 1), a, b))"
        );
        assert_eq!(
            sig(LIA::ReLU()),
            "(Mat(Int(1, 1), a, b)) -> (Mat(Int(1, 1), a, b))"
        );
        assert_eq!(
            sig(LIA::Argmax()),
            "(Mat(Int(1, 1), a, b)) -> (Mat(Int(1, 1), 1, 1))"
        );
        assert_eq!(sig(LIA::Transpose()), "(Mat('t, a, b)) -> (Mat('t, b, a))");
        assert_eq!(sig(LIA::Ite()), "(Mat(Bool(1, 1), 1, 1), 'a, 'a) -> ('a)");
        assert_eq!(sig(LIA::Id()), "('a) -> ('a)");
        assert_eq!(sig(LIA::Uninterpreted("f".into())), "() -> ('a)");
        assert_eq!(sig(LIA::AnyInt([2, 3])), "() -> (Mat(Int(1, 1), 2, 3))");
        assert_eq!(sig(LIA::AnyBool([2, 3])), "() -> (Mat(Bool(1, 1), 2, 3))");
    }

    #[test]
    fn signature_const() {
        assert_eq!(
            sig(LIA::Int(int_tensor(2, 3))),
            "() -> (Mat(Int(1, 1), 2, 3))"
        );
        let b: crate::PyTensor = tch::Tensor::from_slice2(&[[true, false]]).into();
        assert_eq!(sig(LIA::Bool(b.clone())), "() -> (Mat(Bool(1, 1), 1, 2))");
        // the sort is given by the variant, not by the tensor
        assert_eq!(sig(LIA::Int(b)), "() -> (Mat(Int(1, 1), 1, 2))");
        assert_eq!(
            sig(LIA::Bool(int_tensor(2, 3))),
            "() -> (Mat(Bool(1, 1), 2, 3))"
        );
    }

    #[test]
    fn signature_linear() {
        // A=[2,3] maps 3 features to 2
        assert_eq!(
            sig(LIA::Linear(int_tensor(2, 3), int_tensor(2, 1))),
            "(Mat(Int(1, 1), 3, b)) -> (Mat(Int(1, 1), 2, b))"
        );
        assert_eq!(
            sig(LIA::Linear(int_tensor(2, 3), int_tensor(0, 0))),
            "(Mat(Int(1, 1), 3, b)) -> (Mat(Int(1, 1), 2, b))"
        );
    }

    #[test]
    #[should_panic(expected = "expected a 2D tensor, got 1D")]
    fn signature_const_not_2d() {
        let t: crate::PyTensor = tch::Tensor::from_slice(&[0i64, 1]).into();
        let _ = sig(LIA::Int(t));
    }

    #[test]
    fn signature_arity() {
        assert_eq!((LIA::Ite().dom_arity(), LIA::Ite().cod_arity()), (3, 1));
        assert_eq!((LIA::Not().dom_arity(), LIA::Not().cod_arity()), (1, 1));
        let any = LIA::AnyInt([1, 1]);
        assert_eq!((any.dom_arity(), any.cod_arity()), (0, 1));
    }

    #[test]
    fn signature_typecheck() {
        use crate::signature::typecheck;
        assert_eq!(typecheck(LIA::Add(), LIA::Max()), Ok(()));
        // `'a -> 'a` is instantiated by `Mat(Int(1, 1), a, b) -> Mat(Int(1, 1), a, b)`
        assert_eq!(typecheck(LIA::Id(), LIA::ReLU()), Ok(()));
        assert_eq!(
            typecheck(LIA::AnyInt([2, 3]), LIA::Int(int_tensor(2, 3))),
            Ok(())
        );
        assert_eq!(
            typecheck(LIA::AnyInt([2, 3]), LIA::AnyBool([2, 3])),
            Err(
                "Failed unifying `() -> Mat(Int(1, 1), 2, 3)` with `() -> Mat(Bool(1, 1), 2, 3)` \
                 on `Mat(Int(1, 1), 2, 3)` and `Mat(Bool(1, 1), 2, 3)`"
                    .to_string()
            )
        );
        assert_eq!(
            typecheck(LIA::Lt(), LIA::Add()),
            Err(
                "Failed unifying `(Mat(Int(1, 1), a, b), Mat(Int(1, 1), a, b)) -> Mat(Bool(1, 1), a, b)` \
                 with `(Mat(Int(1, 1), a, b), Mat(Int(1, 1), a, b)) -> Mat(Int(1, 1), a, b)` \
                 on `Mat(Bool(1, 1), a, b)` and `Mat(Int(1, 1), a, b)`"
                    .to_string()
            )
        );
        // Transpose is of any sort (of square matrices to unify with ReLU and Not)
        assert_eq!(typecheck(LIA::Transpose(), LIA::ReLU()), Ok(()));
        assert_eq!(typecheck(LIA::Transpose(), LIA::Not()), Ok(()));
        // Linear maps `Mat(Int(1, 1), 3, b)` to `Mat(Int(1, 1), 2, b)`, but Transpose to `Mat('t, b, 3)`
        let linear = || LIA::Linear(int_tensor(2, 3), int_tensor(0, 0));
        assert_eq!(typecheck(linear(), linear()), Ok(()));
        assert!(typecheck(linear(), LIA::Transpose()).is_err());
        assert!(typecheck(linear(), LIA::Linear(int_tensor(3, 3), int_tensor(0, 0))).is_err());
    }
    #[test]
    fn check_errors() {
        assert_eq!(
            LIA::Lt().check([int(1, 1), int(1, 2)].map(ok), [bool_t(1, 1)].map(ok)),
            Err(
                "Lt: Failed unifying `(Mat(Int(1, 1), a, b), Mat(Int(1, 1), a, b)) -> Mat(Bool(1, 1), a, b)` \
                 with `(Mat(Int(1, 1), 1, 1), Mat(Int(1, 1), 1, 2)) -> Mat(Bool(1, 1), 1, 1)` \
                 on `Mat(Int(1, 1), 1, 1)` and `Mat(Int(1, 1), 1, 2)`"
                    .to_string()
            )
        );
        let b = bool_t(1, 1);
        assert_eq!(
            LIA::Not().check([b, b].map(ok), [b].map(ok)),
            Err("Not: Signatures domains have different arity (1 != 2)".to_string())
        );
        // errors of wires are propagated
        assert_eq!(
            LIA::Not().check([Err("bad wire")], [b].map(Ok)),
            Err("Not: bad wire".to_string())
        );
    }

    #[test]
    fn check_ill_formed_ops() {
        // ill-formed tensors are reported, not panicking
        let t: crate::PyTensor = tch::Tensor::from_slice(&[0i64, 1]).into();
        let res = LIA::Int(t).check([].map(ok), [int(1, 2)].map(ok));
        assert!(res.unwrap_err().ends_with("expected a 2D tensor, got 1D"));
        let res = LIA::Linear(int_tensor(2, 3), int_tensor(2, 2))
            .check([int(3, 1)].map(ok), [int(2, 1)].map(ok));
        assert!(
            res.unwrap_err()
                .ends_with("`B` must be a column vector [2x1]")
        );
    }

    #[test]
    fn check_uninterpreted() {
        let f = || LIA::Uninterpreted("f".into());
        let t = int(2, 3);
        // either a single write, or a single read of any sort
        assert!(f().check([].map(ok), [t].map(ok)).is_ok());
        assert!(f().check([t].map(ok), [].map(ok)).is_ok());
        assert!(f().check([bool_t(1, 1)].map(ok), [].map(ok)).is_ok());
        assert!(f().check([t].map(ok), [t].map(ok)).is_err());
        assert!(f().check([t, t].map(ok), [].map(ok)).is_err());
        assert!(f().check([].map(ok), [t, t].map(ok)).is_err());
        assert!(f().check([].map(ok), [].map(ok)).is_err());
    }

    #[test]
    fn check_derivatives() {
        // the derivatives of integers and booleans are zero
        let z = int(2, 3).T();
        assert_eq!(z, Sort::Zero([2, 3]));
        assert_eq!(bool_t(2, 3).T(), z);
        assert_eq!(z.T(), z);
        // operations on values do not apply to derivatives
        assert!(LIA::Add().check([z, z].map(ok), [z].map(ok)).is_err());
        assert!(LIA::Not().check([z].map(ok), [z].map(ok)).is_err());
        assert!(LIA::AnyInt([2, 3]).check([].map(ok), [z].map(ok)).is_err());
        assert!(LIA::Id().check([int(2, 3)].map(ok), [z].map(ok)).is_err());
    }
}
