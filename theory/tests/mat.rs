use std::marker::PhantomData;
use theory::mat::{self, Mat, MatMul, Add, Id};
use theory::{Type, Operation, Operation1To1, Operation2To1, TheoryType, TheoryOperation};

struct Term<T: theory::Theory> {
    itype: T::IT,
    read: Vec<T::DT>,
    write: Vec<T::DT>,
}

impl<T: theory::Theory> Term<T> {
    fn mk_bin_op<O: Operation, T1: TheoryType<T>, T2: TheoryType<T>, R: TheoryType<T>>(
        op: O,
        read: (T1, T2),
        write: R,
    ) -> Term<T>
    where
        O: Into<T::IT>,
        O: Operation2To1<T1, T2, R>,
    {
        Term {
            itype: op.into(),
            read: vec![read.0.into(), read.1.into()],
            write: vec![write.into()],
        }
    }

    fn mk_unary_op<O: Operation, T1: TheoryType<T>, R: TheoryType<T>>(
        op: O,
        read: T1,
        write: R,
    ) -> Term<T>
    where
        O: Into<T::IT>,
        O: Operation1To1<T1, R>,
    {
        Term {
            itype: op.into(),
            read: vec![read.into()],
            write: vec![write.into()],
        }
    }
}

fn mat<T: Type, const M: usize, const N: usize>() -> Mat<T, M, N> {
    Mat(PhantomData)
}

fn matmul<T: Type + Copy, const A: usize, const B: usize, const C: usize>(
) -> MatMul<T, A, B, C> {
    MatMul { t: PhantomData }
}

// --- Mat<Int, M, N> with Add and Id ---

#[test]
fn test_mat_int_add() {
    use theory::int::int::Int;
    let _: Term<mat::Theory> = Term::mk_bin_op(
        Add(),
        (mat::<Int, 2, 3>(), mat::<Int, 2, 3>()),
        mat::<Int, 2, 3>(),
    );
}

#[test]
fn test_mat_int_id() {
    use theory::int::int::Int;
    let _: Term<mat::Theory> = Term::mk_unary_op(
        Id(),
        mat::<Int, 4, 4>(),
        mat::<Int, 4, 4>(),
    );
}

// --- Mat<BV<N>, M, K> with Add and Id ---

#[test]
fn test_mat_bv_add() {
    use theory::bv::BV;
    let _: Term<mat::Theory> = Term::mk_bin_op(
        Add(),
        (mat::<BV<8>, 3, 3>(), mat::<BV<8>, 3, 3>()),
        mat::<BV<8>, 3, 3>(),
    );
}

#[test]
fn test_mat_bv_id() {
    use theory::bv::BV;
    let _: Term<mat::Theory> = Term::mk_unary_op(
        Id(),
        mat::<BV<32>, 2, 5>(),
        mat::<BV<32>, 2, 5>(),
    );
}

// --- MatMul requires T: Type + Copy.
// Int and BV don't derive Copy, so we define a Copy element type.

#[derive(Clone, Copy)]
struct Scalar();
impl Type for Scalar {}

#[test]
fn test_matmul() {
    // MatMul<Scalar, 2, 3, 4>: (Mat<Scalar, 2, 3>, Mat<Scalar, 3, 4>) -> Mat<Scalar, 2, 4>
    let _: Term<mat::Theory> = Term::mk_bin_op(
        matmul::<Scalar, 2, 3, 4>(),
        (mat::<Scalar, 2, 3>(), mat::<Scalar, 3, 4>()),
        mat::<Scalar, 2, 4>(),
    );
}

#[test]
fn test_matmul_square() {
    // Square: (3x3) * (3x3) -> (3x3)
    let _: Term<mat::Theory> = Term::mk_bin_op(
        matmul::<Scalar, 3, 3, 3>(),
        (mat::<Scalar, 3, 3>(), mat::<Scalar, 3, 3>()),
        mat::<Scalar, 3, 3>(),
    );
}
