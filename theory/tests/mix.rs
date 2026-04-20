// use theory::mat::{Mat, MatMul};
// use theory::{Operation, Operation1To1, Operation2To1, TheoryType, Type, mk_theory};
//
// struct Wire<T: theory::Theory> {
//     _id: usize,
//     dtype: T::DT,
// }
//
// struct Term<T: theory::Theory> {
//     _itype: T::IT,
//     _read: Vec<T::DT>,
//     _write: Vec<T::DT>,
// }
//
// impl<T: theory::Theory> Term<T> {
//     fn mk_bin_op<O: Operation, T1: TheoryType<T>, T2: TheoryType<T>, R: TheoryType<T>>(
//         op: O,
//         read: (T1, T2),
//         write: R,
//     ) -> Term<T>
//     where
//         O: Into<T::IT>,
//         O: Operation2To1<T1, T2, R>,
//     {
//         Term {
//             _itype: op.into(),
//             _read: vec![read.0.into(), read.1.into()],
//             _write: vec![write.into()],
//         }
//     }
//
//     fn mk_unary_op<O: Operation, T1: TheoryType<T>, R: TheoryType<T>>(
//         op: O,
//         read: T1,
//         write: R,
//     ) -> Term<T>
//     where
//         O: Into<T::IT>,
//         O: Operation1To1<T1, R>,
//     {
//         Term {
//             _itype: op.into(),
//             _read: vec![read.into()],
//             _write: vec![write.into()],
//         }
//     }
// }
//
// use theory::bool as bools;
// use theory::nat;
//
// // Make theory of Bool, Nat and Matrices over Nat and Bool
// mk_theory!(
//     Types(
//         [] Bool => bools::Bool,
//         [] Nat => nat::Nat,
//         [T: Type, const M: usize, const N: usize] Mat => Mat<T, M, N>
//     ),
//     {
//         [T: Type, const M: usize, const N: usize]
//         Add(Mat<T, M, N>, Mat<T, M, N>) => Mat<T, M, N>,
//         Id(Mat<T, M, N>) => Mat<T, M, N>
//     }
//     {
//         [T: Type + Copy, const A: usize, const B: usize, const C: usize]
//         MatMul => MatMul<T, A, B, C>
//     }
// );
//
// fn mat<T: Type, const M: usize, const N: usize>() -> Mat<T, M, N> {
//     Mat(std::marker::PhantomData)
// }
//
// fn matmul<T: Type + Copy, const A: usize, const B: usize, const C: usize>() -> MatMul<T, A, B, C> {
//     MatMul(std::marker::PhantomData)
// }
//
// #[test]
// fn test_mat_add() {
//     let _: Term<Theory> = Term::mk_bin_op(
//         Add(),
//         (mat::<bools::Bool, 2, 3>(), mat::<bools::Bool, 2, 3>()),
//         mat::<bools::Bool, 2, 3>(),
//     );
//
//     let _: Term<Theory> = Term::mk_bin_op(
//         matmul(),
//         (mat::<bools::Bool, 2, 3>(), mat::<bools::Bool, 3, 3>()),
//         mat::<bools::Bool, 2, 3>(),
//     );
// }
