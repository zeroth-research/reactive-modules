use std::fmt;
// use theory::mk_theory;
// use theory::{bool, int, mat, real};
//
// // These operations are forwarded, so make their use public,
// // so that they appear as being in this theory
// pub use mat::{Add, Id, MatConst};
//
// mk_theory!({DTypes, ITypes, Thry}
//     Types(
//         [const M: usize, const N: usize]
//         Bool => mat::Mat<bool::Bool, M, N>,
//         [const M: usize, const N: usize]
//         Int => mat::Mat<int::Int, M, N>,
//         [const M: usize, const N: usize]
//         Real => mat::Mat<real::Real, M, N>
//         // TODO: missing: Float, Word
//     ),
//     // these operations are defined anew (they must be defined a new even for, e.g., boolean
//     // operations because we lift them to matrix operations)
//     {
//         [T: Type + Into<DTypes>, const M: usize, const N: usize]
//         Ite(mat::Mat<bool::Bool, 1, 1>, mat::Mat<T, M, N>, mat::Mat<T, M, N>) => mat::Mat<T, M, N>,
//     }
//     {
//         // Matrix special operations (TODO: should be defined in `mat::`?)
//         [T: Type + Into<DTypes>, const M: usize, const N: usize]
//         Sum(mat::Mat<T, M, N>) => mat::Mat<T, 1, 1>,
//         Mean(mat::Mat<T, M, N>) => mat::Mat<real::Real, 1, 1>,
//         Max(mat::Mat<T, M, N>) => mat::Mat<real::Real, 1, 1>,
//     }
//     {
//         [const M: usize, const N: usize]
//         // logical operations
//         And(mat::Mat<bool::Bool, M, N>, mat::Mat<bool::Bool, M, N>) => mat::Mat<bool::Bool, M, N>,
//         Or(mat::Mat<bool::Bool, M, N>, mat::Mat<bool::Bool, M, N>) => mat::Mat<bool::Bool, M, N>,
//         Xor(mat::Mat<bool::Bool, M, N>, mat::Mat<bool::Bool, M, N>) => mat::Mat<bool::Bool, M, N>,
//         Not(mat::Mat<bool::Bool, M, N>) => mat::Mat<bool::Bool, M, N>,
//     }
//     // map to existing `Operations`
//     {
//         [T: Type + Into<DTypes>, const A: usize, const B: usize, const C: usize]
//         MatMul => mat::MatMul<T, A, B, C>,
//     }
//     {
//         [C: Copy + Clone + PartialEq, const A: usize, const B: usize]
//         // Constant matrix represented by an object of type `C`
//         Const => mat::MatConst<C, A, B>,
//     }
//    //{
//    //    [S: Into<String>]
//    //    // Uninterpreted Type
//    //    Uninterpreted(S) => crate::Uninterpreted
//    //}
//     {
//         Add => Add,
//         Id => Id,
//     }
// );

// Data types for wires/terms.

// Every variant carries a shape (`Vec<usize>`). Scalars are
// represented as 0-dimensional, i.e. an empty shape `vec![]`.
// pub enum DType {
//    //Bool(Vec<usize>),
//    //Int(Vec<usize>),
//    //Real(Vec<usize>),
//     Float(Vec<usize>),
//     /// Unsigned bitvector: bit-width + shape.
//     UWord(u32, Vec<usize>),
//     /// Signed bitvector: bit-width + shape.
//     SWord(u32, Vec<usize>),
// }
//
// fn fmt_shape(f: &mut fmt::Formatter<'_>, shape: &[usize]) -> fmt::Result {
//     write!(f, "(")?;
//     for (i, dim) in shape.iter().enumerate() {
//         if i > 0 {
//             write!(f, ", ")?;
//         }
//         write!(f, "{dim}")?;
//     }
//     write!(f, ")")
// }
//
// impl fmt::Display for DTypes {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         match self {
//             DTypes::Bool => {
//                 write!(f, "Bool")
//                 //fmt_shape(f, s)
//             }
//             DTypes::Int => {
//                 write!(f, "Int")
//                 //fmt_shape(f, s)
//             }
//             // DTypes::Float(s) => {
//             //     write!(f, "Float")?;
//             //     fmt_shape(f, s)
//             // }
//             DTypes::Real => {
//                 write!(f, "Real")
//                 //fmt_shape(f, s)
//             } // DTypes::UWord(n, s) => {
//               //     write!(f, "UWord<{n}>")?;
//               //     fmt_shape(f, s)
//               // }
//               // DTypes::SWord(n, s) => {
//               //     write!(f, "SWord<{n}>")?;
//               //     fmt_shape(f, s)
//               // }
//         }
//     }
// }
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use base::{Interface, Term};
//
//     #[test]
//     fn test_terms_1() {
//         let _: base::Term<DTypes, ITypes> =
//             Term::new_unchecked(mat::Add().into(), Interface::empty(), Interface::empty());
//         let _: base::Term<DTypes, ITypes> =
//             Term::new_unchecked(Add().into(), Interface::empty(), Interface::empty());
//
//         let _: base::Term<DTypes, ITypes> =
//             Term::new_unchecked(Id().into(), Interface::empty(), Interface::empty());
//
//         let c: MatConst<&str, 2, 2> = MatConst("I'm string representing matrix constant");
//         let _: base::Term<DTypes, ITypes> =
//             Term::new_unchecked(c.into(), Interface::empty(), Interface::empty());
//
//         let c: MatConst<Vec<Vec<i64>>, 2, 2> = MatConst(vec![vec![1, 2], vec![3, 4]]);
//         let _: base::Term<DTypes, ITypes> =
//             Term::new_unchecked(c.into(), Interface::empty(), Interface::empty());
//     }
// }
