// use std::collections::binary_heap::Iter;
//
// use theory::*;
//
// struct Term<T: Theory> {
//     itype: T::IT,
//     read: Vec<T::DT>,
//     write: Vec<T::DT>,
// }
//
// impl<T: Theory> Term<T> {
//     fn check(&self) -> bool {
//         type_check(self.itype, self.read, self.write)
//     }
// }
//
// macro_rules! mk_term {
//     ($op:expr, ($r1:expr, $r2:expr), $w:expr) => {
//         Term::mk_bin_op($op, ($r1, $r2), $w)
//     };
//     ($op:expr, $r:expr, $w:expr) => {
//         Term::mk_unary_op($op, $r, $w)
//     };
// }
//
// #[test]
// fn test_typecheck1() {
//     use theory::nat::*;
//     let _: Term<Theory> = mk_term!(Add(), (Nat(), Nat()), Nat());
//     let t: Term<Theory> = mk_term!(Id(), Nat(), Nat());
//     t.check();
// }
//
// #[test]
// fn test_typecheck2() {
//     {
//         use theory::nat::Theory as NatTheory;
//         use theory::nat::*;
//         let _: Term<NatTheory> = mk_term!(Add(), (Nat(), Nat()), Nat());
//     }
//
//     {
//         let _: Term<int::Theory> = mk_term!(int::Id(), int::Int(), int::Int());
//     }
// }
