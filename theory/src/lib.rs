pub mod bool;
pub mod bv;
pub mod int;
pub mod lia;
pub mod real;

pub trait Theory {
    type DType;

    fn type_check<'a, R, W>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = &'a Self::DType>,
        W: IntoIterator<Item = &'a Self::DType>,
        Self::DType: 'a;
}

// // Operations common to every theory
// #[derive(Clone, Copy, PartialEq, Debug)]
// pub enum FlowOp<T: PartialEq + std::fmt::Debug> {
//     Id,
//     _Phantom(std::marker::PhantomData<T>),
// }
//
// impl<T> Theory for FlowOp<T>
// where
//     T: PartialEq + std::fmt::Debug,
// {
//     type DType = T;
//
//     fn _check(&self, read: &[T], write: &[T]) -> Result<(), String> {
//         match self {
//             FlowOp::Id => {
//                 if read.len() != 1 {
//                     return Err(format!(
//                         "{:?}: must read one value, got {}",
//                         self,
//                         read.len()
//                     ));
//                 }
//                 if write.len() != 1 {
//                     return Err(format!(
//                         "{:?}: must write a single value, got {}",
//                         self,
//                         write.len()
//                     ));
//                 }
//                 if read[0] != write[0] {
//                     return Err(format!(
//                         "{:?}: input and output must have the same type",
//                         self
//                     ));
//                 }
//                 Ok(())
//             }
//             FlowOp::_Phantom(_) => unimplemented!(),
//         }
//     }
// }
