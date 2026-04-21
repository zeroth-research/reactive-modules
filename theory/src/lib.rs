pub mod bool;
pub mod bv;
pub mod int;
pub mod lia;
pub mod real;

pub trait Theory {
    type DType;

    fn _check(&self, read: &[Self::DType], write: &[Self::DType]) -> Result<(), String>;

    fn check<T>(&self, read: &[T], write: &[T]) -> Result<(), String>
    where
        T: TryInto<Self::DType> + Clone,
    {
        // XXX: we have to translate to Vec here, which I'd like to avoid. But using iterators
        // makes `check` harder. It probably makes sense to do that, but let it be the step 2.
        let rd: Result<Vec<Self::DType>, _> = read.iter().cloned().map(TryInto::try_into).collect();
        let wd: Result<Vec<Self::DType>, _> =
            write.iter().cloned().map(TryInto::try_into).collect();
        if let (Ok(r), Ok(w)) = (rd, wd) {
            return self._check(r.as_slice(), w.as_slice());
        }

        Err("Arguments have types from incompatible theory".into())
    }
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
