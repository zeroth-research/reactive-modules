pub mod bool;
pub mod bv;
pub mod int;
pub mod mat;
pub mod nat;
pub mod real;

trait Theory {
    type DType;

    fn _check(&self, read: &[Self::DType], write: &[Self::DType]) -> bool;

    fn check<T>(&self, read: &[T], write: &[T]) -> bool
    where
        T: TryInto<Self::DType>,
    {
        if let (r, w) = (read.try_into()?, write.try_into()?) {
            return self._check(r, w);
        }

        false
    }
}
