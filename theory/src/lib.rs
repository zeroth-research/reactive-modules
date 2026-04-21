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
