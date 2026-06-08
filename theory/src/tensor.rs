use std::fmt;

/// Wraps a tensor value. With the `torch` feature this is a [`tch::Tensor`]; without it
/// this is a placeholder type for future alternative backends.
#[cfg(feature = "torch")]
#[derive(Debug)]
pub struct Tensor(pub tch::Tensor);

#[cfg(not(feature = "torch"))]
#[derive(Debug, Clone)]
pub struct Tensor;

// --- torch implementation ---

#[cfg(feature = "torch")]
impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor(self.0.shallow_clone())
    }
}

#[cfg(feature = "torch")]
impl std::ops::Deref for Tensor {
    type Target = tch::Tensor;
    fn deref(&self) -> &tch::Tensor {
        &self.0
    }
}

#[cfg(feature = "torch")]
impl From<tch::Tensor> for Tensor {
    fn from(t: tch::Tensor) -> Self {
        Tensor(t)
    }
}

#[cfg(feature = "torch")]
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(not(feature = "torch"))]
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "?Tensor")
    }
}

// tch::Tensor doesn't auto-implement Sync; assert it's safe for concurrent reads.
#[cfg(feature = "torch")]
unsafe impl Sync for Tensor {}

#[cfg(feature = "torch")]
impl Tensor {
    pub fn size(&self) -> Vec<i64> {
        return self.0.size();
    }

    pub fn numel(&self) -> usize {
        return self.0.numel();
    }

    pub fn min(&self) -> Tensor {
        return Tensor(self.0.min());
    }

    pub fn max(&self) -> Tensor {
        return Tensor(self.0.max());
    }

    pub fn int64_value(&self, _idx: &[i64]) -> i64 {
        return self.0.int64_value(_idx);
    }
}

#[cfg(not(feature = "torch"))]
impl Tensor {
    pub fn size(&self) -> Vec<i64> {
        unimplemented!()
    }

    pub fn numel(&self) -> usize {
        unimplemented!()
    }

    pub fn min(&self) -> Tensor {
        unimplemented!()
    }

    pub fn max(&self) -> Tensor {
        unimplemented!()
    }

    pub fn int64_value(&self, _idx: &[i64]) -> i64 {
        unimplemented!()
    }
}
