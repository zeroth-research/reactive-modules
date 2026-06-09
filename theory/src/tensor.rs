// use pyo3::pyclass;
// use std::fmt;
//
// /// Wraps a tensor value. With the `torch` feature this is a [`tch::Tensor`]; without it
// /// this is a placeholder type for future alternative backends.
// #[cfg(feature = "torch")]
// #[derive(Debug)]
// #[pyclass(frozen)]
// pub struct Tensor(pub tch::Tensor);
//
// #[cfg(not(feature = "torch"))]
// #[derive(Debug, Clone)]
// pub struct Tensor;
//
// // --- torch implementation ---
//
// #[cfg(feature = "torch")]
// impl Clone for Tensor {
//     fn clone(&self) -> Self {
//         Tensor(self.0.shallow_clone())
//     }
// }
//
// #[cfg(feature = "torch")]
// impl std::ops::Deref for Tensor {
//     type Target = tch::Tensor;
//     fn deref(&self) -> &tch::Tensor {
//         &self.0
//     }
// }
//
// #[cfg(feature = "torch")]
// impl From<tch::Tensor> for Tensor {
//     fn from(t: tch::Tensor) -> Self {
//         Tensor(t)
//     }
// }
//
// #[cfg(feature = "torch")]
// impl fmt::Display for Tensor {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "{}", self.0)
//     }
// }
//
// #[cfg(not(feature = "torch"))]
// impl fmt::Display for Tensor {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "?Tensor")
//     }
// }
//
// // tch::Tensor doesn't auto-implement Sync; assert it's safe for concurrent reads.
// #[cfg(feature = "torch")]
// unsafe impl Sync for Tensor {}
//
// #[cfg(feature = "torch")]
// impl Tensor {
//     pub fn size(&self) -> Vec<i64> {
//         return self.0.size();
//     }
//
//     pub fn numel(&self) -> usize {
//         return self.0.numel();
//     }
//
//     pub fn min(&self) -> Tensor {
//         return Tensor(self.0.min());
//     }
//
//     pub fn max(&self) -> Tensor {
//         return Tensor(self.0.max());
//     }
//
//     pub fn int64_value(&self, _idx: &[i64]) -> i64 {
//         return self.0.int64_value(_idx);
//     }
// }
//
// #[cfg(not(feature = "torch"))]
// impl Tensor {
//     pub fn size(&self) -> Vec<i64> {
//         unimplemented!()
//     }
//
//     pub fn numel(&self) -> usize {
//         unimplemented!()
//     }
//
//     pub fn min(&self) -> Tensor {
//         unimplemented!()
//     }
//
//     pub fn max(&self) -> Tensor {
//         unimplemented!()
//     }
//
//     pub fn int64_value(&self, _idx: &[i64]) -> i64 {
//         unimplemented!()
//     }
// }

// The implementation of PyTensor is mostly taken from the `tch-pyo3` crate:
// https://github.com/LaurentMazare/tch-rs/blob/main/pyo3-tch/src/lib.rs
//
// The used code comes with the following LICENSE (see
// https://github.com/LaurentMazare/tch-rs?tab=readme-ov-file#license):

// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3_tch::tch;
use pyo3_tch::wrap_tch_err;
use std::fmt::Display;

#[derive(Debug)]
pub struct PyTensor {
    pub tensor: tch::Tensor,
}

impl std::ops::Deref for PyTensor {
    type Target = tch::Tensor;

    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

impl<'source> FromPyObject<'source> for PyTensor {
    fn extract_bound(ob: &Bound<'source, PyAny>) -> PyResult<Self> {
        let ptr = ob.as_ptr() as *mut tch::python::CPyObject;
        let tensor = unsafe { tch::Tensor::pyobject_unpack(ptr) };
        tensor
            .map_err(wrap_tch_err)?
            .ok_or_else(|| {
                let type_ = ob.get_type();
                PyErr::new::<PyTypeError, _>(format!("expected a torch.Tensor, got {type_}"))
            })
            .map(|val| PyTensor { tensor: val })
    }
}

impl<'py> IntoPyObject<'py> for PyTensor {
    type Output = Bound<'py, Self::Target>;
    type Target = PyAny;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        // There is no fallible alternative to ToPyObject/IntoPy at the moment so we return
        // None on errors. https://github.com/PyO3/pyo3/issues/1813
        let v = self.tensor.pyobject_wrap().map_or_else(
            |_| py.None(),
            |ptr| unsafe { PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject) },
        );
        Ok(v.into_pyobject(py)?)
    }
}

// It is safe to share PyTensor between threads (first, the C++ API of
// torch::Tensor should be concurrent, second, the implementation of from and
// into python object use Bounds which hold the Python's GIL.
unsafe impl Sync for PyTensor {}

impl Clone for PyTensor {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.shallow_clone(),
        }
    }
}

impl Display for PyTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(f, "{}", self.tensor)
    }
}

impl From<tch::Tensor> for PyTensor {
    fn from(tensor: tch::Tensor) -> Self {
        Self { tensor }
    }
}
