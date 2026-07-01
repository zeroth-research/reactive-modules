// Pick what tensor implementation to use based on the set features
#[cfg(not(feature = "torch"))]
pub use stub::PyTensor;
#[cfg(feature = "torch")]
pub use torch::PyTensor;

/// Tensor backing for the `torch` feature: a real [`tch::Tensor`] that can be
/// converted to/from Python `torch.Tensor` objects via PyO3.
#[cfg(feature = "torch")]
mod torch {
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
}

/// Placeholder tensor for builds without the `torch` feature.
///
/// It carries no data at the moment: it exists only so the op enums
/// (which hold `PyTensor`) still type-check when this crate is used without
/// PyO3 and/or Torch, e.g., from the `base` crate.
/// If required, we can turn this into a basic working implementation of tensor
/// that would be used if torch is not available or desired.
#[cfg(not(feature = "torch"))]
mod stub {
    use std::fmt::{self, Display};

    // The implementation panics with this message if an attempt to use it is made
    const MSG: &str = "PyTensor operations require the `torch` feature";

    #[derive(Clone, Debug)]
    pub struct PyTensor;

    // The subset of the `tch::Tensor` API touched by the constant-op checks.
    impl PyTensor {
        pub fn size(&self) -> Vec<i64> {
            unreachable!("{MSG}")
        }

        pub fn numel(&self) -> i64 {
            unreachable!("{MSG}")
        }

        pub fn min(&self) -> Self {
            unreachable!("{MSG}")
        }

        pub fn max(&self) -> Self {
            unreachable!("{MSG}")
        }

        pub fn int64_value(&self, _index: &[i64]) -> i64 {
            unreachable!("{MSG}")
        }
    }

    impl Display for PyTensor {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "<tensor>")
        }
    }

    // With `pyo3` (but not `torch`) the op enums still derive `#[pyclass]`, whose
    // getters/constructors require these conversions to exist. They are never
    // called — that path only runs from Python, which needs `torch`.
    #[cfg(feature = "pyo3")]
    mod py {
        use super::{MSG, PyTensor};
        use pyo3::prelude::*;

        impl<'py> IntoPyObject<'py> for PyTensor {
            type Target = PyAny;
            type Output = Bound<'py, PyAny>;
            type Error = PyErr;

            fn into_pyobject(self, _py: Python<'py>) -> Result<Self::Output, Self::Error> {
                unreachable!("{MSG}")
            }
        }

        impl<'source> FromPyObject<'source> for PyTensor {
            fn extract_bound(_ob: &Bound<'source, PyAny>) -> PyResult<Self> {
                unreachable!("{MSG}")
            }
        }
    }
}
