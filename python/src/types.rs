#![allow(non_snake_case)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt;
use std::sync::Once;
use theory::Theory;

static FLOAT_WARNING: Once = Once::new();

fn parse_dims(dims: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<[usize; 2]> {
    match dims.len() {
        1 => {
            let first = dims.get_item(0)?;
            if let Ok(list) = first.extract::<Vec<usize>>() {
                return match list.len() {
                    0 => Ok([1, 1]),
                    1 => Ok([1, list[0]]),
                    2 => Ok([list[0], list[1]]),
                    n => Err(PyValueError::new_err(format!(
                        "expected 0–2 dimensions, got {n}"
                    ))),
                };
            }
            Ok([1, first.extract::<usize>()?])
        }
        2 => Ok([
            dims.get_item(0)?.extract::<usize>()?,
            dims.get_item(1)?.extract::<usize>()?,
        ]),
        0 => Ok([1, 1]),
        n => Err(PyValueError::new_err(format!(
            "expected 0–2 dimensions, got {n}"
        ))),
    }
}

// ============================================================================
// DType enum (wire data types)
// ============================================================================

#[pyclass(frozen, eq, str)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DType(pub(crate) theory::any::Type);

#[pymethods]
impl DType {
    #[staticmethod]
    #[pyo3(signature = (*dims))]
    pub fn Bool(dims: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        Ok(DType(theory::any::Type::Bool(parse_dims(dims)?)))
    }

    #[staticmethod]
    #[pyo3(signature = (*dims))]
    pub fn Int(dims: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        Ok(DType(theory::any::Type::Int(parse_dims(dims)?)))
    }

    #[staticmethod]
    #[pyo3(signature = (*dims))]
    pub fn Real(dims: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        Ok(DType(theory::any::Type::Real(parse_dims(dims)?)))
    }

    #[staticmethod]
    #[pyo3(signature = (*dims))]
    pub fn Float(dims: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        FLOAT_WARNING.call_once(|| {
            eprintln!("warning: DType.Float is treated as DType.Real");
        });
        Ok(DType(theory::any::Type::Real(parse_dims(dims)?)))
    }

    #[staticmethod]
    pub fn BV(bw: usize) -> Self {
        DType(theory::any::Type::BV(bw, [1, 1]))
    }

    /// Get the data dimensions of this data type
    #[getter]
    pub(crate) fn shape(&self) -> Vec<usize> {
        match &self.0 {
            theory::any::Type::Bool(s)
            | theory::any::Type::Real(s)
            | theory::any::Type::Int(s) => s.to_vec(),
            theory::any::Type::BV(_, s) => s.to_vec(),
        }
    }

    /// Create the same (Tensor) dtype but with a different shape
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        let arr: [usize; 2] = shape
            .as_slice()
            .try_into()
            .map_err(|_| PyValueError::new_err("shape must have exactly 2 elements"))?;
        match &self.0 {
            theory::any::Type::Bool(_) => Ok(DType(theory::any::Type::Bool(arr))),
            theory::any::Type::Int(_) => Ok(DType(theory::any::Type::Int(arr))),
            theory::any::Type::Real(_) => Ok(DType(theory::any::Type::Real(arr))),
            theory::any::Type::BV(_, _) => {
                Err(PyValueError::new_err("cannot reshape word-level types"))
            }
        }
    }

    pub fn is_bool(&self) -> bool {
        matches!(self.0, theory::any::Type::Bool(_))
    }
    pub fn is_int(&self) -> bool {
        matches!(self.0, theory::any::Type::Int(_))
    }
    pub fn is_float(&self) -> bool {
        false
    }
    pub fn is_real(&self) -> bool {
        matches!(self.0, theory::any::Type::Real(_))
    }
    pub fn is_bv(&self) -> bool {
        matches!(self.0, theory::any::Type::BV(_, _))
    }

    pub fn bv_bitwidth(&self) -> PyResult<u32> {
        match &self.0 {
            theory::any::Type::BV(bw, _) => Ok(*bw as u32),
            _ => Err(pyo3::exceptions::PyTypeError::new_err("not a BV type")),
        }
    }
}

impl DType {
    pub fn is_scalar(&self) -> bool {
        self.shape().iter().all(|x| *x == 1)
    }

    pub fn is_same_kind(&self, b: &Self) -> bool {
        use theory::any::Type::*;
        matches!(
            (&self.0, &b.0),
            (Bool(_), Bool(_)) | (Real(_), Real(_)) | (Int(_), Int(_)) | (BV(_, _), BV(_, _))
        )
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ============================================================================
// IType (instruction / operation types)
// ============================================================================
#[pyclass(str, frozen)]
#[derive(Debug, Clone)]
pub struct IType(pub(crate) theory::any::Any);

impl fmt::Display for IType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Theory for IType {
    type DType = DType;
    const NAME: &'static str = "Python-Any";

    fn check<R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<Self::DType> + fmt::Display,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        fn to_type<D: TryInto<DType> + fmt::Display>(d: D) -> Result<theory::any::Type, String> {
            d.try_into()
                .map(|dt: DType| dt.0)
                .map_err(|_| "type conversion failed".to_string())
        }
        let read_vec: Vec<theory::any::Type> =
            read.into_iter().map(to_type::<D>).collect::<Result<_, _>>()?;
        let write_vec: Vec<theory::any::Type> =
            write.into_iter().map(to_type::<D>).collect::<Result<_, _>>()?;
        self.0.check(read_vec, write_vec)
    }
}
