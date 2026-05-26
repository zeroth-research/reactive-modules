use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt;
use theory::Theory;

// ============================================================================
// DType enum (wire data types)
// ============================================================================

#[pyclass(frozen, eq, str)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DType(pub(crate) theory::any::Type);

#[pymethods]
impl DType {
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

    fn check<R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<Self::DType>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        fn to_type<D: TryInto<DType>>(d: D) -> Result<theory::any::Type, String> {
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
