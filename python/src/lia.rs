use crate::pytensor::PyTensor;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use std::fmt;
use theory::{self, bool, lia};

// ============================================================================
// DType enum (wire data types)
// ============================================================================

//#[pyclass(frozen, eq, str)]
#[pyclass(frozen, eq)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LiaDType(lia::DType);

#[pymethods]
impl LiaDType {
    /// Get the data dimensions of this data type
    #[getter]
    fn shape(&self) -> (usize, usize) {
        unimplemented!()
    }

    // Create the same (Tensor) dtype but with a different shape
    //fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
    //    match self {
    //        DType::Bool(_) => Ok(DType::Bool(shape)),
    //        DType::Int(_) => Ok(DType::Int(shape)),
    //        DType::Float(_) => Ok(DType::Float(shape)),
    //        DType::Real(_) => Ok(DType::Real(shape)),
    //        DType::UWord(_) | DType::SWord(_) => {
    //            Err(PyValueError::new_err("cannot reshape word-level types"))
    //        }
    //    }
    //}
}

fn fmt_comma_separated(f: &mut fmt::Formatter<'_>, items: &Vec<usize>) -> fmt::Result {
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{item}")?;
    }
    Ok(())
}

impl fmt::Display for LiaDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unimplemented!();
        match self {
            //DType::Float(i, j) => {
            //    write!(f, "Float(")?;
            //    fmt_comma_separated(f, &vec![*i, *j])?;
            //    write!(f, ")")?;
            //}
            //DType::Int(i, j) => {
            //    write!(f, "Int(")?;
            //    fmt_comma_separated(f, &vec![*i, *j])?;
            //    write!(f, ")")?;
            //}
            //DType::Bool(shape) => {
            //    write!(f, "Bool(")?;
            //    fmt_comma_separated(f, shape)?;
            //    write!(f, ")")?;
            //}
            //DType::Real(shape) => {
            //    write!(f, "Real(")?;
            //    fmt_comma_separated(f, shape)?;
            //    write!(f, ")")?;
            //}
            //DType::UWord(n) => {
            //    write!(f, "UWord<{}>", n)?;
            //}
            //DType::SWord(n) => {
            //    write!(f, "SWord<{}>", n)?;
            //}
            _ => unimplemented!(),
        };
        Ok(())
    }
}

/// Wire
#[pyclass(frozen, eq, hash)]
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub(crate) struct LiaWire {
    base: base::Wire<lia::DType>,
}

impl LiaWire {
    pub(crate) fn base(&self) -> &base::Wire<lia::DType> {
        &self.base
    }
}

impl From<base::Wire<lia::DType>> for LiaWire {
    fn from(base: base::Wire<lia::DType>) -> Self {
        Self { base: base.into() }
    }
}

impl From<LiaWire> for base::Wire<lia::DType> {
    fn from(w: LiaWire) -> Self {
        w.base().clone()
    }
}

// ============================================================================
// IType enum (flat structure for PyO3 compatibility)
// ============================================================================

#[pyclass(frozen)]
//#[pyclass(str, frozen)]
#[derive(Debug, Clone)]
pub struct LiaIType(lia::LIA);

#[pyclass(frozen)]
pub(crate) struct LIATerm {
    base: base::Term<lia::LIA>,
}

fn try_wire_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = base::Wire<lia::DType>>>
where
    for<'py> LiaWire: pyo3::FromPyObject<'py>,
    base::Wire<lia::DType>: From<LiaWire>,
{
    seq.iter()?
        .map(|item| item?.extract::<LiaWire>().map(Into::into))
        .collect::<PyResult<Vec<_>>>()
        .map(Vec::into_iter)
}

#[pymethods]
impl LIATerm {
    #[staticmethod]
    fn function(
        itype: LiaIType,
        write: &Bound<'_, PyAny>,
        read: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let write = try_wire_iter_cloned(write)?;
        let read = try_wire_iter_cloned(read)?;

        match base::Term::function(itype.0, write, read) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn constant(itype: LiaIType, write: &Bound<'_, PyAny>) -> PyResult<Self> {
        let write = try_wire_iter_cloned(write)?;

        match base::Term::constant(itype.0, write) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[new]
    #[pyo3(signature = (itype, write, read = None))]
    fn new(
        itype: LiaIType,
        write: &Bound<'_, PyAny>,
        read: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        match read {
            Some(read) => Self::function(itype, write, read),
            None => Self::constant(itype, write),
        }
    }

    //#[getter]
    //fn write(slf: PyRef<'_, Self>) -> PyResult<TermInterface> {
    //    Self::interface(slf, TermInterfaceType::Write)
    //}
    //
    //#[getter]
    //fn read(slf: PyRef<'_, Self>) -> PyResult<TermInterface> {
    //    Self::interface(slf, TermInterfaceType::Read)
    //}

    #[getter]
    fn itype(&self) -> LiaIType {
        LiaIType(self.base.itype().clone())
    }

    //fn __str__(&self) -> String {
    //    self.base.to_string()
    //}
    //
    //fn __repr__(&self) -> String {
    //    format!("{:?}", self.base)
    //}
}

impl From<base::Term<lia::LIA>> for LIATerm {
    fn from(base: base::Term<lia::LIA>) -> Self {
        Self { base }
    }
}

impl From<LiaDType> for lia::DType {
    fn from(dt: LiaDType) -> Self {
        dt.0
    }
}
