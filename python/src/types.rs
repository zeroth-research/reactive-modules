#![allow(non_snake_case)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt;

fn parse_dims(dims: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<(usize, usize)> {
    match dims.len() {
        // DType.Bool([n]) or DType.Bool([m, n]) — single list/sequence arg
        1 => {
            let first = dims.get_item(0)?;
            if let Ok(list) = first.extract::<Vec<usize>>() {
                return match list.len() {
                    1 => Ok((1, list[0])),
                    2 => Ok((list[0], list[1])),
                    n => Err(PyValueError::new_err(format!(
                        "expected 1 or 2 dimensions, got {n}"
                    ))),
                };
            }
            Ok((1, first.extract::<usize>()?))
        }
        2 => Ok((
            dims.get_item(0)?.extract::<usize>()?,
            dims.get_item(1)?.extract::<usize>()?,
        )),
        n => Err(PyValueError::new_err(format!(
            "expected 1 or 2 dimensions, got {n}"
        ))),
    }
}

// ============================================================================
// DType
// ============================================================================

#[pyclass(frozen, eq, str)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DType(pub(crate) theory::python::Type);

#[pymethods]
impl DType {
    #[staticmethod]
    #[pyo3(signature = (*dims))]
    fn Bool(dims: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        let (rows, cols) = parse_dims(dims)?;
        Ok(DType(theory::python::Type::Bool(theory::bool::Bool(
            rows, cols,
        ))))
    }

    #[staticmethod]
    #[pyo3(signature = (*dims))]
    fn Int(dims: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        let (rows, cols) = parse_dims(dims)?;
        Ok(DType(theory::python::Type::Int(theory::int::Int(
            rows, cols,
        ))))
    }

    #[staticmethod]
    #[pyo3(signature = (*dims))]
    fn Float(dims: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        let (rows, cols) = parse_dims(dims)?;
        Ok(DType(theory::python::Type::Float(theory::float::Float(
            rows, cols,
        ))))
    }

    #[staticmethod]
    #[pyo3(signature = (*dims))]
    fn Real(dims: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        let (rows, cols) = parse_dims(dims)?;
        Ok(DType(theory::python::Type::Real(theory::real::Real(
            rows, cols,
        ))))
    }

    #[staticmethod]
    fn UWord(bw: usize) -> PyResult<Self> {
        Ok(DType(theory::python::Type::BV32(theory::bv::BV::U(bw, 1, 1))))
    }

    #[staticmethod]
    fn SWord(bw: usize) -> PyResult<Self> {
        Ok(DType(theory::python::Type::BV32(theory::bv::BV::S(bw, 1, 1))))
    }

    /// [rows, cols] shape of this dtype; errors for BV
    #[getter]
    fn shape(&self) -> PyResult<Vec<usize>> {
        match &self.0 {
            theory::python::Type::Bool(t) => {
                let (r, c) = t.shape();
                Ok(vec![r, c])
            }
            theory::python::Type::Int(t) => {
                let (r, c) = t.shape();
                Ok(vec![r, c])
            }
            theory::python::Type::Float(t) => {
                let (r, c) = t.shape();
                Ok(vec![r, c])
            }
            theory::python::Type::Real(t) => {
                let (r, c) = t.shape();
                Ok(vec![r, c])
            }
            theory::python::Type::BV32(t) => {
                let (r, c) = t.shape();
                Ok(vec![r, c])
            }
        }
    }

    /// Same dtype variant but with a new shape
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        let (rows, cols) = match shape.len() {
            1 => (1, shape[0]),
            2 => (shape[0], shape[1]),
            n => return Err(PyValueError::new_err(format!(
                "reshape: expected 1 or 2 dimensions, got {n}"
            ))),
        };
        let inner = match &self.0 {
            theory::python::Type::Bool(_) =>
                theory::python::Type::Bool(theory::bool::Bool(rows, cols)),
            theory::python::Type::Int(_) =>
                theory::python::Type::Int(theory::int::Int(rows, cols)),
            theory::python::Type::Float(_) =>
                theory::python::Type::Float(theory::float::Float(rows, cols)),
            theory::python::Type::Real(_) =>
                theory::python::Type::Real(theory::real::Real(rows, cols)),
            theory::python::Type::BV32(t) => match t {
                theory::bv::BV::U(bw, _, _) =>
                    theory::python::Type::BV32(theory::bv::BV::U(*bw, rows, cols)),
                theory::bv::BV::S(bw, _, _) =>
                    theory::python::Type::BV32(theory::bv::BV::S(*bw, rows, cols)),
            },
        };
        Ok(DType(inner))
    }

    fn is_bool(&self) -> bool {
        matches!(self.0, theory::python::Type::Bool(_))
    }

    fn is_int(&self) -> bool {
        matches!(self.0, theory::python::Type::Int(_))
    }

    fn is_float(&self) -> bool {
        matches!(self.0, theory::python::Type::Float(_))
    }

    fn is_real(&self) -> bool {
        matches!(self.0, theory::python::Type::Real(_))
    }

    fn is_bv(&self) -> bool {
        matches!(self.0, theory::python::Type::BV32(_))
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<theory::python::Type> for DType {
    fn from(t: theory::python::Type) -> DType {
        DType(t)
    }
}

impl From<DType> for theory::python::Type {
    fn from(t: DType) -> theory::python::Type {
        t.0
    }
}
