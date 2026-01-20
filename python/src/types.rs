use pyo3::{exceptions::PyValueError, prelude::*};
use std::fmt;

use crate::pytensor::PyTensor;

// ============================================================================
// DType enum (wire data types)
// ============================================================================

#[pyclass]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DType {
    Tensor(Vec<usize>),
    Bool(),
    Int(),
    Float(),
}

fn parse_dim(dim: &str) -> Option<Vec<usize>> {
    Some(
        dim.split(',')
            .map(str::parse)
            .collect::<Result<_, _>>()
            .ok()?,
    )
}

impl std::str::FromStr for DType {
    type Err = String;

    fn from_str(ty: &str) -> Result<Self, Self::Err> {
        match ty {
            "Bool" => Ok(DType::Bool()),
            "Int" => Ok(DType::Int()),
            "Float" => Ok(DType::Float()),
            _ => {
                if let Some(dim) = ty.strip_prefix("Tensor<")
                    && let Some(inner) = dim.strip_suffix(">")
                    && let Some(dims) = parse_dim(inner)
                {
                    return Ok(DType::Tensor(dims));
                }

                Err(format!("Cannot convert `{}` to DType", ty))
            }
        }
    }
}

#[pymethods]
impl DType {
    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        s.parse().map_err(|e| PyValueError::new_err(e))
    }

    /// Get the data dimensions of this data type
    fn dims(&self) -> Vec<usize> {
        match &self {
            DType::Bool() | DType::Int() | DType::Float() => vec![1],
            DType::Tensor(shape) => shape.clone(),
        }
    }

    fn is_tensor(&self) -> bool {
        match &self {
            DType::Tensor(_) => true,
            _ => false,
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __str__(&self) -> String {
        self.to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Tensor(_) => write!(f, "Tensor<...>"),
            DType::Bool() => write!(f, "Bool"),
            DType::Int() => write!(f, "Int"),
            DType::Float() => write!(f, "Float"),
        }
    }
}

// ============================================================================
// IType enum (flat structure for PyO3 compatibility)
// ============================================================================

#[pyclass]
#[derive(Debug, Clone)]
pub enum IType {
    // Arithmetic operations
    Add(),
    Sub(),
    Mul(),
    Div(),
    MatMul(),

    // Comparison operations
    Eq(),
    Neq(),
    Lt(),
    Le(),
    Gt(),
    Ge(),

    // Logical operations
    And(),
    Or(),
    Not(),

    // Control flow
    Ite(),

    // Special operations
    Id(),
    Argmax(),

    // Constants
    Tensor(PyTensor),
}

impl fmt::Display for IType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IType::Add() => write!(f, "Add"),
            IType::Sub() => write!(f, "Sub"),
            IType::Mul() => write!(f, "Mul"),
            IType::Div() => write!(f, "Div"),
            IType::MatMul() => write!(f, "MatMul"),
            IType::Eq() => write!(f, "Eq"),
            IType::Neq() => write!(f, "Neq"),
            IType::Lt() => write!(f, "Lt"),
            IType::Le() => write!(f, "Le"),
            IType::Gt() => write!(f, "Gt"),
            IType::Ge() => write!(f, "Ge"),
            IType::And() => write!(f, "And"),
            IType::Or() => write!(f, "Or"),
            IType::Not() => write!(f, "Not"),
            IType::Ite() => write!(f, "Ite"),
            IType::Id() => write!(f, "Id"),
            IType::Argmax() => write!(f, "Argmax"),
            IType::Tensor(_) => write!(f, "Tensor(...)"),
        }
    }
}

unsafe impl Sync for IType {}
