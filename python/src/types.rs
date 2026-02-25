use crate::pytensor::PyTensor;
use pyo3::prelude::*;
use std::fmt;

// ============================================================================
// DType enum (wire data types)
// ============================================================================

#[pyclass(frozen, eq, str)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DType {
    // at this moment, we keep the DType flat and encode the type
    // of elements in the names
    Bool(Vec<usize>),
    Int(Vec<usize>),
    Float(Vec<usize>),
    Real(Vec<usize>),
}

#[pymethods]
impl DType {
    // #[staticmethod]
    // fn from_str(s: &str) -> PyResult<Self> {
    //     s.parse().map_err(|e| PyValueError::new_err(e))
    // }

    /// Get the data dimensions of this data type
    #[getter]
    fn shape(&self) -> &Vec<usize> {
        match &self {
            DType::Float(shape) => shape,
            DType::Int(shape) => shape,
            DType::Bool(shape) => shape,
            DType::Real(shape) => shape,
        }
    }

    /// Return whether the type of elements is the same
    // This method is necesary because we do not expose [PrimitiveType]
    fn eq_dtype(&self, other: &Self) -> bool {
        match (self, other) {
            (DType::Float(_), DType::Float(_)) => true,
            (DType::Int(_), DType::Int(_)) => true,
            (DType::Bool(_), DType::Bool(_)) => true,
            (DType::Real(_), DType::Real(_)) => true,
            _ => false,
        }
    }

    /// Create the same (Tensor) dtype but with a different shape
    fn reshape(&self, shape: Vec<usize>) -> Self {
        match self {
            DType::Bool(_) => DType::Bool(shape),
            DType::Int(_) => DType::Int(shape),
            DType::Float(_) => DType::Float(shape),
            DType::Real(_) => DType::Real(shape),
        }
    }

    /// Get the kind/variant of this dtype
    fn kind(&self) -> &'static str {
        match self {
            DType::Bool(_) => "Bool",
            DType::Int(_) => "Int",
            DType::Float(_) => "Float",
            DType::Real(_) => "Real",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = match self {
            DType::Float(shape) => {
                write!(f, "Float(")?;
                shape
            }
            DType::Int(shape) => {
                write!(f, "Int(")?;
                shape
            }
            DType::Bool(shape) => {
                write!(f, "Bool(")?;
                shape
            }
            DType::Real(shape) => {
                write!(f, "Real(")?;
                shape
            }
        };
        let mut first: bool = true;
        for dim in shape {
            if !first {
                write!(f, ", ")?;
                first = false;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

// ============================================================================
// IType enum (flat structure for PyO3 compatibility)
// ============================================================================

#[pyclass(str)]
#[derive(Debug, Clone)]
pub enum IType {
    // Arithmetic operations
    Add(),
    Sub(),
    Mul(),
    Div(),
    MatMul(), // consider differentiation between matmul operators, or parameterisation.
    // To be designed with compliance with lower level platform in mind

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
    // index of maximal value in the flattened tensor
    Argmax(),
    // ReLU activation: max(0, x)
    ReLU(),

    // Tensor operations
    TensorGet(),
    TensorSet(),
    TensorSum(),
    TensorMean(),
    TensorMax(),

    // Constants
    Tensor(PyTensor),

    // Symbol referring to uninterpreted constants or functions,
    // whose signature is known in the context, i.e., the current theory
    Uninterpreted(String),
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
            IType::ReLU() => write!(f, "ReLU"),
            IType::TensorGet() => write!(f, "TensorGet"),
            IType::TensorSet() => write!(f, "TensorSet"),
            IType::TensorSum() => write!(f, "TensorSum"),
            IType::TensorMean() => write!(f, "TensorMean"),
            IType::TensorMax() => write!(f, "TensorMax"),
            IType::Tensor(t) => {
                let flat = t.tensor.view([-1]);

                if let Ok(vals) = Vec::<f64>::try_from(&flat) {
                    let _ = write!(f, "Tensor([");
                    for (n, v) in vals.iter().take(5).enumerate() {
                        if n == 0 {
                            let _ = write!(f, "{}", v);
                        } else {
                            let _ = write!(f, " {}", v);
                        }
                    }
                    if flat.numel() > 3 {
                        let _ = write!(f, " ...");
                    }
                    write!(f, "])")
                } else {
                    write!(f, "Tensor({})", flat)
                }
            }
            IType::Uninterpreted(t) => write!(f, "{t}"),
        }
    }
}

unsafe impl Sync for IType {}
