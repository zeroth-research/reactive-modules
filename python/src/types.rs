use crate::pytensor::PyTensor;
use pyo3::types::PyTuple;
use pyo3::{exceptions::PyValueError, prelude::*};
use std::fmt;

// ============================================================================
// DType enum (wire data types)
// ============================================================================

#[pyclass(frozen, eq, str)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DType {
    // at this moment, we keep the DType flat and encode the type
    // of elements in the names
    TensorBool(Vec<usize>),
    TensorInt(Vec<usize>),
    TensorFloat(Vec<usize>),
    TensorReal(Vec<usize>),
}

enum PrimitiveType {
    Bool,
    Int,
    Float,
    Real,
}

fn parse_dim_with_type(dim_and_type: &str) -> Option<(Vec<usize>, PrimitiveType)> {
    if let Some((dim, ptype)) = dim_and_type.split_once(';') {
        let dim = dim
            .split(',')
            .map(str::trim)
            .map(str::parse)
            .collect::<Result<_, _>>()
            .ok()?;

        let ptype = match ptype.trim() {
            "Bool" => PrimitiveType::Bool,
            "Int" => PrimitiveType::Int,
            "Float" => PrimitiveType::Float,
            "Real" => PrimitiveType::Real,
            _ => return None,
        };

        return Some((dim, ptype));
    }

    None
}

impl std::str::FromStr for DType {
    type Err = String;

    fn from_str(ty: &str) -> Result<Self, Self::Err> {
        if let Some(dim) = ty.strip_prefix("Tensor<")
            && let Some(inner) = dim.strip_suffix(">")
            && let Some((dims, ptype)) = parse_dim_with_type(inner)
        {
            return match ptype {
                PrimitiveType::Float => Ok(DType::TensorFloat(dims)),
                PrimitiveType::Int => Ok(DType::TensorInt(dims)),
                PrimitiveType::Bool => Ok(DType::TensorBool(dims)),
                PrimitiveType::Real => Ok(DType::TensorReal(dims)),
            };
        }

        // try also aliases
        match ty {
            "Float" => Ok(DType::TensorFloat(vec![1])),
            "Int" => Ok(DType::TensorInt(vec![1])),
            "Bool" => Ok(DType::TensorBool(vec![1])),
            "Real" => Ok(DType::TensorReal(vec![1])),
            _ => Err(format!("Cannot convert `{}` to DType", ty)),
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
    #[getter]
    fn shape(&self) -> &Vec<usize> {
        match &self {
            DType::TensorFloat(shape) => shape,
            DType::TensorInt(shape) => shape,
            DType::TensorBool(shape) => shape,
            DType::TensorReal(shape) => shape,
        }
    }

    /// Return whether the type of elements is the same
    // This method is necesary because we do not expose [PrimitiveType]
    fn eq_dtype(&self, other: &Self) -> bool {
        match (self, other) {
            (DType::TensorFloat(_), DType::TensorFloat(_)) => true,
            (DType::TensorInt(_), DType::TensorInt(_)) => true,
            (DType::TensorBool(_), DType::TensorBool(_)) => true,
            (DType::TensorReal(_), DType::TensorReal(_)) => true,
            _ => false,
        }
    }

    /// Create the same (Tensor) dtype but with a different shape
    fn reshape(&self, shape: Vec<usize>) -> Self {
        match self {
            DType::TensorBool(_) => DType::TensorBool(shape),
            DType::TensorInt(_) => DType::TensorInt(shape),
            DType::TensorFloat(_) => DType::TensorFloat(shape),
            DType::TensorReal(_) => DType::TensorReal(shape),
        }
    }

    /// Get the kind/variant of this dtype
    fn kind(&self) -> &'static str {
        match self {
            DType::TensorBool(_) => "TensorBool",
            DType::TensorInt(_) => "TensorInt",
            DType::TensorFloat(_) => "TensorFloat",
            DType::TensorReal(_) => "TensorReal",
        }
    }
}

impl fmt::Display for PrimitiveType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrimitiveType::Bool => write!(f, "Bool"),
            PrimitiveType::Int => write!(f, "Int"),
            PrimitiveType::Float => write!(f, "Float"),
            PrimitiveType::Real => write!(f, "Real"),
        }
    }
}

fn format_tensor(shape: &Vec<usize>, ptype: PrimitiveType) -> String {
    format!(
        "Tensor<{}; {}>",
        shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(", "),
        ptype
    )
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::TensorFloat(shape) => {
                write!(f, "{}", format_tensor(shape, PrimitiveType::Float))
            }
            DType::TensorInt(shape) => write!(f, "{}", format_tensor(shape, PrimitiveType::Int)),
            DType::TensorBool(shape) => write!(f, "{}", format_tensor(shape, PrimitiveType::Bool)),
            DType::TensorReal(shape) => write!(f, "{}", format_tensor(shape, PrimitiveType::Real)),
        }
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
