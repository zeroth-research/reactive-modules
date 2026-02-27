use crate::pytensor::PyTensor;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
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
    UWord(u32),
    SWord(u32),
}

// Keep TensorBool etc. as aliases for backward compatibility in Python code
impl DType {
    pub fn tensor_bool(shape: Vec<usize>) -> Self { DType::Bool(shape) }
    pub fn tensor_int(shape: Vec<usize>) -> Self { DType::Int(shape) }
    pub fn tensor_float(shape: Vec<usize>) -> Self { DType::Float(shape) }
    pub fn tensor_real(shape: Vec<usize>) -> Self { DType::Real(shape) }
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
                PrimitiveType::Float => Ok(DType::Float(dims)),
                PrimitiveType::Int => Ok(DType::Int(dims)),
                PrimitiveType::Bool => Ok(DType::Bool(dims)),
                PrimitiveType::Real => Ok(DType::Real(dims)),
            };
        }

        // Word-level types: UWord<N> / SWord<N>
        if let Some(inner) = ty.strip_prefix("UWord<").and_then(|s| s.strip_suffix(">")) {
            let width: u32 = inner
                .trim()
                .parse()
                .map_err(|_| format!("Invalid UWord width: `{}`", inner))?;
            return Ok(DType::UWord(width));
        }
        if let Some(inner) = ty.strip_prefix("SWord<").and_then(|s| s.strip_suffix(">")) {
            let width: u32 = inner
                .trim()
                .parse()
                .map_err(|_| format!("Invalid SWord width: `{}`", inner))?;
            return Ok(DType::SWord(width));
        }

        // try also aliases
        match ty {
            "Float" => Ok(DType::Float(vec![1])),
            "Int" => Ok(DType::Int(vec![1])),
            "Bool" => Ok(DType::Bool(vec![1])),
            "Real" => Ok(DType::Real(vec![1])),
            _ => Err(format!("Cannot convert `{}` to DType", ty)),
        }
    }
}

#[pymethods]
impl DType {
    // #[staticmethod]
    // fn from_str(s: &str) -> PyResult<Self> {
    //     s.parse().map_err(|e| PyValueError::new_err(e))
    // }

    /// Get the data dimensions of this data type
    #[getter]
    fn shape(&self) -> Vec<usize> {
        match &self {
            DType::Float(shape) => shape.clone(),
            DType::Int(shape) => shape.clone(),
            DType::Bool(shape) => shape.clone(),
            DType::Real(shape) => shape.clone(),
            DType::UWord(_) | DType::SWord(_) => vec![1],
        }
    }

    /// Return whether the type of elements is the same
    fn eq_dtype(&self, other: &Self) -> bool {
        match (self, other) {
            (DType::Float(_), DType::Float(_)) => true,
            (DType::Int(_), DType::Int(_)) => true,
            (DType::Bool(_), DType::Bool(_)) => true,
            (DType::Real(_), DType::Real(_)) => true,
            (DType::UWord(_), DType::UWord(_)) => true,
            (DType::SWord(_), DType::SWord(_)) => true,
            _ => false,
        }
    }

    /// Create the same (Tensor) dtype but with a different shape
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        match self {
            DType::Bool(_) => Ok(DType::Bool(shape)),
            DType::Int(_) => Ok(DType::Int(shape)),
            DType::Float(_) => Ok(DType::Float(shape)),
            DType::Real(_) => Ok(DType::Real(shape)),
            DType::UWord(_) | DType::SWord(_) => {
                Err(PyValueError::new_err("cannot reshape word-level types"))
            }
        }
    }

    /// Get the kind/variant of this dtype
    fn kind(&self) -> &'static str {
        match self {
            DType::Bool(_) => "TensorBool",
            DType::Int(_) => "TensorInt",
            DType::Float(_) => "TensorFloat",
            DType::Real(_) => "TensorReal",
            DType::UWord(_) => "UWord",
            DType::SWord(_) => "SWord",
        }
    }

    /// Get the bit width for word-level types, None for tensor types
    fn width(&self) -> Option<u32> {
        match self {
            DType::UWord(w) | DType::SWord(w) => Some(*w),
            _ => None,
        }
    }
}

fn shape_to_string(shape: &Vec<usize>) -> String {
    shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<String>>()
        .join(", ")
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Float(shape) => {
                write!(f, "Float({})", shape_to_string(shape))?;
            }
            DType::Int(shape) => {
                write!(f, "Int({})", shape_to_string(shape))?;
            }
            DType::Bool(shape) => {
                write!(f, "Bool({})", shape_to_string(shape))?;
            }
            DType::Real(shape) => {
                write!(f, "Real({})", shape_to_string(shape))?;
            }
            DType::UWord(n) => {
                write!(f, "UWord<{}>", n)?;
            }
            DType::SWord(n) => {
                write!(f, "SWord<{}>", n)?;
            }
        };
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
    Mod(),
    Neg(),
    Abs(),
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
    Xor(),
    Xnor(),
    Implies(),

    // Control flow
    Ite(),

    // Special operations
    Id(),
    // index of maximal value in the flattened tensor
    Argmax(),
    // ReLU activation: max(0, x)
    ReLU(),
    // Linear layer: output = input @ weight + bias
    // Reads: [input, weight, bias], Writes: [output]
    Linear(),

    // Tensor operations
    TensorGet(),
    TensorSet(),
    TensorSum(),
    TensorMean(),
    TensorMax(),

    // Word-level operations
    BitSelect(u32, u32),
    Extend(u32),
    ToBool(),
    ToWord1(),
    ToUnsigned(),
    ToSigned(),

    // Constants
    Tensor(PyTensor),
    ConstBool(bool),
    ConstInt(i64),

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
            IType::Mod() => write!(f, "Mod"),
            IType::Neg() => write!(f, "Neg"),
            IType::Abs() => write!(f, "Abs"),
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
            IType::Xor() => write!(f, "Xor"),
            IType::Xnor() => write!(f, "Xnor"),
            IType::Implies() => write!(f, "Implies"),
            IType::Ite() => write!(f, "Ite"),
            IType::Id() => write!(f, "Id"),
            IType::Argmax() => write!(f, "Argmax"),
            IType::ReLU() => write!(f, "ReLU"),
            IType::Linear() => write!(f, "Linear"),
            IType::TensorGet() => write!(f, "TensorGet"),
            IType::TensorSet() => write!(f, "TensorSet"),
            IType::TensorSum() => write!(f, "TensorSum"),
            IType::TensorMean() => write!(f, "TensorMean"),
            IType::TensorMax() => write!(f, "TensorMax"),
            IType::BitSelect(h, l) => write!(f, "BitSelect[{}:{}]", h, l),
            IType::Extend(n) => write!(f, "Extend({})", n),
            IType::ToBool() => write!(f, "ToBool"),
            IType::ToWord1() => write!(f, "ToWord1"),
            IType::ToUnsigned() => write!(f, "ToUnsigned"),
            IType::ToSigned() => write!(f, "ToSigned"),
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
            IType::ConstBool(v) => write!(f, "Const: {}", v),
            IType::ConstInt(v) => write!(f, "Const: {}", v),
            IType::Uninterpreted(t) => write!(f, "{t}"),
        }
    }
}

unsafe impl Sync for IType {}
