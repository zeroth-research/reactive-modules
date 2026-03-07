use crate::pytensor::PyTensor;
use pyo3::exceptions::PyValueError;
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
    UWord(u32),
    SWord(u32),
}

#[pymethods]
impl DType {
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

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Float(shape) => {
                write!(f, "Float(")?;
                fmt_comma_separated(f, shape)?;
                write!(f, ")")?;
            }
            DType::Int(shape) => {
                write!(f, "Int(")?;
                fmt_comma_separated(f, shape)?;
                write!(f, ")")?;
            }
            DType::Bool(shape) => {
                write!(f, "Bool(")?;
                fmt_comma_separated(f, shape)?;
                write!(f, ")")?;
            }
            DType::Real(shape) => {
                write!(f, "Real(")?;
                fmt_comma_separated(f, shape)?;
                write!(f, ")")?;
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

#[pyclass(str, frozen)]
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
