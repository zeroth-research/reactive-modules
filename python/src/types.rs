use pyo3::prelude::*;
use std::fmt;

// ============================================================================
// Tensor wrapper for constants
// ============================================================================

#[pyclass]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MyTensor {
    data: Vec<usize>,
}

#[pymethods]
impl MyTensor {
    #[new]
    pub fn new(data: Vec<usize>) -> Self {
        Self { data }
    }
}

// ============================================================================
// IType enum (flat structure for PyO3 compatibility)
// ============================================================================

#[pyclass]
#[derive(Debug, Clone, Eq, PartialEq)]
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
    Const(MyTensor),
}

// ============================================================================
// DType enum (wire data types)
// ============================================================================

#[pyclass]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DType {
    Tensor(),
    Bool(),
    Int(),
    Float(),
}

// ============================================================================
// Display implementations
// ============================================================================

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
            IType::Const(_) => write!(f, "Const(tensor)"),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Tensor() => write!(f, "Tensor"),
            DType::Bool() => write!(f, "Bool"),
            DType::Int() => write!(f, "Int"),
            DType::Float() => write!(f, "Float"),
        }
    }
}