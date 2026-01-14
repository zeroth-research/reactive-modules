use pyo3::prelude::*;
use std::fmt;

// ============================================================================
// Tensor wrapper for constants
// ============================================================================

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
pub struct MyTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

#[pymethods]
impl MyTensor {
    #[new]
    #[pyo3(signature = (data, shape = None))]
    pub fn new(data: Vec<f32>, shape: Option<Vec<usize>>) -> Self {
        let shape = shape.unwrap_or_else(|| vec![data.len()]);
        Self { data, shape }
    }
    
    #[getter]
    pub fn data(&self) -> Vec<f32> {
        self.data.clone()
    }
    
    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
    
    fn __repr__(&self) -> String {
        format!("MyTensor(shape={:?}, data=[...{}])", self.shape, self.data.len())
    }
}

// Implement Eq manually since f32 doesn't implement Eq
impl Eq for MyTensor {}

impl std::hash::Hash for MyTensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.shape.hash(state);
        // For f32, we hash the bit representation
        for &f in &self.data {
            f.to_bits().hash(state);
        }
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