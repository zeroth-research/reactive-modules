use crate::pytensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt;
use theory::{self, bool, lia};

// ============================================================================
// DType enum (wire data types)
// ============================================================================

//#[pyclass(frozen, eq, str)]
#[pyclass(frozen, eq)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PyBool(bool::Bool);

impl PyBool {
    fn shape(&self) -> (usize, usize) {
        self.0.shape()
    }
}

#[pyclass(frozen, eq, str)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DType {
    Bool(PyBool),
    Int(usize, usize),
    Float(usize, usize),
    Real(usize, usize),
    UWord(u32),
    SWord(u32),
}

#[pymethods]
impl DType {
    /// Get the data dimensions of this data type
    #[getter]
    fn shape(&self) -> (usize, usize) {
        match &self {
            DType::Float(i, j) => (*i, *j),
            DType::Int(i, j) => (*i, *j),
            DType::Bool(b) => b.shape(),
            DType::Real(i, j) => (*i, *j),
            DType::UWord(_) | DType::SWord(_) => (1, 1),
        }
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

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Float(i, j) => {
                write!(f, "Float(")?;
                fmt_comma_separated(f, &vec![*i, *j])?;
                write!(f, ")")?;
            }
            DType::Int(i, j) => {
                write!(f, "Int(")?;
                fmt_comma_separated(f, &vec![*i, *j])?;
                write!(f, ")")?;
            }
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

// ============================================================================
// IType enum (flat structure for PyO3 compatibility)
// ============================================================================

#[pyclass(frozen)]
//#[pyclass(str, frozen)]
#[derive(Debug, Clone)]
pub struct LiaIType(theory::lia::LIA);

#[pyclass(str, frozen)]
#[derive(Debug, Clone)]
pub enum IType {
    // Arithmetic operations
    Lia(LiaIType),
    Div(),
    Abs(),

    // Transcendental functions
    Sin(),
    Cos(),

    // Special operations
    Id(),
    // index of maximal value in the flattened tensor
    Argmax(),
    // ReLU activation: max(0, x)
    ReLU(),
    // Tanh activation
    Tanh(),
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

impl TryFrom<DType> for theory::lia::DType {
    type Error = ();

    fn try_from(dt: DType) -> Result<lia::DType, ()> {
        match dt {
            DType::Bool(pybool) => Ok(lia::DType::Bool(pybool.0)),
            _ => Err(()),
        }
    }
}

impl theory::Theory for IType {
    type DType = DType;

    fn type_check<'a, R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<&'a Self::DType>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
        Self::DType: 'a,
    {
        match self {
            _ => unimplemented!(),
        }
    }
}

impl fmt::Display for IType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // TODO: add formatter to lia..
            IType::Lia(lia) => write!(f, "{:?}", lia),
            IType::Div() => write!(f, "Div"),
            IType::Abs() => write!(f, "Abs"),
            IType::Sin() => write!(f, "Sin"),
            IType::Cos() => write!(f, "Cos"),
            IType::Id() => write!(f, "Id"),
            IType::Argmax() => write!(f, "Argmax"),
            IType::ReLU() => write!(f, "ReLU"),
            IType::Tanh() => write!(f, "Tanh"),
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
