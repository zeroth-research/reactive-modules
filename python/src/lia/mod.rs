//use crate::pytensor::PyTensor;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use std::fmt;
use theory::lia::{CmpOp, FlowOp, LinearOp};
use theory::{self, bool, lia, int};

mod atom;
mod module;
mod term;

//pub use atom::Atom;
use crate::IType;
pub use atom::Atom;
pub use module::Module;
pub use term::Term;

// ============================================================================
// Types (wire data types)
// ============================================================================

#[pyclass(frozen, eq, str)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Type(lia::Type);

#[pyfunction]
pub fn Int(i: usize, j: usize) -> Type {
    Type(lia::Type::Int(int::Int(i, j)))
}

#[pyfunction]
pub fn Bool(i: usize, j: usize) -> Type {
    Type(lia::Type::Bool(bool::Bool(i, j)))
}

#[pymethods]
impl Type {
    /// Get the data dimensions of this data type
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.0.shape()
    }

    #[getter]
    fn is_bool(&self) -> bool {
        self.0.is_bool()
    }

    #[getter]
    fn is_int(&self) -> bool {
        self.0.is_int()
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

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Wire
#[pyclass(frozen, eq, hash)]
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub(crate) struct Wire {
    base: base::Wire<lia::Type>,
}

impl Wire {
    pub(crate) fn base(&self) -> &base::Wire<lia::Type> {
        &self.base
    }
}

#[pymethods]
impl Wire {
    #[new]
    pub(crate) fn new(typ: Type) -> Self {
        let base = base::Wire::new(typ.0);
        Self { base }
    }

    #[getter]
    fn id(&self) -> usize {
        self.base.id()
    }

    #[getter]
    fn dtype(&self) -> Type {
        Type(*self.base.dtype()) 
    }


    fn __repr__(&self) -> String {
        format!("{:?}", self.base)
    }
}

impl From<base::Wire<lia::Type>> for Wire {
    fn from(base: base::Wire<lia::Type>) -> Self {
        Self { base: base.into() }
    }
}

impl From<Wire> for base::Wire<lia::Type> {
    fn from(w: Wire) -> Self {
        w.base().clone()
    }
}

impl TryFrom<IType> for lia::LIA {
    type Error = ();

    fn try_from(itype: IType) -> Result<lia::LIA, Self::Error> {
        use lia::LIA;
        match itype {
            IType::And() => Ok(LIA::Bool(bool::Prop::And)),
            IType::Add() => Ok(LIA::Linear(LinearOp::Add)),
            IType::Or() => Ok(LIA::Bool(bool::Prop::Or)),
            IType::Xor() => Ok(LIA::Bool(bool::Prop::Xor)),
            IType::Not() => Ok(LIA::Bool(bool::Prop::Not)),
            IType::ConstBool(v) => Ok(LIA::Bool(bool::Prop::Const(vec![vec![v]]))),
            IType::ConstInt(v) => Ok(LIA::Const(vec![vec![v]])),
            IType::Le() => Ok(LIA::Cmp(CmpOp::Le)),
            IType::Lt() => Ok(LIA::Cmp(CmpOp::Lt)),
            IType::Ge() => Ok(LIA::Cmp(CmpOp::Ge)),
            IType::Gt() => Ok(LIA::Cmp(CmpOp::Gt)),
            IType::Eq() => Ok(LIA::Cmp(CmpOp::Eq)),
            IType::Neq() => Ok(LIA::Cmp(CmpOp::Ne)),
            IType::Id() => Ok(LIA::Flow(FlowOp::Id)),
            IType::Ite() => Ok(LIA::Flow(FlowOp::Ite)),
            IType::ReLU() => Ok(LIA::Linear(LinearOp::ReLU)),
            IType::Argmax() => Ok(LIA::Linear(LinearOp::Argmax)),
            _ => Err(()),
        }
    }
}

impl From<lia::LIA> for IType {
    fn from(t: lia::LIA) -> IType {
        use lia::LIA;
        match t {
            LIA::Const(cm) => {
                if cm.len() == 1 && cm[0].len() == 1 {
                    IType::ConstInt(cm[0][0])
                } else {
                    unimplemented!()
                }
            }
            LIA::Bool(op) => match &op {
                bool::Prop::Const(cm) => {
                    if cm.len() == 1 && cm[0].len() == 1 {
                        IType::ConstBool(cm[0][0])
                    } else {
                        unimplemented!()
                    }
                }
                bool::Prop::And => IType::And(),
                bool::Prop::Or => IType::Or(),
                bool::Prop::Xor => IType::Xor(),
                bool::Prop::Not => IType::Not(),
            },
            LIA::Linear(op) => match &op {
                LinearOp::Linear(_, _) => IType::Linear(),
                LinearOp::ReLU => IType::ReLU(),
                LinearOp::Argmax => IType::Argmax(),
                _ => unimplemented!(),
            },
            LIA::Cmp(op) => match op {
                CmpOp::Le => IType::Le(),
                CmpOp::Lt => IType::Lt(),
                CmpOp::Ge => IType::Ge(),
                CmpOp::Gt => IType::Gt(),
                CmpOp::Eq => IType::Eq(),
                CmpOp::Ne => IType::Neq(),
            },
            LIA::Flow(op) => match op {
                FlowOp::Id => IType::Id(),
                FlowOp::Ite => IType::Ite(),
            },
        }
    }
}
