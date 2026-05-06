#![allow(non_snake_case)]

use pyo3::prelude::*;
use theory::bool::BoolOp;
use theory::float::ArithFloat;
use theory::int::ArithInt;
use theory::python::{CmpOp, FlowOp, NNOp, TensorOp};
use theory::real::ArithReal;

// IType wrapper

#[pyclass(frozen)]
#[derive(Debug, Clone)]
pub struct IType(pub(crate) theory::python::IType);

impl From<theory::python::IType> for IType {
    fn from(t: theory::python::IType) -> Self {
        IType(t)
    }
}

impl From<IType> for theory::python::IType {
    fn from(t: IType) -> Self {
        t.0
    }
}

#[pymethods]
impl IType {
    #[classattr]
    fn Bool(py: Python<'_>) -> Py<PyAny> {
        // bind the `BoolIType` to `IType` as attribute,
        // so that we can get it as `IType.Bool` in Python
        py.get_type::<BoolIType>().into_any().unbind()
    }
    #[classattr]
    fn Int(py: Python<'_>) -> Py<PyAny> {
        py.get_type::<IntIType>().into_any().unbind()
    }
    #[classattr]
    fn Float(py: Python<'_>) -> Py<PyAny> {
        py.get_type::<FloatIType>().into_any().unbind()
    }
    #[classattr]
    fn Real(py: Python<'_>) -> Py<PyAny> {
        py.get_type::<RealIType>().into_any().unbind()
    }
    #[classattr]
    fn Cmp(py: Python<'_>) -> Py<PyAny> {
        py.get_type::<CmpIType>().into_any().unbind()
    }
    #[classattr]
    fn NN(py: Python<'_>) -> Py<PyAny> {
        py.get_type::<NNIType>().into_any().unbind()
    }
    #[classattr]
    fn Tensor(py: Python<'_>) -> Py<PyAny> {
        py.get_type::<TensorIType>().into_any().unbind()
    }
    #[classattr]
    fn Flow(py: Python<'_>) -> Py<PyAny> {
        py.get_type::<FlowIType>().into_any().unbind()
    }

    #[classattr]
    fn Id() -> Self {
        IType(theory::python::IType::Id)
    }
    #[classattr]
    fn Ite() -> Self {
        IType(theory::python::IType::Ite)
    }
    #[classattr]
    fn BVToBool() -> Self {
        IType(theory::python::IType::BVToBool)
    }
    #[classattr]
    fn BVToWord1() -> Self {
        IType(theory::python::IType::BVToWord1)
    }

    #[staticmethod]
    fn Uninterpreted(name: String) -> Self {
        IType(theory::python::IType::Uninterpreted(name))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
}

// IType.Bool namespace

#[pyclass]
pub struct BoolIType;

#[pymethods]
impl BoolIType {
    // NOTE: previously we used `Add()` in Python but now it is `Add`.
    // To change to the old behavior, change `classattr` to `staticmethod`
    #[classattr]
    fn And() -> IType {
        IType(theory::python::IType::Bool(BoolOp::And))
    }
    #[classattr]
    fn Or() -> IType {
        IType(theory::python::IType::Bool(BoolOp::Or))
    }
    #[classattr]
    fn Xor() -> IType {
        IType(theory::python::IType::Bool(BoolOp::Xor))
    }
    #[classattr]
    fn Not() -> IType {
        IType(theory::python::IType::Bool(BoolOp::Not))
    }
    #[staticmethod]
    fn Const(data: Vec<Vec<bool>>) -> IType {
        IType(theory::python::IType::Bool(BoolOp::Const(data)))
    }
}

// IType.Int namespace

#[pyclass]
pub struct IntIType;

#[pymethods]
impl IntIType {
    #[classattr]
    fn Add() -> IType {
        IType(theory::python::IType::Int(ArithInt::Add))
    }
    #[classattr]
    fn Sub() -> IType {
        IType(theory::python::IType::Int(ArithInt::Sub))
    }
    #[classattr]
    fn Mul() -> IType {
        IType(theory::python::IType::Int(ArithInt::Mul))
    }
    #[classattr]
    fn Div() -> IType {
        IType(theory::python::IType::Int(ArithInt::Div))
    }
    #[classattr]
    fn Mod() -> IType {
        IType(theory::python::IType::Int(ArithInt::Mod))
    }
    #[classattr]
    fn Neg() -> IType {
        IType(theory::python::IType::Int(ArithInt::Neg))
    }
    #[classattr]
    fn Abs() -> IType {
        IType(theory::python::IType::Int(ArithInt::Abs))
    }
    #[classattr]
    fn MatMul() -> IType {
        IType(theory::python::IType::Int(ArithInt::MatMul))
    }
    #[staticmethod]
    fn Const(data: Vec<Vec<i64>>) -> IType {
        IType(theory::python::IType::Int(ArithInt::Const(data)))
    }
}

// IType.Float namespace

#[pyclass]
pub struct FloatIType;

#[pymethods]
impl FloatIType {
    #[classattr]
    fn Add() -> IType {
        IType(theory::python::IType::Float(ArithFloat::Add))
    }
    #[classattr]
    fn Sub() -> IType {
        IType(theory::python::IType::Float(ArithFloat::Sub))
    }
    #[classattr]
    fn Mul() -> IType {
        IType(theory::python::IType::Float(ArithFloat::Mul))
    }
    #[classattr]
    fn Div() -> IType {
        IType(theory::python::IType::Float(ArithFloat::Div))
    }
    #[classattr]
    fn Mod() -> IType {
        IType(theory::python::IType::Float(ArithFloat::Mod))
    }
    #[classattr]
    fn Neg() -> IType {
        IType(theory::python::IType::Float(ArithFloat::Neg))
    }
    #[classattr]
    fn Abs() -> IType {
        IType(theory::python::IType::Float(ArithFloat::Abs))
    }
    #[classattr]
    fn MatMul() -> IType {
        IType(theory::python::IType::Float(ArithFloat::MatMul))
    }
    #[staticmethod]
    fn Const(data: Vec<Vec<f64>>) -> IType {
        IType(theory::python::IType::Float(ArithFloat::Const(data)))
    }
}

// IType.Real namespace

#[pyclass]
pub struct RealIType;

#[pymethods]
impl RealIType {
    #[classattr]
    fn Add() -> IType {
        IType(theory::python::IType::Real(ArithReal::Add))
    }
    #[classattr]
    fn Sub() -> IType {
        IType(theory::python::IType::Real(ArithReal::Sub))
    }
    #[classattr]
    fn Mul() -> IType {
        IType(theory::python::IType::Real(ArithReal::Mul))
    }
    #[classattr]
    fn Div() -> IType {
        IType(theory::python::IType::Real(ArithReal::Div))
    }
    #[classattr]
    fn Mod() -> IType {
        IType(theory::python::IType::Real(ArithReal::Mod))
    }
    #[classattr]
    fn Neg() -> IType {
        IType(theory::python::IType::Real(ArithReal::Neg))
    }
    #[classattr]
    fn Abs() -> IType {
        IType(theory::python::IType::Real(ArithReal::Abs))
    }
    #[classattr]
    fn MatMul() -> IType {
        IType(theory::python::IType::Real(ArithReal::MatMul))
    }
    #[classattr]
    fn Sin() -> IType {
        IType(theory::python::IType::Real(ArithReal::Sin))
    }
    #[classattr]
    fn Cos() -> IType {
        IType(theory::python::IType::Real(ArithReal::Cos))
    }
    #[staticmethod]
    fn Const(data: Vec<Vec<f64>>) -> IType {
        IType(theory::python::IType::Real(ArithReal::Const(data)))
    }
}

// IType.Cmp namespace

#[pyclass]
pub struct CmpIType;

#[pymethods]
impl CmpIType {
    #[classattr]
    fn Le() -> IType {
        IType(theory::python::IType::Cmp(CmpOp::Le))
    }
    #[classattr]
    fn Lt() -> IType {
        IType(theory::python::IType::Cmp(CmpOp::Lt))
    }
    #[classattr]
    fn Ge() -> IType {
        IType(theory::python::IType::Cmp(CmpOp::Ge))
    }
    #[classattr]
    fn Gt() -> IType {
        IType(theory::python::IType::Cmp(CmpOp::Gt))
    }
    #[classattr]
    fn Eq() -> IType {
        IType(theory::python::IType::Cmp(CmpOp::Eq))
    }
    #[classattr]
    fn Ne() -> IType {
        IType(theory::python::IType::Cmp(CmpOp::Ne))
    }
}

// IType.NN namespace

#[pyclass]
pub struct NNIType;

#[pymethods]
impl NNIType {
    #[classattr]
    fn ReLU() -> IType {
        IType(theory::python::IType::NN(NNOp::ReLU))
    }
    #[classattr]
    fn Tanh() -> IType {
        IType(theory::python::IType::NN(NNOp::Tanh))
    }
    #[classattr]
    fn Linear() -> IType {
        IType(theory::python::IType::NN(NNOp::Linear))
    }
}

// IType.Tensor namespace

#[pyclass]
pub struct TensorIType;

#[pymethods]
impl TensorIType {
    #[classattr]
    fn Get() -> IType {
        IType(theory::python::IType::Tensor(TensorOp::Get))
    }
    #[classattr]
    fn Set() -> IType {
        IType(theory::python::IType::Tensor(TensorOp::Set))
    }
    #[classattr]
    fn Sum() -> IType {
        IType(theory::python::IType::Tensor(TensorOp::Sum))
    }
    #[classattr]
    fn Mean() -> IType {
        IType(theory::python::IType::Tensor(TensorOp::Mean))
    }
    #[classattr]
    fn Max() -> IType {
        IType(theory::python::IType::Tensor(TensorOp::Max))
    }
    #[classattr]
    fn Argmax() -> IType {
        IType(theory::python::IType::Tensor(TensorOp::Argmax))
    }
}

// IType.Flow namespace

#[pyclass]
pub struct FlowIType;

#[pymethods]
impl FlowIType {
    #[classattr]
    fn Ite() -> IType {
        IType(theory::python::IType::Flow(FlowOp::Ite))
    }
    #[classattr]
    fn Id() -> IType {
        IType(theory::python::IType::Flow(FlowOp::Id))
    }
}
