#![allow(non_snake_case)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use theory::bool::BoolOp;
use theory::float::ArithFloat;
use theory::int::ArithInt;
use theory::bv::BVTheory;
use theory::python::CastOp;
use theory::python::IType as TheoryIType;
use theory::python::{NNOp, TensorOp};
use theory::real::ArithReal;
use theory::{Arith, CmpOp, FlowOp};

// ============================================================================
// IType — hierarchical concrete op wrapper
// ============================================================================

#[pyclass(frozen, name = "IType")]
#[derive(Debug, Clone)]
pub struct IType(pub(crate) TheoryIType);

impl From<TheoryIType> for IType {
    fn from(t: TheoryIType) -> Self {
        IType(t)
    }
}

impl From<IType> for TheoryIType {
    fn from(t: IType) -> Self {
        t.0
    }
}

#[pymethods]
impl IType {
    // Namespace class-attributes
    #[classattr]
    fn Bool(py: Python<'_>) -> Py<PyAny> {
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
    fn BV(py: Python<'_>) -> Py<PyAny> {
        py.get_type::<BVIType>().into_any().unbind()
    }

    // Top-level singleton ops
    #[classattr]
    fn Id() -> Self {
        IType(TheoryIType::Flow(FlowOp::Id))
    }
    #[classattr]
    fn Ite() -> Self {
        IType(TheoryIType::Flow(FlowOp::Ite))
    }
    #[classattr]
    fn BVToBool() -> Self {
        IType(TheoryIType::Cast(CastOp::BVToBool))
    }
    #[classattr]
    fn BVToWord1() -> Self {
        IType(TheoryIType::Cast(CastOp::BVToWord1))
    }
    #[classattr]
    fn ToUnsigned() -> Self {
        IType(TheoryIType::BV(BVTheory::ToUnsigned()))
    }
    #[classattr]
    fn ToSigned() -> Self {
        IType(TheoryIType::BV(BVTheory::ToSigned()))
    }

    // Static constructor helpers
    #[staticmethod]
    fn Uninterpreted(name: String) -> Self {
        IType(TheoryIType::Uninterpreted(name))
    }

    /// Inline integer scalar constant: `IType.ConstInt(v)` → `IType.Int.Const([[v]])`
    #[staticmethod]
    fn ConstInt(v: i64) -> Self {
        IType(TheoryIType::Int(ArithInt::Const(vec![vec![v]])))
    }

    /// Inline bool scalar constant: `IType.ConstBool(v)` → `IType.Bool.Const([[v]])`
    #[staticmethod]
    fn ConstBool(v: bool) -> Self {
        IType(TheoryIType::Bool(BoolOp::Const(vec![vec![v]])))
    }

    /// Extract bits [high..low] from a bitvector.
    #[staticmethod]
    fn BitSelect(high: usize, low: usize) -> Self {
        IType(TheoryIType::BV(BVTheory::Select(high as u32, low as u32)))
    }

    /// Extend a bitvector to the given width.
    #[staticmethod]
    fn Extend(width: usize) -> Self {
        IType(TheoryIType::BV(BVTheory::Extend(width as u32)))
    }

    /// Convert a torch.Tensor to the appropriate `IType.<Type>.Const(data)`.
    /// 1-D tensors are treated as row vectors (unsqueezed to 1×n).
    #[staticmethod]
    fn from_tensor(t: crate::pytensor::PyTensor) -> PyResult<Self> {
        use tch::Kind;

        let t: tch::Tensor = if t.dim() == 1 {
            t.tensor.unsqueeze(0)
        } else {
            t.tensor.shallow_clone()
        };

        let sz = t.size();
        let (rows, cols) = (sz[0] as usize, sz[1] as usize);

        match t.kind() {
            Kind::Float | Kind::Double | Kind::BFloat16 | Kind::Half => {
                let data: Vec<Vec<f64>> = (0..rows)
                    .map(|i| {
                        (0..cols)
                            .map(|j| t.double_value(&[i as i64, j as i64]))
                            .collect()
                    })
                    .collect();
                Ok(IType(TheoryIType::Float(ArithFloat::Const(data))))
            }
            Kind::Int | Kind::Int8 | Kind::Int16 | Kind::Int64 | Kind::Uint8 => {
                let data: Vec<Vec<i64>> = (0..rows)
                    .map(|i| {
                        (0..cols)
                            .map(|j| t.int64_value(&[i as i64, j as i64]))
                            .collect()
                    })
                    .collect();
                Ok(IType(TheoryIType::Int(ArithInt::Const(data))))
            }
            Kind::Bool => {
                let t_int = t.to_kind(Kind::Int64);
                let data: Vec<Vec<bool>> = (0..rows)
                    .map(|i| {
                        (0..cols)
                            .map(|j| t_int.int64_value(&[i as i64, j as i64]) != 0)
                            .collect()
                    })
                    .collect();
                Ok(IType(TheoryIType::Bool(BoolOp::Const(data))))
            }
            kind => Err(PyValueError::new_err(format!(
                "IType.from_tensor: unsupported kind {kind:?}"
            ))),
        }
    }

    fn __eq__(&self, other: &IType) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        format!("{:?}", self.0).hash(&mut h);
        h.finish()
    }

    #[getter]
    fn is_const(&self) -> bool {
        matches!(
            &self.0,
            TheoryIType::Float(ArithFloat::Const(_))
                | TheoryIType::Int(ArithInt::Const(_))
                | TheoryIType::Bool(BoolOp::Const(_))
                | TheoryIType::Real(ArithReal::Const(_))
        )
    }

    #[getter]
    fn const_data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.0 {
            TheoryIType::Float(ArithFloat::Const(data)) => {
                Ok(data.clone().into_pyobject(py)?.into_any())
            }
            TheoryIType::Int(ArithInt::Const(data)) => {
                Ok(data.clone().into_pyobject(py)?.into_any())
            }
            TheoryIType::Bool(BoolOp::Const(data)) => {
                Ok(data.clone().into_pyobject(py)?.into_any())
            }
            TheoryIType::Real(ArithReal::Const(data)) => {
                Ok(data.clone().into_pyobject(py)?.into_any())
            }
            _ => Err(PyValueError::new_err("not a Const op")),
        }
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
    #[classattr]
    fn And() -> IType {
        IType(TheoryIType::Bool(BoolOp::And))
    }
    #[classattr]
    fn Or() -> IType {
        IType(TheoryIType::Bool(BoolOp::Or))
    }
    #[classattr]
    fn Xor() -> IType {
        IType(TheoryIType::Bool(BoolOp::Xor))
    }
    #[classattr]
    fn Not() -> IType {
        IType(TheoryIType::Bool(BoolOp::Not))
    }
    #[classattr]
    fn Xnor() -> IType {
        IType(TheoryIType::Bool(BoolOp::Xnor))
    }
    #[classattr]
    fn Implies() -> IType {
        IType(TheoryIType::Bool(BoolOp::Implies))
    }
    #[staticmethod]
    fn Const(data: Vec<Vec<bool>>) -> IType {
        IType(TheoryIType::Bool(BoolOp::Const(data)))
    }
}

// IType.Int namespace

#[pyclass]
pub struct IntIType;

#[pymethods]
impl IntIType {
    #[classattr]
    fn Add() -> IType { IType(TheoryIType::Int(Arith::Add.into())) }
    #[classattr]
    fn Sub() -> IType { IType(TheoryIType::Int(Arith::Sub.into())) }
    #[classattr]
    fn Mul() -> IType { IType(TheoryIType::Int(Arith::Mul.into())) }
    #[classattr]
    fn Div() -> IType { IType(TheoryIType::Int(Arith::Div.into())) }
    #[classattr]
    fn Mod() -> IType { IType(TheoryIType::Int(Arith::Mod.into())) }
    #[classattr]
    fn Neg() -> IType { IType(TheoryIType::Int(Arith::Neg.into())) }
    #[classattr]
    fn Abs() -> IType { IType(TheoryIType::Int(Arith::Abs.into())) }
    #[classattr]
    fn MatMul() -> IType { IType(TheoryIType::Int(Arith::MatMul.into())) }
    #[classattr]
    fn Transpose() -> IType { IType(TheoryIType::Int(Arith::Transpose.into())) }
    #[staticmethod]
    fn Const(data: Vec<Vec<i64>>) -> IType {
        IType(TheoryIType::Int(ArithInt::Const(data)))
    }
}

// IType.Float namespace

#[pyclass]
pub struct FloatIType;

#[pymethods]
impl FloatIType {
    #[classattr]
    fn Add() -> IType { IType(TheoryIType::Float(Arith::Add.into())) }
    #[classattr]
    fn Sub() -> IType { IType(TheoryIType::Float(Arith::Sub.into())) }
    #[classattr]
    fn Mul() -> IType { IType(TheoryIType::Float(Arith::Mul.into())) }
    #[classattr]
    fn Div() -> IType { IType(TheoryIType::Float(Arith::Div.into())) }
    #[classattr]
    fn Mod() -> IType { IType(TheoryIType::Float(Arith::Mod.into())) }
    #[classattr]
    fn Neg() -> IType { IType(TheoryIType::Float(Arith::Neg.into())) }
    #[classattr]
    fn Abs() -> IType { IType(TheoryIType::Float(Arith::Abs.into())) }
    #[classattr]
    fn MatMul() -> IType { IType(TheoryIType::Float(Arith::MatMul.into())) }
    #[classattr]
    fn Transpose() -> IType { IType(TheoryIType::Float(Arith::Transpose.into())) }
    #[staticmethod]
    fn Const(data: Vec<Vec<f64>>) -> IType {
        IType(TheoryIType::Float(ArithFloat::Const(data)))
    }
}

// IType.Real namespace

#[pyclass]
pub struct RealIType;

#[pymethods]
impl RealIType {
    #[classattr]
    fn Add() -> IType { IType(TheoryIType::Real(Arith::Add.into())) }
    #[classattr]
    fn Sub() -> IType { IType(TheoryIType::Real(Arith::Sub.into())) }
    #[classattr]
    fn Mul() -> IType { IType(TheoryIType::Real(Arith::Mul.into())) }
    #[classattr]
    fn Div() -> IType { IType(TheoryIType::Real(Arith::Div.into())) }
    #[classattr]
    fn Mod() -> IType { IType(TheoryIType::Real(Arith::Mod.into())) }
    #[classattr]
    fn Neg() -> IType { IType(TheoryIType::Real(Arith::Neg.into())) }
    #[classattr]
    fn Abs() -> IType { IType(TheoryIType::Real(Arith::Abs.into())) }
    #[classattr]
    fn MatMul() -> IType { IType(TheoryIType::Real(Arith::MatMul.into())) }
    #[classattr]
    fn Transpose() -> IType { IType(TheoryIType::Real(Arith::Transpose.into())) }
    #[classattr]
    fn Sin() -> IType { IType(TheoryIType::Real(ArithReal::Sin)) }
    #[classattr]
    fn Cos() -> IType { IType(TheoryIType::Real(ArithReal::Cos)) }
    #[staticmethod]
    fn Const(data: Vec<Vec<f64>>) -> IType {
        IType(TheoryIType::Real(ArithReal::Const(data)))
    }
}

// IType.Cmp namespace

#[pyclass]
pub struct CmpIType;

#[pymethods]
impl CmpIType {
    #[classattr]
    fn Le() -> IType {
        IType(TheoryIType::Cmp(CmpOp::Le))
    }
    #[classattr]
    fn Lt() -> IType {
        IType(TheoryIType::Cmp(CmpOp::Lt))
    }
    #[classattr]
    fn Ge() -> IType {
        IType(TheoryIType::Cmp(CmpOp::Ge))
    }
    #[classattr]
    fn Gt() -> IType {
        IType(TheoryIType::Cmp(CmpOp::Gt))
    }
    #[classattr]
    fn Eq() -> IType {
        IType(TheoryIType::Cmp(CmpOp::Eq))
    }
    #[classattr]
    fn Ne() -> IType {
        IType(TheoryIType::Cmp(CmpOp::Ne))
    }
}

// IType.NN namespace

#[pyclass]
pub struct NNIType;

#[pymethods]
impl NNIType {
    #[classattr]
    fn ReLU() -> IType {
        IType(TheoryIType::NN(NNOp::ReLU))
    }
    #[classattr]
    fn Tanh() -> IType {
        IType(TheoryIType::NN(NNOp::Tanh))
    }
    #[classattr]
    fn Linear() -> IType {
        IType(TheoryIType::NN(NNOp::Linear))
    }
}

// IType.Tensor namespace

#[pyclass]
pub struct TensorIType;

#[pymethods]
impl TensorIType {
    #[classattr]
    fn Get() -> IType {
        IType(TheoryIType::Tensor(TensorOp::Get))
    }
    #[classattr]
    fn Set() -> IType {
        IType(TheoryIType::Tensor(TensorOp::Set))
    }
    #[classattr]
    fn Sum() -> IType {
        IType(TheoryIType::Tensor(TensorOp::Sum))
    }
    #[classattr]
    fn Mean() -> IType {
        IType(TheoryIType::Tensor(TensorOp::Mean))
    }
    #[classattr]
    fn Max() -> IType {
        IType(TheoryIType::Tensor(TensorOp::Max))
    }
    #[classattr]
    fn Argmax() -> IType {
        IType(TheoryIType::Tensor(TensorOp::Argmax))
    }
}

// IType.Flow namespace

#[pyclass]
pub struct FlowIType;

#[pymethods]
impl FlowIType {
    #[classattr]
    fn Ite() -> IType {
        IType(TheoryIType::Flow(FlowOp::Ite))
    }
    #[classattr]
    fn Id() -> IType {
        IType(TheoryIType::Flow(FlowOp::Id))
    }
}

// IType.BV namespace

#[pyclass]
pub struct BVIType;

#[pymethods]
impl BVIType {
    #[classattr]
    fn Add() -> IType { IType(TheoryIType::BV(Arith::Add.into())) }
    #[classattr]
    fn Sub() -> IType { IType(TheoryIType::BV(Arith::Sub.into())) }
    #[classattr]
    fn Mul() -> IType { IType(TheoryIType::BV(Arith::Mul.into())) }
    #[classattr]
    fn Div() -> IType { IType(TheoryIType::BV(Arith::Div.into())) }
    #[classattr]
    fn Mod() -> IType { IType(TheoryIType::BV(Arith::Mod.into())) }
    #[classattr]
    fn MatMul() -> IType { IType(TheoryIType::BV(Arith::MatMul.into())) }
    #[classattr]
    fn Transpose() -> IType { IType(TheoryIType::BV(Arith::Transpose.into())) }
    #[classattr]
    fn And() -> IType { IType(TheoryIType::BV(BVTheory::And)) }
    #[classattr]
    fn Or() -> IType { IType(TheoryIType::BV(BVTheory::Or)) }
    #[classattr]
    fn Xor() -> IType { IType(TheoryIType::BV(BVTheory::Xor)) }
    #[classattr]
    fn Not() -> IType { IType(TheoryIType::BV(BVTheory::Not)) }
    #[staticmethod]
    fn Const(data: Vec<Vec<usize>>) -> IType {
        IType(TheoryIType::BV(BVTheory::Const(data)))
    }
}
