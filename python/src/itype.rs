#![allow(non_snake_case)]

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

use crate::pytensor::PyTensor;
use crate::types::IType;

// ============================================================================
// IType.LIA namespace
// ============================================================================

#[pyclass]
pub struct LIAIType;

#[pymethods]
impl LIAIType {
    #[classattr]
    fn And() -> IType { IType(theory::any::Any::LIA(theory::lia::LIA::And)) }
    #[classattr]
    fn Or() -> IType { IType(theory::any::Any::LIA(theory::lia::LIA::Or)) }
    #[classattr]
    fn Not() -> IType { IType(theory::any::Any::LIA(theory::lia::LIA::Not)) }
    #[classattr]
    fn Xor() -> IType { IType(theory::any::Any::LIA(theory::lia::LIA::Xor)) }
    #[classattr]
    fn Le() -> IType { IType(theory::any::Any::LIA(theory::lia::LIA::Le)) }
    #[classattr]
    fn Lt() -> IType { IType(theory::any::Any::LIA(theory::lia::LIA::Lt)) }
    #[classattr]
    fn Ge() -> IType { IType(theory::any::Any::LIA(theory::lia::LIA::Ge)) }
    #[classattr]
    fn Gt() -> IType { IType(theory::any::Any::LIA(theory::lia::LIA::Gt)) }
    #[classattr]
    fn Eq() -> IType { IType(theory::any::Any::LIA(theory::lia::LIA::Eq)) }
    #[classattr]
    fn Ne() -> IType { IType(theory::any::Any::LIA(theory::lia::LIA::Ne)) }
    #[classattr]
    fn Add() -> IType { IType(theory::any::Any::LIA(theory::lia::LIA::Add)) }
    #[classattr]
    fn ReLU() -> IType { IType(theory::any::Any::LIA(theory::lia::LIA::ReLU)) }
    #[classattr]
    fn Argmax() -> IType { IType(theory::any::Any::LIA(theory::lia::LIA::Argmax)) }
    #[classattr]
    fn Min() -> IType { IType(theory::any::Any::LIA(theory::lia::LIA::Min)) }
    #[classattr]
    fn Max() -> IType { IType(theory::any::Any::LIA(theory::lia::LIA::Max)) }
    #[classattr]
    fn Ite() -> IType { IType(theory::any::Any::LIA(theory::lia::LIA::Ite)) }
    #[classattr]
    fn Id() -> IType { IType(theory::any::Any::LIA(theory::lia::LIA::Id)) }

    #[staticmethod]
    fn ConstBool(t: PyTensor) -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::ConstBool(theory::Tensor(t.tensor))))
    }

    #[staticmethod]
    fn ConstInt(t: PyTensor) -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::ConstInt(theory::Tensor(t.tensor))))
    }

    #[staticmethod]
    fn Uninterpreted(name: String) -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::Uninterpreted(name)))
    }
}

// ============================================================================
// IType.LRA namespace
// ============================================================================

#[pyclass]
pub struct LRAIType;

#[pymethods]
impl LRAIType {
    #[classattr]
    fn And() -> IType { IType(theory::any::Any::LRA(theory::lra::LRA::And)) }
    #[classattr]
    fn Or() -> IType { IType(theory::any::Any::LRA(theory::lra::LRA::Or)) }
    #[classattr]
    fn Not() -> IType { IType(theory::any::Any::LRA(theory::lra::LRA::Not)) }
    #[classattr]
    fn Xor() -> IType { IType(theory::any::Any::LRA(theory::lra::LRA::Xor)) }
    #[classattr]
    fn Le() -> IType { IType(theory::any::Any::LRA(theory::lra::LRA::Le)) }
    #[classattr]
    fn Lt() -> IType { IType(theory::any::Any::LRA(theory::lra::LRA::Lt)) }
    #[classattr]
    fn Ge() -> IType { IType(theory::any::Any::LRA(theory::lra::LRA::Ge)) }
    #[classattr]
    fn Gt() -> IType { IType(theory::any::Any::LRA(theory::lra::LRA::Gt)) }
    #[classattr]
    fn Eq() -> IType { IType(theory::any::Any::LRA(theory::lra::LRA::Eq)) }
    #[classattr]
    fn Ne() -> IType { IType(theory::any::Any::LRA(theory::lra::LRA::Ne)) }
    #[classattr]
    fn Add() -> IType { IType(theory::any::Any::LRA(theory::lra::LRA::Add)) }
    #[classattr]
    fn ReLU() -> IType { IType(theory::any::Any::LRA(theory::lra::LRA::ReLU)) }
    #[classattr]
    fn Argmax() -> IType { IType(theory::any::Any::LRA(theory::lra::LRA::Argmax)) }
    #[classattr]
    fn Min() -> IType { IType(theory::any::Any::LRA(theory::lra::LRA::Min)) }
    #[classattr]
    fn Max() -> IType { IType(theory::any::Any::LRA(theory::lra::LRA::Max)) }
    #[classattr]
    fn Ite() -> IType { IType(theory::any::Any::LRA(theory::lra::LRA::Ite)) }
    #[classattr]
    fn Id() -> IType { IType(theory::any::Any::LRA(theory::lra::LRA::Id)) }

    #[staticmethod]
    fn ConstBool(t: PyTensor) -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::ConstBool(theory::Tensor(t.tensor))))
    }

    #[staticmethod]
    fn ConstReal(t: PyTensor) -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::ConstReal(theory::Tensor(t.tensor))))
    }

    #[staticmethod]
    fn Uninterpreted(name: String) -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::Uninterpreted(name)))
    }
}

// ============================================================================
// IType.BV namespace
// ============================================================================

#[pyclass]
pub struct BVIType;

#[pymethods]
impl BVIType {
    #[classattr]
    fn And() -> IType { IType(theory::any::Any::BV(theory::bv::BV::And)) }
    #[classattr]
    fn Or() -> IType { IType(theory::any::Any::BV(theory::bv::BV::Or)) }
    #[classattr]
    fn Not() -> IType { IType(theory::any::Any::BV(theory::bv::BV::Not)) }
    #[classattr]
    fn Xor() -> IType { IType(theory::any::Any::BV(theory::bv::BV::Xor)) }
    #[classattr]
    fn Le() -> IType { IType(theory::any::Any::BV(theory::bv::BV::Le)) }
    #[classattr]
    fn Lt() -> IType { IType(theory::any::Any::BV(theory::bv::BV::Lt)) }
    #[classattr]
    fn Ge() -> IType { IType(theory::any::Any::BV(theory::bv::BV::Ge)) }
    #[classattr]
    fn Gt() -> IType { IType(theory::any::Any::BV(theory::bv::BV::Gt)) }
    #[classattr]
    fn Eq() -> IType { IType(theory::any::Any::BV(theory::bv::BV::Eq)) }
    #[classattr]
    fn Ne() -> IType { IType(theory::any::Any::BV(theory::bv::BV::Ne)) }
    #[classattr]
    fn Add() -> IType { IType(theory::any::Any::BV(theory::bv::BV::Add)) }
    #[classattr]
    fn Mul() -> IType { IType(theory::any::Any::BV(theory::bv::BV::Mul)) }
    #[classattr]
    fn UDiv() -> IType { IType(theory::any::Any::BV(theory::bv::BV::UDiv)) }
    #[classattr]
    fn SDiv() -> IType { IType(theory::any::Any::BV(theory::bv::BV::SDiv)) }
    #[classattr]
    fn MatMul() -> IType { IType(theory::any::Any::BV(theory::bv::BV::MatMul)) }
    #[classattr]
    fn Ite() -> IType { IType(theory::any::Any::BV(theory::bv::BV::Ite)) }
    #[classattr]
    fn Id() -> IType { IType(theory::any::Any::BV(theory::bv::BV::Id)) }

    #[staticmethod]
    fn Const(t: PyTensor) -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Const(theory::Tensor(t.tensor))))
    }

    #[staticmethod]
    fn Uninterpreted(name: String) -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Uninterpreted(name)))
    }
}

// ============================================================================
// Top-level IType methods: namespaces + cross-theory shortcuts + unimplemented
// ============================================================================

#[pymethods]
impl IType {
    // Namespace class-attributes
    #[classattr]
    fn LIA(py: Python<'_>) -> Py<PyAny> {
        py.get_type::<LIAIType>().into_any().unbind()
    }
    #[classattr]
    fn LRA(py: Python<'_>) -> Py<PyAny> {
        py.get_type::<LRAIType>().into_any().unbind()
    }
    #[classattr]
    fn BV(py: Python<'_>) -> Py<PyAny> {
        py.get_type::<BVIType>().into_any().unbind()
    }

    // Convenience shortcuts used in tests (map to LIA as the bool/int theory)
    #[classattr]
    pub fn And() -> Self { LIAIType::And() }
    #[classattr]
    pub fn Or() -> Self { LIAIType::Or() }
    #[classattr]
    pub fn Not() -> Self { LIAIType::Not() }
    #[classattr]
    pub fn Xor() -> Self { LIAIType::Xor() }
    #[classattr]
    pub fn Id() -> Self { LIAIType::Id() }
    #[classattr]
    pub fn Ite() -> Self { LIAIType::Ite() }
    #[classattr]
    pub fn Add() -> Self { LIAIType::Add() }
    #[classattr]
    pub fn Argmax() -> Self { LIAIType::Argmax() }
    #[classattr]
    pub fn Mul() -> Self { BVIType::Mul() }
    #[classattr]
    pub fn MatMul() -> Self { BVIType::MatMul() }

    #[staticmethod]
    pub fn ConstBool(t: PyTensor) -> Self { LIAIType::ConstBool(t) }
    #[staticmethod]
    pub fn ConstInt(t: PyTensor) -> Self { LIAIType::ConstInt(t) }

    // Not yet implemented
    #[staticmethod]
    pub fn Sub() -> PyResult<Self> {
        Err(PyNotImplementedError::new_err("Sub is not yet implemented"))
    }
    #[staticmethod]
    pub fn Tensor(_t: &Bound<'_, PyAny>) -> PyResult<Self> {
        Err(PyNotImplementedError::new_err("Tensor is not yet implemented"))
    }
    #[staticmethod]
    pub fn TensorSum() -> PyResult<Self> {
        Err(PyNotImplementedError::new_err("TensorSum is not yet implemented"))
    }
}
