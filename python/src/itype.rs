#![allow(non_snake_case)]

use pyo3::prelude::*;

use crate::pytensor::PyTensor;
use crate::types::IType;

fn scalar_tensor_bool(v: bool) -> tch::Tensor {
    tch::Tensor::from_slice(&[v as u8])
        .to_kind(tch::Kind::Bool)
        .reshape([1, 1])
}

fn scalar_tensor_int(v: i64) -> tch::Tensor {
    tch::Tensor::from_slice(&[v]).reshape([1, 1])
}

fn coerce_to_tensor(value: &Bound<'_, PyAny>) -> PyResult<tch::Tensor> {
    // Try torch.Tensor first — a torch.Tensor *also* satisfies `extract::<bool>`
    // via its `__bool__` method for 1-element tensors, which would silently
    // discard data; only fall back to scalar coercion when the input is not a
    // tensor.
    if let Ok(t) = value.extract::<PyTensor>() {
        let t = if t.dim() <= 1 {
            t.tensor.reshape([1, -1])
        } else {
            t.tensor.shallow_clone()
        };
        return Ok(t);
    }
    if let Ok(b) = value.extract::<bool>() {
        return Ok(scalar_tensor_bool(b));
    }
    if let Ok(i) = value.extract::<i64>() {
        return Ok(scalar_tensor_int(i));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected bool, int, or torch.Tensor",
    ))
}

// ============================================================================
// IType.LIA namespace
// ============================================================================

#[pyclass]
pub struct LIAIType;

#[pymethods]
impl LIAIType {
    #[classattr]
    fn And() -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::And))
    }
    #[classattr]
    fn Or() -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::Or))
    }
    #[classattr]
    fn Not() -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::Not))
    }
    #[classattr]
    fn Xor() -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::Xor))
    }
    #[classattr]
    fn Le() -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::Le))
    }
    #[classattr]
    fn Lt() -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::Lt))
    }
    #[classattr]
    fn Ge() -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::Ge))
    }
    #[classattr]
    fn Gt() -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::Gt))
    }
    #[classattr]
    fn Eq() -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::Eq))
    }
    #[classattr]
    fn Ne() -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::Ne))
    }
    #[classattr]
    fn Add() -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::Add))
    }
    #[classattr]
    fn ReLU() -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::ReLU))
    }
    #[classattr]
    fn Argmax() -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::Argmax))
    }
    #[classattr]
    fn Min() -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::Min))
    }
    #[classattr]
    fn Max() -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::Max))
    }
    #[classattr]
    fn Ite() -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::Ite))
    }
    #[classattr]
    fn Id() -> IType {
        IType(theory::any::Any::LIA(theory::lia::LIA::Id))
    }

    #[staticmethod]
    fn ConstBool(value: &Bound<'_, PyAny>) -> PyResult<IType> {
        let t = coerce_to_tensor(value)?;
        Ok(IType(theory::any::Any::LIA(theory::lia::LIA::ConstBool(
            theory::Tensor(t),
        ))))
    }

    #[staticmethod]
    fn ConstInt(value: &Bound<'_, PyAny>) -> PyResult<IType> {
        let t = coerce_to_tensor(value)?;
        Ok(IType(theory::any::Any::LIA(theory::lia::LIA::ConstInt(
            theory::Tensor(t),
        ))))
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
    fn And() -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::And))
    }
    #[classattr]
    fn Or() -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::Or))
    }
    #[classattr]
    fn Not() -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::Not))
    }
    #[classattr]
    fn Xor() -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::Xor))
    }
    #[classattr]
    fn Le() -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::Le))
    }
    #[classattr]
    fn Lt() -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::Lt))
    }
    #[classattr]
    fn Ge() -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::Ge))
    }
    #[classattr]
    fn Gt() -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::Gt))
    }
    #[classattr]
    fn Eq() -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::Eq))
    }
    #[classattr]
    fn Ne() -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::Ne))
    }
    #[classattr]
    fn Add() -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::Add))
    }
    #[classattr]
    fn ReLU() -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::ReLU))
    }
    #[classattr]
    fn Argmax() -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::Argmax))
    }
    #[classattr]
    fn Min() -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::Min))
    }
    #[classattr]
    fn Max() -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::Max))
    }
    #[classattr]
    fn Ite() -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::Ite))
    }
    #[classattr]
    fn Id() -> IType {
        IType(theory::any::Any::LRA(theory::lra::LRA::Id))
    }

    #[staticmethod]
    fn ConstBool(value: &Bound<'_, PyAny>) -> PyResult<IType> {
        let t = coerce_to_tensor(value)?;
        Ok(IType(theory::any::Any::LRA(theory::lra::LRA::ConstBool(
            theory::Tensor(t),
        ))))
    }

    #[staticmethod]
    fn ConstReal(value: &Bound<'_, PyAny>) -> PyResult<IType> {
        let t = coerce_to_tensor(value)?;
        Ok(IType(theory::any::Any::LRA(theory::lra::LRA::ConstReal(
            theory::Tensor(t),
        ))))
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
    fn And() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::And))
    }
    #[classattr]
    fn Or() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Or))
    }
    #[classattr]
    fn Not() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Not))
    }
    #[classattr]
    fn Xor() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Xor))
    }
    #[classattr]
    fn Le() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Le))
    }
    #[classattr]
    fn Lt() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Lt))
    }
    #[classattr]
    fn Ge() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Ge))
    }
    #[classattr]
    fn Gt() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Gt))
    }
    #[classattr]
    fn Eq() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Eq))
    }
    #[classattr]
    fn Ne() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Ne))
    }
    #[classattr]
    fn Add() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Add))
    }
    #[classattr]
    fn Sub() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Sub))
    }
    #[classattr]
    fn Neg() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Neg))
    }
    #[classattr]
    fn Abs() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Abs))
    }
    #[classattr]
    fn Mul() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Mul))
    }
    #[classattr]
    fn UDiv() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::UDiv))
    }
    #[classattr]
    fn SDiv() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::SDiv))
    }
    #[classattr]
    fn UMod() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::UMod))
    }
    #[classattr]
    fn SMod() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::SMod))
    }
    #[classattr]
    fn MatMul() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::MatMul))
    }
    #[classattr]
    fn Ite() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Ite))
    }
    #[classattr]
    fn Id() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Id))
    }
    /// SMV `bool(word)`: BV<n> → BV<1> via `x != 0`.
    /// (SMV's dual `word1(bool)` is just `Id` on a BV<1>, so no separate op.)
    #[classattr]
    fn BVToBool() -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::BVToBool))
    }

    #[staticmethod]
    fn Const(value: &Bound<'_, PyAny>) -> PyResult<IType> {
        let t = coerce_to_tensor(value)?;
        Ok(IType(theory::any::Any::BV(theory::bv::BV::Const(
            theory::Tensor(t),
        ))))
    }

    /// `BitSelect(high, low)`: select bits `[high..=low]` from a BV input.
    #[staticmethod]
    fn BitSelect(high: usize, low: usize) -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::BitSelect {
            high,
            low,
        }))
    }

    /// `Extend(extra)`: zero-extend a BV input by `extra` bits.
    #[staticmethod]
    fn Extend(extra: usize) -> IType {
        IType(theory::any::Any::BV(theory::bv::BV::Extend { extra }))
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

    fn __eq__(&self, other: &IType) -> bool {
        format!("{:?}", self.0) == format!("{:?}", other.0)
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        format!("{:?}", self.0).hash(&mut h);
        h.finish()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.0)
    }

    /// Allow `IType.X()` to behave the same as `IType.X` (the façade in
    /// `zrth.__init__` resolves `IType.X` to a Rust IType instance; older code
    /// kept the call-syntax habit).
    fn __call__(slf: PyRef<'_, IType>) -> IType {
        slf.clone()
    }

    /// Name of the underlying operation, stripped of any arguments / theory
    /// prefix. E.g. `IType.LIA.Add.op_name == "Add"`, both
    /// `LIA.ConstBool(...)` and `LRA.ConstBool(...)` give `"ConstBool"`.
    #[getter]
    fn op_name(&self) -> String {
        op_name_of(&self.0).to_string()
    }

    /// Name of the theory: "LIA", "LRA", or "BV".
    #[getter]
    fn theory_name(&self) -> &'static str {
        match &self.0 {
            theory::any::Any::LIA(_) => "LIA",
            theory::any::Any::LRA(_) => "LRA",
            theory::any::Any::BV(_) => "BV",
        }
    }

    /// For `Const*` ops, the constant's tensor payload as a `torch.Tensor`.
    /// Errors for ops that don't carry a constant tensor.
    #[getter]
    fn const_data(&self) -> PyResult<PyTensor> {
        use theory::any::Any;
        use theory::bv::BV;
        use theory::lia::LIA;
        use theory::lra::LRA;
        let t = match &self.0 {
            Any::LIA(LIA::ConstInt(t) | LIA::ConstBool(t)) => &t.0,
            Any::LRA(LRA::ConstReal(t) | LRA::ConstBool(t)) => &t.0,
            Any::BV(BV::Const(t)) => &t.0,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("not a Const op")),
        };
        Ok(PyTensor {
            tensor: t.shallow_clone(),
        })
    }

    /// For `Uninterpreted(name)` ops, the symbol name. Errors otherwise.
    #[getter]
    fn name(&self) -> PyResult<String> {
        use theory::any::Any;
        use theory::bv::BV;
        use theory::lia::LIA;
        use theory::lra::LRA;
        match &self.0 {
            Any::LIA(LIA::Uninterpreted(s))
            | Any::LRA(LRA::Uninterpreted(s))
            | Any::BV(BV::Uninterpreted(s)) => Ok(s.clone()),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "not an Uninterpreted op",
            )),
        }
    }
}

fn op_name_of(a: &theory::any::Any) -> &'static str {
    use theory::any::Any;
    use theory::bv::BV;
    use theory::lia::LIA;
    use theory::lra::LRA;
    match a {
        Any::LIA(op) => match op {
            LIA::ConstInt(_) => "ConstInt",
            LIA::ConstBool(_) => "ConstBool",
            LIA::And => "And",
            LIA::Or => "Or",
            LIA::Xor => "Xor",
            LIA::Not => "Not",
            LIA::Le => "Le",
            LIA::Lt => "Lt",
            LIA::Ge => "Ge",
            LIA::Gt => "Gt",
            LIA::Eq => "Eq",
            LIA::Ne => "Ne",
            LIA::Linear(_, _) => "Linear",
            LIA::Add => "Add",
            LIA::ReLU => "ReLU",
            LIA::Argmax => "Argmax",
            LIA::Min => "Min",
            LIA::Max => "Max",
            LIA::Ite => "Ite",
            LIA::Id => "Id",
            LIA::Uninterpreted(_) => "Uninterpreted",
        },
        Any::LRA(op) => match op {
            LRA::ConstReal(_) => "ConstReal",
            LRA::ConstBool(_) => "ConstBool",
            LRA::And => "And",
            LRA::Or => "Or",
            LRA::Xor => "Xor",
            LRA::Not => "Not",
            LRA::Le => "Le",
            LRA::Lt => "Lt",
            LRA::Ge => "Ge",
            LRA::Gt => "Gt",
            LRA::Eq => "Eq",
            LRA::Ne => "Ne",
            LRA::Linear(_, _) => "Linear",
            LRA::Add => "Add",
            LRA::ReLU => "ReLU",
            LRA::Argmax => "Argmax",
            LRA::Min => "Min",
            LRA::Max => "Max",
            LRA::Ite => "Ite",
            LRA::Id => "Id",
            LRA::Uninterpreted(_) => "Uninterpreted",
        },
        Any::BV(op) => match op {
            BV::Const(_) => "Const",
            BV::Add => "Add",
            BV::Sub => "Sub",
            BV::Neg => "Neg",
            BV::Abs => "Abs",
            BV::Mul => "Mul",
            BV::UDiv => "UDiv",
            BV::SDiv => "SDiv",
            BV::UMod => "UMod",
            BV::SMod => "SMod",
            BV::MatMul => "MatMul",
            BV::And => "And",
            BV::Or => "Or",
            BV::Xor => "Xor",
            BV::Not => "Not",
            BV::Le => "Le",
            BV::Lt => "Lt",
            BV::Ge => "Ge",
            BV::Gt => "Gt",
            BV::Eq => "Eq",
            BV::Ne => "Ne",
            BV::Ite => "Ite",
            BV::Id => "Id",
            BV::BVToBool => "BVToBool",
            BV::BitSelect { .. } => "BitSelect",
            BV::Extend { .. } => "Extend",
            BV::Uninterpreted(_) => "Uninterpreted",
        },
    }
}
