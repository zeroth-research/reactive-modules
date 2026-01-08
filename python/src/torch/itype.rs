use crate::torch::pytensor::PyTensor;
use pyo3::prelude::*;
use std::fmt;

#[pyclass]
#[derive(Debug, Clone)]
pub struct IType(torch::IType);

impl fmt::Display for IType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

#[pymethods]
impl IType {
    #[staticmethod]
    fn mk_const_tensor(val: PyTensor) -> Self {
        IType(torch::IType::ConstTensor(val.tensor))
    }

    #[staticmethod]
    fn mk_const_bool(val: bool) -> Self {
        IType(torch::IType::ConstBool(val))
    }

    #[staticmethod]
    fn mk_eq() -> Self {
        IType(torch::IType::Eq)
    }

    // Neq,
    // Lt,

    #[staticmethod]
    fn mk_lt() -> Self {
        IType(torch::IType::Lt)
    }

    // Le,
    // Gt,
    // Ge,
    // // product and sum of elements of the tensor
    // Prod,
    // Sum,

    #[staticmethod]
    fn mk_add() -> Self {
        IType(torch::IType::Add)
    }

    // Mul,
    // Sub,
    // Div,

    #[staticmethod]
    fn mk_matmul() -> Self {
        IType(torch::IType::MatMul)
    }

    #[staticmethod]
    fn mk_id() -> Self {
        IType(torch::IType::Id)
    }

    #[staticmethod]
    fn mk_ite() -> Self {
        IType(torch::IType::Ite)
    }

    #[staticmethod]
    fn mk_ifthen() -> Self {
        IType(torch::IType::IfThen)
    }

    #[staticmethod]
    fn mk_choose() -> Self {
        IType(torch::IType::Choose)
    }

    #[staticmethod]
    fn mk_choose_or() -> Self {
        IType(torch::IType::ChooseOr)
    }

    // boolean operations
    #[staticmethod]
    fn mk_not() -> Self {
        IType(torch::IType::Not)
    }

    #[staticmethod]
    fn mk_or() -> Self {
        IType(torch::IType::Or)
    }

    #[staticmethod]
    fn mk_and() -> Self {
        IType(torch::IType::And)
    }
}

unsafe impl Sync for IType {}
