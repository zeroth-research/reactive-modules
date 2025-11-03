use crate::dtype::Type;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Val {
    None, // no value
    Real(f64),
    NReal(f64),
    Int(i64),
    NInt(u64),
    Bool(bool),
}

impl Val {
    pub fn new() -> Self {
        Val::None
    }

    // TODO: implement Add trait
    // TODO: should we check overflows?
    pub(crate) fn add(&self, rhs: &Val) -> Option<Self> {
        match (self, rhs) {
            (Val::Real(x), Val::Real(y)) => Some(Val::Real(x + y)),
            (Val::NReal(x), Val::NReal(y)) => Some(Val::NReal(x + y)),
            (Val::Int(x), Val::Int(y)) => Some(Val::Int(x + y)),
            (Val::NInt(x), Val::NInt(y)) => Some(Val::NInt(x + y)),
            _ => None,
        }
    }

    pub(crate) fn sub(&self, rhs: &Val) -> Option<Self> {
        match (self, rhs) {
            (Val::Real(x), Val::Real(y)) => Some(Val::Real(x - y)),
            (Val::NReal(x), Val::NReal(y)) => Some(Val::NReal(x - y)),
            (Val::Int(x), Val::Int(y)) => Some(Val::Int(x - y)),
            (Val::NInt(x), Val::NInt(y)) => Some(Val::NInt(x - y)),
            _ => None,
        }
    }

    pub(crate) fn mul(&self, rhs: &Val) -> Option<Self> {
        match (self, rhs) {
            (Val::Real(x), Val::Real(y)) => Some(Val::Real(x * y)),
            (Val::NReal(x), Val::NReal(y)) => Some(Val::NReal(x * y)),
            (Val::Int(x), Val::Int(y)) => Some(Val::Int(x * y)),
            (Val::NInt(x), Val::NInt(y)) => Some(Val::NInt(x * y)),
            _ => None,
        }
    }

    pub(crate) fn div(&self, rhs: &Val) -> Option<Self> {
        match (self, rhs) {
            (Val::Real(x), Val::Real(y)) => Some(Val::Real(x / y)),
            (Val::NReal(x), Val::NReal(y)) => Some(Val::NReal(x / y)),
            (Val::Int(x), Val::Int(y)) => Some(Val::Int(x / y)),
            (Val::NInt(x), Val::NInt(y)) => Some(Val::NInt(x / y)),
            _ => None,
        }
    }
}

impl Val {
    pub fn has_type(&self, ty: &Type) -> bool {
        match self {
            Val::None => false,
            Val::Real(_) => *ty == Type::Real,
            Val::NReal(_) => *ty == Type::NReal,
            Val::Int(_) => *ty == Type::Int,
            Val::NInt(_) => *ty == Type::NInt,
            Val::Bool(_) => *ty == Type::Bool,
        }
    }

    pub fn same_type(&self, rhs: &Val) -> bool {
        match self {
            Val::None => matches!(rhs, Val::None),
            Val::Real(_) => matches!(rhs, Val::Real { .. }),
            Val::NReal(_) => matches!(rhs, Val::NReal { .. }),
            Val::Int(_) => matches!(rhs, Val::Int { .. }),
            Val::NInt(_) => matches!(rhs, Val::NInt { .. }),
            Val::Bool(_) => matches!(rhs, Val::Bool { .. }),
        }
    }
}
