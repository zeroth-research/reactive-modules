use crate::dtype::Type;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Val {
    None, // no value
    Real(f64),
    Int(i64),
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
            (Val::Int(x), Val::Int(y)) => Some(Val::Int(x + y)),
            _ => None,
        }
    }

    pub(crate) fn sub(&self, rhs: &Val) -> Option<Self> {
        match (self, rhs) {
            (Val::Real(x), Val::Real(y)) => Some(Val::Real(x - y)),
            (Val::Int(x), Val::Int(y)) => Some(Val::Int(x - y)),
            _ => None,
        }
    }

    pub(crate) fn mul(&self, rhs: &Val) -> Option<Self> {
        match (self, rhs) {
            (Val::Real(x), Val::Real(y)) => Some(Val::Real(x * y)),
            (Val::Int(x), Val::Int(y)) => Some(Val::Int(x * y)),
            _ => None,
        }
    }

    pub(crate) fn div(&self, rhs: &Val) -> Option<Self> {
        match (self, rhs) {
            (Val::Real(x), Val::Real(y)) => {
                assert!(*y != 0.0);
                Some(Val::Real(x / y))
            }
            (Val::Int(x), Val::Int(y)) => {
                assert!(*y != 0);
                Some(Val::Int(x / y))
            }
            _ => None,
        }
    }

    pub fn has_type(&self, ty: &Type) -> bool {
        match self {
            Val::None => false,
            Val::Real(_) => *ty == Type::Real,
            Val::Int(_) => *ty == Type::Int,
            Val::Bool(_) => *ty == Type::Bool,
        }
    }

    pub fn same_type(&self, rhs: &Val) -> bool {
        match self {
            Val::None => matches!(rhs, Val::None),
            Val::Real(_) => matches!(rhs, Val::Real { .. }),
            Val::Int(_) => matches!(rhs, Val::Int { .. }),
            Val::Bool(_) => matches!(rhs, Val::Bool { .. }),
        }
    }

    pub fn from_str(val: &str, ty: Type) -> Option<Val> {
        match ty {
            Type::Bool => {
                if let Ok(val) = val.parse() {
                    return Some(Val::Bool(val));
                }
            }
            Type::Int => {
                if let Ok(val) = val.parse() {
                    return Some(Val::Int(val));
                }
            }
            Type::Real => {
                if let Ok(val) = val.parse() {
                    return Some(Val::Real(val));
                }
            }
        }

        None
    }
}

impl Default for Val {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Val::None => write!(f, "None"),
            Val::Real(x) => write!(f, "Real({})", x),
            Val::Int(x) => write!(f, "Int({})", x),
            Val::Bool(x) => write!(f, "Bool({})", x),
        }
    }
}
