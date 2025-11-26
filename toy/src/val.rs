use crate::{dtype::Type, mat::MatVecIter, mat::VecIter};
use std::fmt;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Val {
    None, // no value
    Real(f64),
    Int(i64),
    Bool(bool),
    MatInt(Vec<Vec<i64>>),
    MatReal(Vec<Vec<f64>>),
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

    pub fn get_type(&self) -> Type {
        match self {
            Val::None => panic!("None has no type"),
            Val::Real(_) => Type::Real,
            Val::Int(_) => Type::Int,
            Val::Bool(_) => Type::Bool,
            Val::MatInt(v) => {
                let n = v.len();
                let m = if n > 0 { v[0].len() } else { 0 };
                Type::MatInt(n, m)
            }
            Val::MatReal(v) => {
                let n = v.len();
                let m = if n > 0 { v[0].len() } else { 0 };
                Type::MatReal(n, m)
            }
        }
    }

    pub fn has_type(&self, ty: &Type) -> bool {
        if matches!(self, Val::None) {
            return false;
        }
        self.get_type() == *ty
    }

    pub fn same_type(&self, rhs: &Val) -> bool {
        self.get_type() == rhs.get_type()
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
            Type::MatInt(m, n) => {
                let vecs = MatVecIter::new(val)
                    .map(|s| VecIter::<i64>::new(s).collect::<Vec<i64>>())
                    .collect::<Vec<Vec<i64>>>();
                // parsing some vectors may have failed which would yield an incomplete matrix
                // check the dimensions
                if vecs.len() == m && vecs.iter().all(|v| v.len() == n) {
                    return Some(Val::MatInt(vecs));
                }
                return None;
            }
            Type::MatReal(m, n) => {
                let vecs = MatVecIter::new(val)
                    .map(|s| VecIter::<f64>::new(s).collect::<Vec<f64>>())
                    .collect::<Vec<Vec<f64>>>();
                // parsing some vectors may have failed which would yield an incomplete matrix
                // check the dimensions
                if vecs.len() == m && vecs.iter().all(|v| v.len() == n) {
                    return Some(Val::MatReal(vecs));
                }
                return None;
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

fn format_mat_row<T: std::fmt::Display>(row: &[T]) -> String {
    let inner = row
        .iter()
        .take(3)
        .map(|i| i.to_string())
        .collect::<Vec<String>>()
        .join(", ");
    let dots = if row.len() > 3 { ", ..." } else { "" };

    format!("[{inner}{dots}]")
}

fn format_mat_rows<T: std::fmt::Display>(rows: &[Vec<T>]) -> String {
    let inner = rows
        .iter()
        .take(3)
        .map(|row| format_mat_row(row))
        .collect::<Vec<String>>()
        .join(", ");
    let dots = if rows.len() > 3 { ", ..." } else { "" };

    format!("{inner}{dots}")
}

impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Val::None => write!(f, "None"),
            Val::Real(x) => write!(f, "Real({})", x),
            Val::Int(x) => write!(f, "Int({})", x),
            Val::Bool(x) => write!(f, "Bool({})", x),
            Val::MatInt(v) => write!(f, "{}({})", self.get_type(), format_mat_rows(v)),
            Val::MatReal(v) => write!(f, "{}({})", self.get_type(), format_mat_rows(v)),
        }
    }
}
