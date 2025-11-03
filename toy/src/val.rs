use crate::dtype::Type;

#[derive(Debug, Clone, Copy)]
pub enum Val {
    Real(f64),
    NReal(f64),
    Int(usize),
    NInt(isize),
    Bool(bool),
}

impl Val {
    pub fn same_type(&self, ty: &Type) -> bool {
        match self {
            Val::Real(_) => *ty == Type::Real,
            Val::NReal(_) => *ty == Type::NReal,
            Val::Int(_) => *ty == Type::Int,
            Val::NInt(_) => *ty == Type::NInt,
            Val::Bool(_) => *ty == Type::Bool,
        }
    }
}
