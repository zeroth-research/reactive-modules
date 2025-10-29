#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Type {
    Real,
    NReal,
    Int,
    NInt,
    Bool,
}

#[derive(Debug, Clone, Copy)]
pub enum Val {
    Real(f64),
    NReal(f64),
    Int(usize),
    NInt(isize),
    Bool(bool),
}
