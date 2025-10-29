#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Type {
    Real,
    NReal,
    Bool,
}

#[derive(Debug, Clone, Copy)]
pub enum Val {
    Real(f64),
    NReal(f64),
    Bool(bool),
}
