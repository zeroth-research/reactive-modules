use crate::val::Val;

#[derive(Debug)]
pub enum Instruction {
    // constant and identity terms
    Const(Val),
    Id,
    // Comparisons
    Eq,
    Lt,
    // Logical ops
    Or,
    Ite,
    // arith
    Sum,
}
