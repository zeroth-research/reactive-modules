use crate::val::Val;

#[derive(Debug)]
pub enum Instruction {
    // constant and identity terms
    Const(Val),
    Id,
    // Comparisons
    Eq,
    Lt,
    Le,
    // Logical ops
    And,
    Or,
    Ite,
    Not,
    // arith
    Add,
    Sub,
    Mul,
    Div,
}
