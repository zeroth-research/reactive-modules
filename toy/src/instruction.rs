use crate::val::Val;

#[derive(Debug, Copy, Clone)]
pub enum CmpOp {
    Eq,
    Lt,
    Le,
}

#[derive(Debug, Copy, Clone)]
pub enum LogicalOp {
    And,
    Or,
    Not,
}

#[derive(Debug, Copy, Clone)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Copy, Clone)]
pub enum Instruction {
    // constant and identity terms
    Const(Val),
    Id,
    // Comparisons
    Cmp(CmpOp),
    // Logical ops
    Logical(LogicalOp),
    Ite,
    // arith
    Arith(ArithOp),
}
