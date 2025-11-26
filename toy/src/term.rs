use base::term::Term as BaseTerm;

use crate::dtype::Type;
use crate::instruction::{ArithOp, CmpOp, Instruction, LogicalOp};
use crate::val::Val;
use base::wire::{Interface, Wire};

pub type Term = BaseTerm<Type, Instruction>;

pub fn construct(
    ins: Instruction,
    reads: Interface<Type>,
    writes: Interface<Type>,
) -> Result<Term, &'static str> {
    match &ins {
        Instruction::Const(val) => {
            if !reads.is_empty() {
                return Err("Const should read nothing");
            }
            if writes.len() != 1 {
                return Err("Const must write to exactly one wire");
            }
            let out = writes.wires().next().unwrap();
            if !val.has_type(out.dtype()) {
                return Err("Const writes to incompatible wire");
            }
        }
        Instruction::Cmp(_) => {
            if reads.len() != 2 {
                return Err("Comparison must read two values");
            }
            if writes.len() != 1 {
                return Err("Comparison must write 1 value");
            }
            let out = writes.wires().next().unwrap();
            if *out.dtype() != Type::Bool {
                return Err("Comparison does not write bool");
            }
        }
        Instruction::Ite => {
            if writes.len() != 1 {
                return Err("Logical operation must write one wire");
            }
            if reads.len() != 3 {
                return Err("Ite must read three wires");
            }

            let types: Vec<&Type> = reads.wires().map(Wire::dtype).collect();

            if *types[0] != Type::Bool {
                return Err("Ite first argument must be Bool");
            }
            if *types[1] != *types[2] {
                return Err("Ite second and third arguments must have the same type");
            }
            let out_ty = writes.wires().next().unwrap().dtype();
            if *out_ty != *types[1] {
                return Err("Ite must write the same type as the second and third arguments");
            }
        }
        Instruction::Choose => {
            if writes.len() != 1 {
                return Err("Logical operation must write one wire");
            }

            if reads.is_empty() || !reads.len().is_multiple_of(2) {
                return Err("Choose must read non-zero even number of wires");
            }

            // even types must be bool
            let types = reads.wires().map(Wire::dtype).collect::<Vec<&Type>>();
            for (n, ty) in types.iter().enumerate() {
                if n % 2 == 0 {
                    if **ty != Type::Bool {
                        return Err("Even types in Choose must be Bool");
                    }
                } else if *ty != types[1] {
                    return Err("Odd types in Choose must be all the same");
                }
            }

            // odd types must be the same
            // out type must be the same as odd types
            let out_ty = writes.wires().next().unwrap().dtype();
            if *out_ty != *types[1] {
                return Err("Choose must write the same type as the odd arguments");
            }
        }
        Instruction::Logical(op) => {
            if writes.len() != 1 {
                return Err("Logical operation must write one wire");
            }

            let types: Vec<&Type> = reads.wires().map(Wire::dtype).collect();

            if matches!(op, LogicalOp::Not) {
                if reads.len() != 1 {
                    return Err("Logical and/or must read two values");
                }
            } else if reads.len() != 2 {
                return Err("Logical and/or must read two values");
            }

            for ty in types {
                if *ty != Type::Bool {
                    return Err("Input to or/and/not must be a boolean");
                }
            }
        }
        Instruction::Id => {
            if reads.len() != 1 {
                return Err("Id must read one wire");
            }
            if writes.len() != 1 {
                return Err("Id must write one wire");
            }
            let in_ = reads.wires().next().unwrap();
            let out_ = writes.wires().next().unwrap();
            if *in_.dtype() != *out_.dtype() {
                return Err("Id reads and writes different types");
            }
        }
        Instruction::Arith(_) => {
            if reads.len() != 2 {
                return Err("Add/Sub/Mul/Div must read two wires");
            }
            if writes.len() != 1 {
                return Err("Sum must write one wire");
            }
            let out_ = writes.wires().next().unwrap();
            if *out_.dtype() == Type::Bool {
                return Err("Sum writes Bool");
            }
            for w in reads.wires() {
                if *w.dtype() != *out_.dtype() {
                    return Err("Sum reads and writes different types");
                }
            }
        }
    }

    Ok(Term::new_unchecked(ins, writes, reads))
}

pub fn mk_const(val: &Val, write: Interface<Type>) -> Result<Term, &'static str> {
    construct(Instruction::Const(val.clone()), Interface::empty(), write)
}

pub fn mk_id(read: Interface<Type>, write: Interface<Type>) -> Result<Term, &'static str> {
    construct(Instruction::Id, read, write)
}

pub fn mk_cmp(
    op: CmpOp,
    read: Interface<Type>,
    write: Interface<Type>,
) -> Result<Term, &'static str> {
    construct(Instruction::Cmp(op), read, write)
}

pub fn mk_eq(read: Interface<Type>, write: Interface<Type>) -> Result<Term, &'static str> {
    mk_cmp(CmpOp::Eq, read, write)
}

pub fn mk_le(read: Interface<Type>, write: Interface<Type>) -> Result<Term, &'static str> {
    mk_cmp(CmpOp::Le, read, write)
}

pub fn mk_lt(read: Interface<Type>, write: Interface<Type>) -> Result<Term, &'static str> {
    mk_cmp(CmpOp::Lt, read, write)
}

pub fn mk_logical(
    op: LogicalOp,
    read: Interface<Type>,
    write: Interface<Type>,
) -> Result<Term, &'static str> {
    construct(Instruction::Logical(op), read, write)
}

pub fn mk_or(read: Interface<Type>, write: Interface<Type>) -> Result<Term, &'static str> {
    mk_logical(LogicalOp::Or, read, write)
}

pub fn mk_and(read: Interface<Type>, write: Interface<Type>) -> Result<Term, &'static str> {
    mk_logical(LogicalOp::And, read, write)
}

pub fn mk_not(read: Interface<Type>, write: Interface<Type>) -> Result<Term, &'static str> {
    mk_logical(LogicalOp::Not, read, write)
}

pub fn mk_ite(read: Interface<Type>, write: Interface<Type>) -> Result<Term, &'static str> {
    construct(Instruction::Ite, read, write)
}

pub fn mk_add(read: Interface<Type>, write: Interface<Type>) -> Result<Term, &'static str> {
    construct(Instruction::Arith(ArithOp::Add), read, write)
}

pub fn mk_sub(read: Interface<Type>, write: Interface<Type>) -> Result<Term, &'static str> {
    construct(Instruction::Arith(ArithOp::Sub), read, write)
}

pub fn mk_mul(read: Interface<Type>, write: Interface<Type>) -> Result<Term, &'static str> {
    construct(Instruction::Arith(ArithOp::Mul), read, write)
}

pub fn mk_div(read: Interface<Type>, write: Interface<Type>) -> Result<Term, &'static str> {
    construct(Instruction::Arith(ArithOp::Div), read, write)
}
