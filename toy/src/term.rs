use base::term::Term as BaseTerm;

use crate::dtype::DType;
use crate::itype::{ArithOp, CmpOp, IType, LogicalOp};
use crate::val::Val;
use base::wire::{Interface, Wire};

pub type Term = BaseTerm<DType, IType>;

pub fn construct(
    ins: IType,
    reads: Interface<DType>,
    writes: Interface<DType>,
) -> Result<Term, &'static str> {
    match &ins {
        IType::Const(val) => {
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
        IType::Cmp(_) => {
            if reads.len() != 2 {
                return Err("Comparison must read two values");
            }
            if writes.len() != 1 {
                return Err("Comparison must write 1 value");
            }
            let out = writes.wires().next().unwrap();
            if *out.dtype() != DType::Bool {
                return Err("Comparison does not write bool");
            }
        }
        IType::Ite => {
            if writes.len() != 1 {
                return Err("Logical operation must write one wire");
            }
            if reads.len() != 3 {
                return Err("Ite must read three wires");
            }

            let types: Vec<&DType> = reads.wires().map(Wire::dtype).collect();

            if *types[0] != DType::Bool {
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
        IType::Choose => {
            if writes.len() != 1 {
                return Err("Choose must write one wire");
            }

            if reads.is_empty() {
                return Err("Choose must read non-zero number of wires");
            }

            let out_ty = writes.wires().next().unwrap().dtype();
            let types = reads.wires().map(Wire::dtype).collect::<Vec<&DType>>();
            for ty in types {
                if out_ty != ty {
                    return Err("Inputs and outputs to Choose must have the same type");
                }
            }
        }
        IType::Filter => {
            if writes.len() != 1 {
                return Err("Filter must write one wire");
            }

            if reads.len() != 2 {
                return Err("Filter must read two wires");
            }

            let out_ty = writes.wires().next().unwrap().dtype();
            let types = reads.wires().map(Wire::dtype).collect::<Vec<&DType>>();
            if out_ty != types[1] {
                return Err("The second argument of Filter must have the same type as its output");
            }
            if *types[0] != DType::Bool {
                return Err("The first argument of Filter must be Bool");
            }
        }
        IType::Logical(op) => {
            if writes.len() != 1 {
                return Err("Logical operation must write one wire");
            }

            let types: Vec<&DType> = reads.wires().map(Wire::dtype).collect();

            if matches!(op, LogicalOp::Not) {
                if reads.len() != 1 {
                    return Err("Logical and/or must read two values");
                }
            } else if reads.len() != 2 {
                return Err("Logical and/or must read two values");
            }

            for ty in types {
                if *ty != DType::Bool {
                    return Err("Input to or/and/not must be a boolean");
                }
            }
        }
        IType::Id => {
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
        IType::Arith(_) => {
            if reads.len() != 2 {
                return Err("Add/Sub/Mul/Div must read two wires");
            }
            if writes.len() != 1 {
                return Err("Sum must write one wire");
            }
            let out_ = writes.wires().next().unwrap();
            if *out_.dtype() == DType::Bool {
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

pub fn mk_const(val: &Val, write: Interface<DType>) -> Result<Term, &'static str> {
    construct(IType::Const(val.clone()), Interface::empty(), write)
}

pub fn mk_id(read: Interface<DType>, write: Interface<DType>) -> Result<Term, &'static str> {
    construct(IType::Id, read, write)
}

pub fn mk_cmp(
    op: CmpOp,
    read: Interface<DType>,
    write: Interface<DType>,
) -> Result<Term, &'static str> {
    construct(IType::Cmp(op), read, write)
}

pub fn mk_eq(read: Interface<DType>, write: Interface<DType>) -> Result<Term, &'static str> {
    mk_cmp(CmpOp::Eq, read, write)
}

pub fn mk_le(read: Interface<DType>, write: Interface<DType>) -> Result<Term, &'static str> {
    mk_cmp(CmpOp::Le, read, write)
}

pub fn mk_lt(read: Interface<DType>, write: Interface<DType>) -> Result<Term, &'static str> {
    mk_cmp(CmpOp::Lt, read, write)
}

pub fn mk_logical(
    op: LogicalOp,
    read: Interface<DType>,
    write: Interface<DType>,
) -> Result<Term, &'static str> {
    construct(IType::Logical(op), read, write)
}

pub fn mk_or(read: Interface<DType>, write: Interface<DType>) -> Result<Term, &'static str> {
    mk_logical(LogicalOp::Or, read, write)
}

pub fn mk_and(read: Interface<DType>, write: Interface<DType>) -> Result<Term, &'static str> {
    mk_logical(LogicalOp::And, read, write)
}

pub fn mk_not(read: Interface<DType>, write: Interface<DType>) -> Result<Term, &'static str> {
    mk_logical(LogicalOp::Not, read, write)
}

pub fn mk_ite(read: Interface<DType>, write: Interface<DType>) -> Result<Term, &'static str> {
    construct(IType::Ite, read, write)
}

pub fn mk_filter(read: Interface<DType>, write: Interface<DType>) -> Result<Term, &'static str> {
    construct(IType::Filter, read, write)
}

pub fn mk_add(read: Interface<DType>, write: Interface<DType>) -> Result<Term, &'static str> {
    construct(IType::Arith(ArithOp::Add), read, write)
}

pub fn mk_sub(read: Interface<DType>, write: Interface<DType>) -> Result<Term, &'static str> {
    construct(IType::Arith(ArithOp::Sub), read, write)
}

pub fn mk_mul(read: Interface<DType>, write: Interface<DType>) -> Result<Term, &'static str> {
    construct(IType::Arith(ArithOp::Mul), read, write)
}

pub fn mk_div(read: Interface<DType>, write: Interface<DType>) -> Result<Term, &'static str> {
    construct(IType::Arith(ArithOp::Div), read, write)
}
