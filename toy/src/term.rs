use base::term::Term as BaseTerm;

use crate::dtype::Type;
use crate::instruction::Instruction;
use base::wire::Wire;

pub type Term = BaseTerm<Type, Instruction>;

pub fn construct(
    ins: Instruction,
    writes: Wire<Type>,
    reads: Wire<Type>,
) -> Result<Term, &'static str> {
    match ins {
        Instruction::Const(val) => {
            if !reads.is_empty() {
                return Err("Const should read nothing");
            }
            if writes.len() != 1 {
                return Err("Const must write to exactly one wire");
            }
            let out = writes.iter().next().unwrap();
            if !val.has_type(out.1) {
                return Err("Const writes to incompatible wire");
            }
        }
        Instruction::Eq | Instruction::Lt => {
            if !reads.len() == 2 {
                return Err("Comparison must read two values");
            }
            if !writes.len() == 1 {
                return Err("Comparison must write 1 value");
            }
            let out = writes.iter().next().unwrap();
            if *out.1 != Type::Bool {
                return Err("Comparison does not write bool");
            }
        }
        Instruction::Or => {
            if !reads.len() == 2 {
                return Err("Comparison must read two values");
            }
            if !writes.len() == 1 {
                return Err("Comparison must write 1 value");
            }
            let out = writes.iter().next().unwrap();
            if *out.1 != Type::Bool {
                return Err("Comparison does not write bool");
            }
            for (_, ty) in reads.iter() {
                if *ty != Type::Bool {
                    return Err("Input to Or must be a boolean");
                }
            }
        }
        Instruction::Id => {
            if !reads.len() == 1 {
                return Err("Id must read one wire");
            }
            if !writes.len() == 1 {
                return Err("Id must write one wire");
            }
            let in_ = reads.iter().next().unwrap();
            let out_ = writes.iter().next().unwrap();
            if *in_.1 != *out_.1 {
                return Err("Id reads and writes different types");
            }
        }
        Instruction::Sum => {
            if !reads.len() == 2 {
                return Err("Sum must read two wires");
            }
            if !writes.len() == 1 {
                return Err("Sum must write one wire");
            }
            let out_ = writes.iter().next().unwrap();
            if *out_.1 == Type::Bool {
                return Err("Sum writes Bool");
            }
            for (_, ty) in reads.iter() {
                if *ty != *out_.1 {
                    return Err("Sum reads and writes different types");
                }
            }
        }
        Instruction::Ite => {
            if !reads.len() == 3 {
                return Err("Ite must read three wires");
            }
            if !writes.len() == 1 {
                return Err("Ite must write one wire");
            }
            let out_ty = writes.iter().next().unwrap().1;
            let types: Vec<&Type> = reads.iter().map(|(_, ty)| ty).collect();

            if *types[0] != Type::Bool {
                return Err("Ite first argument must be Bool");
            }
            if *types[1] != *types[2] {
                return Err("Ite second and thrid arguments must have the same type");
            }
            if *out_ty != *types[1] {
                return Err("Ite must write the same type as the second and third arguments");
            }
        }
    }

    Ok(Term::new(ins, writes, reads))
}
