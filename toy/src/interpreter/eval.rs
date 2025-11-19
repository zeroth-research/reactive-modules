use crate::dtype::Type;
use crate::instruction::{ArithOp, CmpOp, Instruction, LogicalOp};
use crate::val::Val;

pub fn eval(op: Instruction, read: &[&Val], write: &mut [&mut Val]) -> Result<(), &'static str> {
    match op {
        Instruction::Const(val) => {
            // passing a wrong number of arguments is a programmatical error,
            // use asserts instead of returning Err()
            debug_assert!(read.is_empty());
            debug_assert!(write.len() == 1);
            *write[0] = val;
        }
        Instruction::Id => {
            debug_assert!(read.len() == 1);
            debug_assert!(write.len() == 1);
            *write[0] = *read[0];
        }
        // Comparisons
        Instruction::Cmp(op) => {
            debug_assert!(read.len() == 2);
            debug_assert!(write.len() == 1);
            if read[0].same_type(read[1]) {
                *write[0] = match op {
                    CmpOp::Eq => Val::Bool(read[0] == read[1]),
                    CmpOp::Lt => Val::Bool(read[0] < read[1]),
                    CmpOp::Le => Val::Bool(read[0] <= read[1]),
                }
            } else {
                return Err("Can compare only values with the same type for EQ");
            }
        }
        Instruction::Logical(op) => {
            debug_assert!(write.len() == 1);
            if matches!(op, LogicalOp::Not)
                && let Val::Bool(b) = read[0]
            {
                debug_assert!(read.len() == 1);
                *write[0] = Val::Bool(!*b);
                return Ok(());
            }

            debug_assert!(read.len() == 2);
            if let (Val::Bool(b1), Val::Bool(b2)) = (read[0], read[1]) {
                *write[0] = match op {
                    LogicalOp::And => Val::Bool(*b1 && *b2),
                    LogicalOp::Or => Val::Bool(*b1 || *b2),
                    _ => panic!("BUG: those should be handled now"),
                }
            } else {
                return Err("Or/And expects two boolean inputs");
            }
        }
        Instruction::Ite => {
            debug_assert!(read.len() == 3);
            debug_assert!(write.len() == 1);
            if read[0].has_type(&Type::Bool)
                && read[1].same_type(read[2])
                && let Val::Bool(cond) = read[0]
            {
                if *cond {
                    *write[0] = *read[1];
                } else {
                    *write[0] = *read[2];
                }
            } else {
                return Err(
                    "Ite expects first input be boolean and then two values of the same type",
                );
            }
        }
        // arith
        Instruction::Arith(op) => {
            debug_assert!(read.len() == 2);
            debug_assert!(write.len() == 1);
            if read[0].same_type(read[1]) {
                let res = match op {
                    ArithOp::Add => read[0].add(read[1]),
                    ArithOp::Mul => read[0].mul(read[1]),
                    ArithOp::Sub => read[0].sub(read[1]),
                    ArithOp::Div => read[0].div(read[1]),
                };
                if let Some(val) = res {
                    *write[0] = val;
                } else {
                    return Err("Couldn't use the input values");
                }
            } else {
                return Err("Can compare only values with the same type for LT");
            }
        }
    }

    Ok(())
}
