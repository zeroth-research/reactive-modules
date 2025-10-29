use base::term::Term as BaseTerm;

use crate::instruction::Instruction;
use crate::val::Type;

pub type Term = BaseTerm<Type, Instruction>;
