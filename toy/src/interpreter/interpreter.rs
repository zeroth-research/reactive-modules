use crate::interpreter::State;

use crate::dtype::Type;
use crate::instruction::Instruction;
use base::module::Module;

pub struct Interpreter<'a> {
    module: &'a Module<Type, Instruction>,
    state: State,
}

impl<'a> Interpreter<'a> {
    pub fn new() -> Self {
        unimplemented!()
    }

    fn step(&mut self) -> Result<(), &'static str> {
        unimplemented!()
    }
}
