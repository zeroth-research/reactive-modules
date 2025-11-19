mod eval;
mod state;

pub use eval::eval;

pub use state::State;

use crate::dtype::Type;
use crate::instruction::Instruction;
use base::module::Module;

pub struct Interpreter<'a> {
    module: &'a Module<Type, Instruction>,
    state: State,
}

impl<'a> Interpreter<'a> {
    pub fn new(module: &'a Module<Type, Instruction>) -> Self {
        Self {
            module,
            state: State::new(),
        }
    }

    /// initialize the execution state
    fn initialize(&mut self) -> Result<(), &'static str> {
        unimplemented!()
    }

    /// take the next step in execution
    fn step(&mut self) -> Result<(), &'static str> {
        unimplemented!()
    }
}
