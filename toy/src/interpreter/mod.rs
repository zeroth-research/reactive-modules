mod eval;
mod state;

pub use eval::eval;

pub use state::State;

use crate::IType;
use crate::DType;
use base::module::Module;

pub struct Interpreter<'a> {
    _module: &'a Module<DType, IType>,
    _state: State,
}

impl<'a> Interpreter<'a> {
    pub fn new(module: &'a Module<DType, IType>) -> Self {
        Self {
            _module: module,
            _state: State::new(),
        }
    }

    /// initialize the execution state
    fn _initialize(&mut self) -> Result<(), &'static str> {
        unimplemented!()
    }

    /// take the next step in execution
    fn _step(&mut self) -> Result<(), &'static str> {
        unimplemented!()
    }
}
