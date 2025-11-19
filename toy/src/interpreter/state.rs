use crate::val::Val;
use std::collections::HashMap;

pub struct State {
    _values: HashMap<usize, Val>,
}

impl State {
    pub fn new() -> Self {
        Self {
            _values: HashMap::new(),
        }
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}
