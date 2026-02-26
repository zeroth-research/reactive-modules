use std::collections::HashMap;
use tch::Tensor;

pub struct State {
    values: HashMap<usize, Tensor>,
}

impl State {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    pub fn get(&self, wire_id: usize) -> Option<&Tensor> {
        self.values.get(&wire_id)
    }

    pub fn set(&mut self, wire_id: usize, value: Tensor) {
        self.values.insert(wire_id, value);
    }

    pub fn iter(&self) -> impl Iterator<Item = (&usize, &Tensor)> {
        self.values.iter()
    }
}
