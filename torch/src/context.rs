use pyo3::prelude::*;

use std::collections::HashMap;

/// Context for generating Atoms and Modules from Python
#[pyclass]
pub struct Context {
    // maps names of variables to numerical identifiers in wires
    name_to_id: HashMap<String, usize>, // TODO: should we keep track of variable types?
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl Context {
    #[new]
    pub fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
        }
    }

    pub fn fresh_var(&mut self) -> usize {
        let v = self.name_to_id.len();
        self.name_to_id.insert(format!("__c_{}", v), v);
        v
    }

    pub fn fresh_var_with_name(&mut self) -> (String, usize) {
        let v = self.name_to_id.len();
        let name = format!("__c_{}", v);
        self.name_to_id.insert(name.clone(), v);
        (name, v)
    }

    pub fn get_var(&mut self, name: &str) -> usize {
        let next_id = self.name_to_id.len();
        *self.name_to_id.entry(name.to_string()).or_insert(next_id)
    }
}
