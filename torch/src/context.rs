use std::collections::HashMap;

/// Context for generating Atoms and Modules from Python
pub struct Context {
    // maps names of variables to numerical identifiers in wires
    name_to_id: HashMap<String, usize>, // TODO: should we keep track of variable types?
    id_to_name: HashMap<usize, String>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    pub fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
            id_to_name: HashMap::new(),
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
        let new_id = self.name_to_id.len();
        let name = name.to_string();
        let res = *self.name_to_id.entry(name.clone()).or_insert(new_id);
        if res == new_id {
            // the entry was just inserted, remember the name
            let _inserted = self.id_to_name.insert(new_id, name);
            assert!(_inserted.is_none());
        }
        res
    }

    pub fn get_name(&self, id: usize) -> Option<&str> {
        self.id_to_name.get(&id).map(|s| s.as_str())
    }
}
