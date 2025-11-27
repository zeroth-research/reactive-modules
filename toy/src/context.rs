use std::collections::HashMap;

use base::wire::Interface;

// Context for building modules and atoms.
// This is Toy-specific, it will be replaced in the future.
pub struct Context<D: Copy + Eq> {
    vars: HashMap<String, (usize, D)>,
    // XXX: it would be more efficient make the hash map point to the Entry in `vars`
    names: HashMap<usize, String>,
}

impl<D: Copy + Eq> Context<D> {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
            names: HashMap::new(),
        }
    }

    pub fn get_name(&self, id: usize) -> Option<&str> {
        self.names.get(&id).map(|s| s.as_str())
    }

    pub fn get(&self, name: &str) -> (usize, D) {
        let (id, ty) = self.vars.get(name).expect("Not existing value");
        (*id, *ty)
    }

    pub fn get_intf(&self, name: &str) -> Interface<D> {
        let (id, ty) = self.vars.get(name).expect("Not existing value");
        Interface::single(*id, *ty)
    }

    pub fn get_intf_with_type(&self, name: &str) -> (Interface<D>, D) {
        let (id, ty) = self.vars.get(name).expect("Not existing value");
        (Interface::single(*id, *ty), *ty)
    }

    /// Get or create a variable
    /// Does not check if the type is compatible if the var exists
    /// This method also remembers names of variables in `self.names`, because
    /// here we are creating named variables
    pub fn var(&mut self, name: &str, ty: D) -> (usize, D) {
        let new_id = self.vars.len();
        let name = name.to_string();
        let res = *self.vars.entry(name.clone()).or_insert((new_id, ty));
        if res.0 == new_id {
            // the entry was just inserted
            let _inserted = self.names.insert(new_id, name);
            assert!(_inserted.is_none());
        }
        res
    }

    /// TODO: Does not check if the type is compatible if the var exists
    pub fn tmp_var(&mut self, ty: D) -> usize {
        let new_id = self.vars.len();
        self.vars
            .entry(format!("__c_{}", new_id))
            .or_insert((new_id, ty));

        new_id
    }

    pub fn tmp_intf(&mut self, ty: D) -> Interface<D> {
        Interface::single(self.tmp_var(ty), ty)
    }

    pub fn get_vars(&self, names: &[&'static str]) -> Interface<D> {
        let mut vars: Vec<(usize, D)> = Vec::with_capacity(names.len());
        for name in names {
            let v = self.vars.get(*name).expect("Invalid variable");
            vars.push(*v);
        }

        Interface::sequence(vars).unwrap()
    }

    pub fn vars(&mut self, ty: D, names: &[&'static str]) -> Interface<D> {
        let mut tmp = Vec::with_capacity(names.len());
        for name in names {
            let v = self.var(name, ty);
            tmp.push(v)
        }

        Interface::sequence(tmp).unwrap()
    }
}

impl<D: Copy + Eq> Default for Context<D> {
    fn default() -> Self {
        Self::new()
    }
}
