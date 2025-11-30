use std::collections::HashMap;

use base::wire::{Interface, Wire};

use crate::dtype::Type;

// Context for building modules and atoms.
// This is Toy-specific, it will be replaced in the future.
pub struct Context {
    vars: HashMap<String, (usize, Type)>,
    // XXX: it would be more efficient make the hash map point to the Entry in `vars`
    names: HashMap<usize, String>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
            names: HashMap::new(),
        }
    }

    pub fn get_name(&self, id: usize) -> Option<&str> {
        self.names.get(&id).map(|s| s.as_str())
    }

    pub fn get(&self, name: &str) -> Interface<Type> {
        let (id, ty) = self.vars.get(name).expect("Not existing value");
        Interface::one(*id, *ty)
    }

    pub fn get_with_type(&self, name: &str) -> (Interface<Type>, Type) {
        let (id, ty) = self.vars.get(name).expect("Not existing value");
        (Interface::one(*id, *ty), *ty)
    }

    /// Get or create a variable
    /// Does not check if the type is compatible if the var exists
    /// This method also remembers names of variables in `self.names`, because
    /// here we are creating named variables
    pub fn var(&mut self, name: &str, ty: Type) -> (usize, Type) {
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

    /// Does not check if the type is compatible if the var exists
    fn tmp_var(&mut self, ty: Type) -> usize {
        let new_id = self.vars.len();
        self.vars
            .entry(format!("__c_{}", new_id))
            .or_insert((new_id, ty));

        new_id
    }

    pub fn tmp_wire(&mut self, ty: Type) -> Interface<Type> {
        Interface::one(self.tmp_var(ty), ty)
    }

    pub fn get_vars(&self, names: &[&'static str]) -> Interface<Type> {
        let mut vars: Vec<(usize, Type)> = Vec::with_capacity(names.len());
        for name in names {
            let v = self.vars.get(*name).expect("Invalid variable");
            vars.push(*v);
        }

        Interface::sequence(vars).unwrap()
    }

    // Union several wires
    pub fn concat<'a, I>(&self, wires: I) -> Interface<Type>
    where
        I: IntoIterator<Item = &'a Interface<Type>>,
    {
        let mut tmp: Vec<(usize, Type)> = Vec::new();
        for wire in wires {
            tmp.extend(wire.wires().map(|w| (w.id(), w.dtype().clone())))
        }

        Interface::sequence(tmp).unwrap()
    }

    pub fn vars(&mut self, ty: Type, names: &[&'static str]) -> Interface<Type> {
        let mut tmp = Vec::with_capacity(names.len());
        for name in names {
            let v = self.var(name, ty);
            tmp.push(v)
        }

        Interface::sequence(tmp).unwrap()
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
