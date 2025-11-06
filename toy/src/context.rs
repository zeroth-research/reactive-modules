use std::collections::HashMap;

use base::wire::Wire;

use crate::dtype::Type;

///
// Context for building modules and atoms.
// This is Toy-specific, it will be replaced in the future.
pub struct Context {
    vars: HashMap<String, (usize, Type)>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    pub fn get(&mut self, name: &str) -> Wire<Type> {
        let (id, ty) = self.vars.get(name).expect("Not existing value");
        Wire::one(*id, *ty)
    }

    pub fn get_with_type(&mut self, name: &str) -> (Wire<Type>, Type) {
        let (id, ty) = self.vars.get(name).expect("Not existing value");
        (Wire::one(*id, *ty), *ty)
    }

    /// Get or create a variable
    /// Does not check if the type is compatible if the var exists
    pub fn var(&mut self, name: &str, ty: Type) -> (usize, Type) {
        let new_id = self.vars.len();
        *self.vars.entry(name.to_string()).or_insert((new_id, ty))
    }

    /// Does not check if the type is compatible if the var exists
    fn tmp_var(&mut self, ty: Type) -> usize {
        let new_id = self.vars.len();
        self.vars
            .entry(format!("__c_{}", new_id))
            .or_insert((new_id, ty));

        new_id
    }

    pub fn tmp_wire(&mut self, ty: Type) -> Wire<Type> {
        Wire::one(self.tmp_var(ty), ty)
    }

    pub fn get_vars(&mut self, names: &[&'static str]) -> Wire<Type> {
        let mut vars: Vec<(usize, Type)> = Vec::with_capacity(names.len());
        for name in names {
            let v = self.vars.get(*name).expect("Invalid variable");
            vars.push(*v);
        }

        Wire::from_iter(vars)
    }

    // Union several wires
    pub fn concat<'a, I>(&mut self, wires: I) -> Wire<Type>
    where
        I: IntoIterator<Item = &'a Wire<Type>>,
    {
        let mut tmp: Vec<(usize, Type)> = Vec::new();
        for wire in wires {
            tmp.extend(wire.iter().map(|(id, ty)| (id, *ty)))
        }

        Wire::from_iter(tmp)
    }

    pub fn vars(&mut self, ty: Type, names: &[&'static str]) -> Wire<Type> {
        let mut tmp = Vec::with_capacity(names.len());
        for name in names {
            let v = self.var(name, ty);
            tmp.push(v)
        }

        Wire::from_iter(tmp)
    }
}
