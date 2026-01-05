use std::collections::{HashMap, HashSet};

use base::{Interface, Wire};

/// Context for building modules.
/// It keeps track of known wires IDs and names associated to
/// some of the wires.
pub struct Context<D: Copy + Eq> {
    /// Associating names to some wires. Values are tuples
    /// instead of [Wire]s as having [Wire]s in the map would
    /// result in a lot of conversions between tuples and [Wire]s
    vars: HashMap<String, (usize, D)>,

    /// Associating wires (their IDs, more precisely) to names
    // XXX: would it be more efficient to make the hash map point to the Entries in `vars`
    names: HashMap<usize, String>,

    /// IDs of temporary wires. We could keep them in `names` with some dummy name,
    /// but having them separately is cleaner in prone to optimizations in the future.
    temps: HashSet<usize>,
}

impl<D: Copy + Eq> Context<D> {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
            names: HashMap::new(),
            temps: HashSet::new(),
        }
    }

    /// Create a context and populate it from a module.
    /// We do not have names in this case, so assign some dummy names.
    /// This is useful if we need to create new variables and make sure
    /// they are not already used ("in the current context")
    pub fn from_module<IT>(module: &base::Module<D, IT>) -> Self {
        let mut ctx: Context<D> = Context::new();

        // create variables in the context for module wires
        for [wl, wn] in module.ctrl().iter().chain(module.extl()) {
            ctx.var(format!("x{}", wl.id()).as_str(), *wl.dtype());
            ctx.var(format!("x{}", wn.id()).as_str(), *wn.dtype());
        }

        // create variables in the context for temporary wires
        for atom in module.atoms() {
            for w in atom.temp() {
                ctx.insert_tmp_id(w.id());
            }
        }

        ctx
    }

    /// Get a name for a wire ID. Return `Some(name)` if the wire with the given
    /// ID has associated a name, otherwise return `None`
    pub fn get_name(&self, id: usize) -> Option<&str> {
        self.names.get(&id).map(|s| s.as_str())
    }

    /// Get a named wire in the context. Return it as a tuple (id, DType)
    pub fn get(&self, name: &str) -> Option<(usize, D)> {
        self.vars.get(name).cloned()
    }

    /// Get a wire associated with the given `name` and wrap it into an [Interface].
    pub fn get_wire(&self, name: &str) -> Option<Wire<D>> {
        if let Some((id, ty)) = self.vars.get(name) {
            return Some(Wire::new(*id, *ty));
        }
        None
    }

    /// Get interface from named wires. Return `None` if some of the given names
    /// is not found.
    pub fn get_intf(&self, names: &[&str]) -> Option<Interface<D>> {
        let mut vars: Vec<(usize, D)> = Vec::with_capacity(names.len());
        for name in names {
            let v = self.vars.get(*name)?;
            vars.push(*v);
        }

        Some(Interface::sequence(vars).ok()?)
    }

    /// Return a reference to the HashMap with names
    /// TODO: turn to iterator if possible
    pub fn names(&self) -> &HashMap<usize, String> {
        &self.names
    }

    /// Create a temporary wire and return it
    pub fn tmp_wire(&mut self, ty: D) -> Wire<D> {
        Wire::new(self.tmp_id(), ty)
    }

    /// Create an interface for named wires
    pub fn intf(&mut self, ty: D, names: &[&'static str]) -> Interface<D> {
        let mut tmp = Vec::with_capacity(names.len());
        for name in names {
            let v = self.var(name, ty);
            tmp.push(v)
        }

        Interface::sequence(tmp).unwrap()
    }

    /// Create a temporary wire and return it wrapped in an interface
    pub fn tmp_intf(&mut self, ty: D) -> Interface<D> {
        Interface::single(self.tmp_id(), ty)
    }

    /// create a "copy"" of interface with new fresh wires.
    /// That is, create a new interface with fresh wires that have the same DTypes
    /// as wires in the given interface.
    pub fn fresh_intf(&mut self, intf: &Interface<D, 1>) -> Interface<D, 1> {
        Interface::sequence(
            intf.wires()
                .map(|w| base::Wire::new(self.tmp_id(), *w.dtype())),
        )
        .unwrap()
    }

    /// Get or create a variable
    /// Does not check if the type is compatible if the var exists
    /// This method also remembers names of variables in `self.names`, because
    /// here we are creating named variables
    pub fn var(&mut self, name: &str, ty: D) -> (usize, D) {
        let new_id = self.get_next_id();
        let name = name.to_string();
        let res = *self.vars.entry(name.clone()).or_insert((new_id, ty));
        if res.0 == new_id {
            // the entry was just inserted
            let _inserted = self.names.insert(new_id, name);
            debug_assert!(_inserted.is_none());
            debug_assert!(self.get_next_id() != new_id);
        }
        res
    }

    pub fn tmp_id(&mut self) -> usize {
        let new_id = self.get_next_id();
        if !self.insert_tmp_id(new_id) {
            panic!("BUG: Already has this temp ID");
        }
        debug_assert!(self.get_next_id() != new_id);
        new_id
    }

    fn get_next_id(&self) -> usize {
        self.temps.len() + self.vars.len()
    }

    fn insert_tmp_id(&mut self, new_id: usize) -> bool {
        self.temps.insert(new_id)
    }
}

impl<D: Copy + Eq> Default for Context<D> {
    fn default() -> Self {
        Self::new()
    }
}
