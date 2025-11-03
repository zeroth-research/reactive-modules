use std::collections::HashMap;

use base::wire::Wire;

use toy::dtype::Type;
use toy::instruction::Instruction;
use toy::term::*;
use toy::val::Val;

struct Context {
    vars: HashMap<String, (usize, Type)>,
}

impl Context {
    fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    fn get(&mut self, name: &'static str) -> Wire<Type> {
        let (id, ty) = self.vars.get(name).expect("Not existing value");
        Wire::one(*id, *ty)
    }

    /// Get or create a variable
    /// Does not check if the type is compatible if the var exists
    fn var(&mut self, name: &'static str, ty: Type) -> (usize, Type) {
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

    fn tmp_wire(&mut self, ty: Type) -> Wire<Type> {
        Wire::one(self.tmp_var(ty), ty)
    }

    fn get_vars(&mut self, names: &[&'static str]) -> Wire<Type> {
        let mut vars: Vec<(usize, Type)> = Vec::with_capacity(names.len());
        for name in names {
            let v = self.vars.get(*name).expect("Invalid variable");
            vars.push(*v);
        }

        Wire::from_iter(vars)
    }

    // Union several wires
    fn concat<'a, I>(&mut self, wires: I) -> Wire<Type>
    where
        I: IntoIterator<Item = &'a Wire<Type>>,
    {
        let mut tmp: Vec<(usize, Type)> = Vec::new();
        for wire in wires {
            tmp.extend(wire.iter().map(|(id, ty)| (id, *ty)))
        }

        Wire::from_iter(tmp)
    }

    fn vars(&mut self, ty: Type, names: &[&'static str]) -> Wire<Type> {
        let mut tmp = Vec::with_capacity(names.len());
        for name in names {
            let v = self.var(name, ty);
            tmp.push(v)
        }

        Wire::from_iter(tmp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::atom::Atom;
    use base::module::Module;

    fn init(ctx: &mut Context) -> Vec<Term> {
        let init_x = mk_const(&Val::Int(0), ctx.get("x'")).unwrap();
        let init_y = mk_id(ctx.get("y0'"), ctx.get("y'")).unwrap();
        let init_z = mk_id(ctx.get("z0'"), ctx.get("z'")).unwrap();

        vec![init_x, init_y, init_z]
    }

    fn update(ctx: &mut Context) -> Vec<Term> {
        // wire10 = x < y
        let reads = ctx.get_vars(&["x", "y"]);
        let wire10 = ctx.tmp_wire(Type::Bool).clone();
        let xlty = mk_lt(reads, wire10.clone()).unwrap();

        // wire11 = x < z
        let reads = ctx.get_vars(&["x", "z"]);
        let wire11 = ctx.tmp_wire(Type::Bool);
        let xltz = mk_lt(reads, wire11.clone()).unwrap();

        // wire12 = wire10 || wire11
        let wire12 = ctx.tmp_wire(Type::Bool);
        let reads = ctx.concat(&[wire10, wire11]);
        let or = mk_or(reads, wire12.clone()).unwrap();

        // zero
        let const0 = ctx.tmp_wire(Type::Int).clone();
        let term0 = mk_const(&Val::Int(0), const0.clone()).unwrap();

        // one
        let const1 = ctx.tmp_wire(Type::Int).clone();
        let term1 = mk_const(&Val::Int(1), const1.clone()).unwrap();

        // wire15 = vars[0] + const1
        let wire15 = ctx.tmp_wire(Type::Int).clone();
        let x = ctx.get("x");
        let reads = ctx.concat(&[x, const1]);
        let sum = mk_add(reads, wire15.clone()).unwrap();

        // wire5 = ite(wire12, wire15, const0)
        let reads = ctx.concat(&[wire12, wire15, const0]);
        let ite = mk_ite(reads, ctx.get("x'")).unwrap();

        // y' := y
        let id_y = mk_id(ctx.get("y"), ctx.get("y'")).unwrap();
        let id_z = mk_id(ctx.get("z"), ctx.get("z'")).unwrap();

        vec![xlty, xltz, or, term0, term1, sum, ite, id_y, id_z]
    }

    #[test]
    fn toy_example() {
        let mut ctx = Context::new();

        // create variables
        ctx.vars(
            Type::Int,
            &["x", "y", "z", "y0", "z0", "x'", "y'", "z'", "y0'", "z0'"],
        );

        // build module
        let module = build_module(&mut ctx);
        dbg!(module);

        let prop = build_prop(&mut ctx);
        dbg!(prop);
    }

    fn build_module(ctx: &mut Context) -> Module<Type, Instruction> {
        let init_terms = init(ctx);
        let update_terms = update(ctx);

        let latched = ctx.get_vars(&["x", "y", "z", "y0", "z0"]);
        let next = ctx.get_vars(&["x'", "y'", "z'", "y0'", "z0'"]);

        let atom =
            Atom::with_module_wire(&[latched.clone(), next.clone()], init_terms, update_terms)
                .expect("failed creating atom");

        Module::observable([latched, next], vec![atom]).expect("Failed building module")
    }

    fn build_prop(ctx: &mut Context) -> Vec<Term> {
        let reads = ctx.get_vars(&["x", "y"]);
        let wire16 = ctx.tmp_wire(Type::Bool).clone();
        let xeqy = mk_eq(reads, wire16.clone()).unwrap();

        let reads = ctx.get_vars(&["x", "z"]);
        let wire17 = ctx.tmp_wire(Type::Bool).clone();
        let xeqz = mk_eq(reads, wire17.clone()).unwrap();

        let out = ctx.tmp_wire(Type::Bool);
        let or = mk_or(ctx.concat(&[wire16, wire17]), out).expect("Failed creating term");

        vec![xeqy, xeqz, or]
    }
}
