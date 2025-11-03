use std::collections::HashMap;

use base::wire::Wire;

use toy::dtype::Type;
use toy::instruction::Instruction;
use toy::term::{Term, construct};
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
        let init_x =
            construct(Instruction::Const(Val::Int(0)), ctx.get("x'"), Wire::none()).unwrap();
        let init_y = construct(Instruction::Id, ctx.get("y'"), ctx.get("y0'")).unwrap();
        let init_z = construct(Instruction::Id, ctx.get("z'"), ctx.get("z0'")).unwrap();

        vec![init_x, init_y, init_z]
    }

    fn update(ctx: &mut Context) -> Vec<Term> {
        // wire10 = x < y
        let reads = ctx.get_vars(&["x", "y"]);
        let wire10 = ctx.tmp_wire(Type::Bool).clone();
        let xlty = construct(Instruction::Lt, wire10.clone(), reads).unwrap();

        // wire11 = x < z
        let reads = ctx.get_vars(&["x", "z"]);
        let wire11 = ctx.tmp_wire(Type::Bool);
        let xltz = construct(Instruction::Lt, wire11.clone(), reads).unwrap();

        // wire12 = wire10 || wire11
        let wire12 = ctx.tmp_wire(Type::Bool);
        let reads = ctx.concat(&[wire10, wire11]);
        let or = construct(Instruction::Or, wire12.clone(), reads).unwrap();

        // zero
        let const0 = ctx.tmp_wire(Type::Int).clone();
        let term0 = construct(
            Instruction::Const(Val::Int(0)),
            const0.clone(),
            Wire::none(),
        )
        .unwrap();

        // one
        let const1 = ctx.tmp_wire(Type::Int).clone();
        let term1 = construct(
            Instruction::Const(Val::Int(1)),
            const1.clone(),
            Wire::none(),
        )
        .unwrap();

        // wire15 = vars[0] + const1
        let wire15 = ctx.tmp_wire(Type::Int).clone();
        let x = ctx.get("x");
        let reads = ctx.concat(&[x, const1]);
        let sum = construct(Instruction::Add, wire15.clone(), reads).unwrap();

        // wire5 = ite(wire12, wire15, const0)
        let reads = ctx.concat(&[wire12, wire15, const0]);
        let ite = construct(Instruction::Ite, ctx.get("x'"), reads).unwrap();

        // y' := y
        let id_y = construct(Instruction::Id, ctx.get("y'"), ctx.get("y")).unwrap();
        let id_z = construct(Instruction::Id, ctx.get("z'"), ctx.get("z")).unwrap();

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
        let xeqy = construct(Instruction::Eq, wire16.clone(), reads).unwrap();

        let reads = ctx.get_vars(&["x", "z"]);
        let wire17 = ctx.tmp_wire(Type::Bool).clone();
        let xeqz = construct(Instruction::Eq, wire17.clone(), reads).unwrap();

        let out = ctx.tmp_wire(Type::Bool);
        let or = construct(Instruction::Or, out, ctx.concat(&[wire16, wire17]))
            .expect("Failed creating term");

        vec![xeqy, xeqz, or]
    }
}
