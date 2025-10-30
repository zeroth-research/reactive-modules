use std::collections::HashMap;

use base::wire::Wire;

use toy::instruction::Instruction;
use toy::term::{Term, construct};
use toy::val::{Type, Val};

struct Context {
    vars: HashMap<String, Wire<Type>>,
}

impl Context {
    fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    fn get(&mut self, name: &'static str) -> &Wire<Type> {
        self.vars.get(name).expect("Not existing value")
    }

    fn get_cloned(&mut self, name: &'static str) -> Wire<Type> {
        self.vars.get(name).expect("Not existing value").clone()
    }

    /// Get or create a variable
    /// Does not check if the type is compatible if the var exists
    fn var(&mut self, name: &'static str, ty: Type) -> &Wire<Type> {
        let new_id = self.vars.len();
        self.vars
            .entry(name.to_string())
            .or_insert(Wire::one(new_id, ty))
    }

    /// Does not check if the type is compatible if the var exists
    fn tmp_var(&mut self, ty: Type) -> &Wire<Type> {
        let new_id = self.vars.len();
        self.vars
            .entry(format!("__c_{}", new_id))
            .or_insert(Wire::one(new_id, ty))
    }

    fn get_vars(&mut self, names: Vec<&'static str>) -> Wire<Type> {
        // XXX: not very efficient
        let mut wire = Wire::none();
        for name in names {
            let v = self.get(name);
            wire = wire.union(v).unwrap();
        }

        wire
    }

    fn vars(&mut self, ty: Type, names: Vec<&'static str>) -> Wire<Type> {
        // XXX: not very efficient
        let mut wire = Wire::none();
        for name in names {
            let v = self.var(name, ty);
            wire = wire.union(v).unwrap();
        }

        wire
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::atom::Atom;
    use base::module::Module;

    fn init(ctx: &mut Context) -> Vec<Term> {
        let init_x = construct(
            Instruction::Const(Val::Int(0)),
            ctx.get_cloned("x'"),
            Wire::none(),
        )
        .unwrap();
        let init_y =
            construct(Instruction::Id, ctx.get_cloned("y'"), ctx.get_cloned("y0'")).unwrap();
        let init_z =
            construct(Instruction::Id, ctx.get_cloned("z'"), ctx.get_cloned("z0'")).unwrap();

        vec![init_x, init_y, init_z]
    }

    fn update(ctx: &mut Context) -> Vec<Term> {
        // wire10 = x < y
        let reads = ctx.get_vars(vec!["x", "y"]);
        let wire10 = ctx.tmp_var(Type::Bool).clone();
        let xlty = construct(Instruction::Lt, wire10.clone(), reads).unwrap();

        // wire11 = x < z
        let reads = ctx.get_vars(vec!["x", "z"]);
        let wire11 = ctx.tmp_var(Type::Bool).clone();
        let xltz = construct(Instruction::Lt, wire11.clone(), reads).unwrap();

        // wire12 = wire10 || wire11
        let wire12 = ctx.tmp_var(Type::Bool).clone();
        let reads = wire10.union(&wire11).unwrap();
        let or = construct(Instruction::Or, wire12.clone(), reads).unwrap();

        // zero
        let const0 = ctx.tmp_var(Type::Int).clone();
        let term0 = construct(
            Instruction::Const(Val::Int(0)),
            const0.clone(),
            Wire::none(),
        )
        .unwrap();

        // one
        let const1 = ctx.tmp_var(Type::Int).clone();
        let term1 = construct(
            Instruction::Const(Val::Int(1)),
            const1.clone(),
            Wire::none(),
        )
        .unwrap();

        // wire15 = vars[0] + const1
        let wire15 = ctx.tmp_var(Type::Int).clone();
        let reads = ctx.get("x").union(&const1).unwrap();
        let sum = construct(Instruction::Sum, wire15.clone(), reads).unwrap();

        // wire5 = ite(wire12, wire15, const0)
        let reads = wire12.union(&wire15).unwrap().union(&const0).unwrap();
        let ite = construct(Instruction::Ite, ctx.get_cloned("x'"), reads).unwrap();

        // y' := y
        let id_y = construct(Instruction::Id, ctx.get_cloned("y'"), ctx.get_cloned("y")).unwrap();
        let id_z = construct(Instruction::Id, ctx.get_cloned("z'"), ctx.get_cloned("z")).unwrap();

        vec![xlty, xltz, or, term0, term1, sum, ite, id_y, id_z]
    }

    #[test]
    fn toy_example() {
        let mut ctx = Context::new();

        // create variables
        ctx.vars(
            Type::Int,
            vec!["x", "y", "z", "y0", "z0", "x'", "y'", "z'", "y0'", "z0'"],
        );
        const NEXT_OFFSET: isize = 5;

        // build module
        let module = build_module(&mut ctx, NEXT_OFFSET);
        dbg!(module);

        let prop = build_prop(&mut ctx);
        dbg!(prop);
    }

    fn build_module(ctx: &mut Context, next_offset: isize) -> Module<Type, Instruction> {
        let init_terms = init(ctx);
        let update_terms = update(ctx);

        let latched = ctx.get_vars(vec!["x", "y", "z", "y0", "z0"]);
        let next = latched
            .twin(next_offset)
            .expect("Failed getting primed variables");

        let atom =
            Atom::with_module_wire(&[latched.clone(), next.clone()], init_terms, update_terms)
                .expect("failed creating atom");

        Module::with_atoms([latched, next], vec![atom]).expect("Failed building module")
    }

    fn build_prop(ctx: &mut Context) -> Vec<Term> {
        let reads = ctx.get_vars(vec!["x", "y"]);
        let wire16 = ctx.tmp_var(Type::Bool).clone();
        let xeqy = construct(Instruction::Eq, wire16.clone(), reads).unwrap();

        let reads = ctx.get_cloned("x").union(ctx.get("z")).unwrap();
        let wire17 = ctx.tmp_var(Type::Bool).clone();
        let xeqz = construct(Instruction::Eq, wire17.clone(), reads).unwrap();

        let out = Wire::one(18, Type::Bool);
        let or = construct(Instruction::Or, out, wire16.union(&wire17).unwrap())
            .expect("Failed creating term");

        vec![xeqy, xeqz, or]
    }
}
