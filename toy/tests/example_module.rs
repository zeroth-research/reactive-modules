use toy::context::Context;

use base::{Atom, Module};

use toy::dtype::Type;
use toy::instruction::Instruction;
use toy::term::*;
use toy::val::Val;

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

pub fn build_module(ctx: &mut Context) -> Module<Type, Instruction> {
    let init_terms = init(ctx);
    let update_terms = update(ctx);

    let latched = ctx.get_vars(&["x", "y", "z", "y0", "z0"]);
    let next = ctx.get_vars(&["x'", "y'", "z'", "y0'", "z0'"]);

    let atom = Atom::with_module_wire(&[latched.clone(), next.clone()], init_terms, update_terms)
        .expect("failed creating atom");

    Module::observable([latched, next], vec![atom]).expect("Failed building module")
}

pub fn _build_prop(ctx: &mut Context) -> Vec<Term> {
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
