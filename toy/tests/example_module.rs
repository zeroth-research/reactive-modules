use base::wire::Interface;
use toy::term::*;
use toy::val::Val;

use toy::{DType, ToyContext, ToyModule};

use std::iter::zip;

fn concat_intf<D: Eq + Clone>(items: &[&Interface<D>]) -> Interface<D> {
    let i = items.iter().flat_map(|intf| intf.wires()).cloned();
    Interface::sequence(i).unwrap()
}

fn init(ctx: &mut ToyContext) -> Vec<Term> {
    let init_x = mk_const(&Val::Int(0), ctx.get_intf(&["x'"]).unwrap()).unwrap();
    let init_y = mk_id(
        ctx.get_intf(&["y0'"]).unwrap(),
        ctx.get_intf(&["y'"]).unwrap(),
    )
    .unwrap();
    let init_z = mk_id(
        ctx.get_intf(&["z0'"]).unwrap(),
        ctx.get_intf(&["z'"]).unwrap(),
    )
    .unwrap();

    vec![init_x, init_y, init_z]
}

fn update(ctx: &mut ToyContext) -> Vec<Term> {
    // wire10 = x < y
    let reads = ctx.get_intf(&["x", "y"]).unwrap();
    let wire10 = ctx.tmp_intf(DType::Bool).clone();
    let xlty = mk_lt(reads, wire10.clone()).unwrap();

    // wire11 = x < z
    let reads = ctx.get_intf(&["x", "z"]).unwrap();
    let wire11 = ctx.tmp_intf(DType::Bool);
    let xltz = mk_lt(reads, wire11.clone()).unwrap();

    // wire12 = wire10 || wire11
    let wire12 = ctx.tmp_intf(DType::Bool);
    let reads = concat_intf(&[&wire10, &wire11]);
    let or = mk_or(reads, wire12.clone()).unwrap();

    // zero
    let const0 = ctx.tmp_intf(DType::Int).clone();
    let term0 = mk_const(&Val::Int(0), const0.clone()).unwrap();

    // one
    let const1 = ctx.tmp_intf(DType::Int).clone();
    let term1 = mk_const(&Val::Int(1), const1.clone()).unwrap();

    // wire15 = vars[0] + const1
    let wire15 = ctx.tmp_intf(DType::Int).clone();
    let x = ctx.get_intf(&["x"]).unwrap();
    let reads = concat_intf(&[&x, &const1]);
    let sum = mk_add(reads, wire15.clone()).unwrap();

    // wire5 = ite(wire12, wire15, const0)
    let reads = concat_intf(&[&wire12, &wire15, &const0]);
    let ite = mk_ite(reads, ctx.get_intf(&["x'"]).unwrap()).unwrap();

    // y' := y
    let id_y = mk_id(
        ctx.get_intf(&["y"]).unwrap(),
        ctx.get_intf(&["y'"]).unwrap(),
    )
    .unwrap();
    let id_z = mk_id(
        ctx.get_intf(&["z"]).unwrap(),
        ctx.get_intf(&["z'"]).unwrap(),
    )
    .unwrap();

    vec![xlty, xltz, or, term0, term1, sum, ite, id_y, id_z]
}

pub fn build_module(ctx: &mut ToyContext) -> ToyModule {
    let init_terms = init(ctx);
    let update_terms = update(ctx);

    let latched = ctx.get_intf(&["x", "y", "z", "y0", "z0"]).unwrap();
    let next = ctx.get_intf(&["x'", "y'", "z'", "y0'", "z0'"]).unwrap();

    ToyModule::sequential(
        zip(latched, next).map(|([l], [n])| [l, n]),
        init_terms,
        update_terms,
    )
    .expect("Failed building module")
}

pub fn _build_prop(ctx: &mut ToyContext) -> Vec<Term> {
    let reads = ctx.get_intf(&["x", "y"]).unwrap();
    let wire16 = ctx.tmp_intf(DType::Bool).clone();
    let xeqy = mk_eq(reads, wire16.clone()).unwrap();

    let reads = ctx.get_intf(&["x", "z"]).unwrap();
    let wire17 = ctx.tmp_intf(DType::Bool).clone();
    let xeqz = mk_eq(reads, wire17.clone()).unwrap();

    let out = ctx.tmp_intf(DType::Bool);
    let or = mk_or(concat_intf(&[&wire16, &wire17]), out).expect("Failed creating term");

    vec![xeqy, xeqz, or]
}
