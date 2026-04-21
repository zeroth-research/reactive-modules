use base::module::Module;
use base::term;
use base::term::Term;
use base::wire::Interface;
use base::wire::Wire;
use theory::Theory;

#[derive(Clone, Debug)]
#[allow(unused)]
struct Ops(&'static str);

impl Theory for Ops {
    type DType = &'static str;

    fn _check(&self, _read: &[Self::DType], _write: &[Self::DType]) -> Result<(), String> {
        Ok(())
    }
}

fn mk_op(name: &'static str) -> Ops {
    Ops(name)
}

#[allow(clippy::vec_init_then_push)]
fn example_counter() -> Result<Module<Ops>, &'static str> {
    let x0 = Wire::new("real");
    let y0 = Wire::new("real");
    let z0 = Wire::new("real");
    let y00 = Wire::new("real");
    let z00 = Wire::new("real");
    let x1 = Wire::new("real");
    let y1 = Wire::new("real");
    let z1 = Wire::new("real");
    let y01 = Wire::new("real");
    let z01 = Wire::new("real");

    let mut init: Vec<Term<Ops>> = Vec::new();

    let tmp1 = Wire::new("real");
    init.push(term!(mk_op("ZERO"), [tmp1.clone()])?);

    let tmp2 = Wire::new("bool");
    let tmp3 = Wire::new("bool");
    init.push(term!(mk_op("ID"), [x1.clone()], [tmp1.clone()])?);
    init.push(term!(mk_op("ABS"), [tmp2.clone()], [y01.clone()])?);
    init.push(term!(mk_op("ID"), [y1.clone()], [tmp2.clone()])?);
    init.push(term!(mk_op("ABS"), [tmp3.clone()], [z01.clone()])?);
    init.push(term!(mk_op("ID"), [z1.clone()], [tmp3.clone()])?);

    let mut update: Vec<Term<Ops>> = Vec::new();

    let tmp4 = Wire::new("bool");
    let tmp5 = Wire::new("bool");
    let tmp6 = Wire::new("bool");
    update.push(term!(mk_op("ZERO"), [tmp1.clone()])?);
    update.push(term!(
        mk_op("LEQ"),
        [tmp4.clone()],
        [x0.clone(), y0.clone()]
    )?);
    update.push(term!(
        mk_op("LEQ"),
        [tmp5.clone()],
        [x0.clone(), z0.clone()]
    )?);
    update.push(term!(
        mk_op("OR"),
        [tmp6.clone()],
        [tmp4.clone(), tmp5.clone()]
    )?);

    let tmp7 = Wire::new("real");
    let tmp8 = Wire::new("real");
    update.push(term!(mk_op("ONE"), [tmp7.clone()])?);
    update.push(term!(
        mk_op("ADD"),
        [tmp8.clone()],
        [x0.clone(), tmp7.clone()]
    )?);

    update.push(term!(
        mk_op("ITE"),
        [x1.clone()],
        [tmp6.clone(), tmp8.clone(), tmp1.clone()]
    )?);
    update.push(term!(mk_op("ID"), [y1.clone()], [y0.clone()])?);
    update.push(term!(mk_op("ID"), [z1.clone()], [z0.clone()])?);

    let obs = Interface::from_iter([[x0, x1], [y0, y1], [z0, z1], [y00, y01], [z00, z01]]);

    Module::sequential_observable(obs, init, update)
}

#[allow(clippy::vec_init_then_push)]
fn example_peterson1() -> Result<Module<Ops>, &'static str> {
    let stype = "{outCS, reqCS, inCS}";
    let pc1: [Wire<&str>; 2] = [Wire::new(stype), Wire::new(stype)];
    let x1: [Wire<&str>; 2] = [Wire::new("bool"), Wire::new("bool")].map(Into::into);
    let pc2: [Wire<&str>; 2] = [Wire::new(stype), Wire::new(stype)].map(Into::into);
    let x2: [Wire<&str>; 2] = [Wire::new("bool"), Wire::new("bool")].map(Into::into);

    let mut init: Vec<Term<Ops>> = Vec::new();
    init.push(term!(mk_op("CONST(outCS)"), [pc1[1].clone()]).unwrap());
    init.push(term!(mk_op("CONST(true)"), [x1[1].clone()]).unwrap());

    let mut update: Vec<Term<Ops>> = Vec::new();
    let out_cs = Wire::new(stype);
    let cond1 = Wire::new("bool");
    update.push(term!(mk_op("CONST(outCS)"), [out_cs.clone()]).unwrap());
    update.push(
        term!(
            mk_op("EQ"),
            [cond1.clone()],
            [out_cs.clone(), pc1[0].clone()]
        )
        .unwrap(),
    );

    let req_cs = Wire::new(stype);
    let cond2 = Wire::new("bool");
    update.push(term!(mk_op("CONST(reqCS)"), [req_cs.clone()]).unwrap());
    let tmp11 = Wire::new("bool");
    update.push(
        term!(
            mk_op("EQ"),
            [tmp11.clone()],
            [req_cs.clone(), pc1[0].clone()]
        )
        .unwrap(),
    );

    let tmp12 = Wire::new("bool");
    let tmp13 = Wire::new("bool");
    let tmp14 = Wire::new("bool");
    update.push(
        term!(
            mk_op("EQ"),
            [tmp12.clone()],
            [out_cs.clone(), pc2[0].clone()]
        )
        .unwrap(),
    );
    update.push(
        term!(
            mk_op("NEQ"),
            [tmp13.clone()],
            [x1[0].clone(), x2[0].clone()]
        )
        .unwrap(),
    );
    update.push(term!(mk_op("OR"), [tmp14.clone()], [tmp12.clone(), tmp13.clone()]).unwrap());
    update.push(
        term!(
            mk_op("AND"),
            [cond2.clone()],
            [tmp14.clone(), tmp11.clone()]
        )
        .unwrap(),
    );

    let in_cs = Wire::new(stype);
    let cond3 = Wire::new("bool");
    update.push(term!(mk_op("CONST(inCS)"), [in_cs.clone()]).unwrap());
    update.push(
        term!(
            mk_op("EQ"),
            [cond3.clone()],
            [in_cs.clone(), pc1[0].clone()]
        )
        .unwrap(),
    );

    let const_true = Wire::new("bool");
    update.push(term!(mk_op("CONST(true)"), [const_true.clone()]).unwrap());

    update.push(
        term!(
            mk_op("CASE"),
            [pc1[1].clone(), x1[1].clone()],
            [
                cond1,
                req_cs,
                x2[0].clone(),
                cond2,
                in_cs.clone(),
                x1[0].clone(),
                cond3,
                out_cs.clone(),
                x1[0].clone(),
                const_true,
                pc1[0].clone(),
                x1[0].clone(),
            ]
        )
        .unwrap(),
    );

    let obs = Interface::from_iter([pc1, x1, pc2, x2]);
    Module::sequential_observable(obs, init, update)
}

fn example_tiny1(
    external: [Wire<&'static str>; 2],
    interface: [Wire<&'static str>; 2],
    wait: bool,
) -> Result<Module<Ops>, &'static str> {
    let private = [Wire::new("Tny"), Wire::new("Tny")];
    let temp = Wire::new("Tny");

    let cons = Term::constant(mk_op("CONST"), [temp.clone()]).unwrap();

    let update = if wait {
        Term::function(
            mk_op("AWAIT"),
            [interface[1].clone(), private[1].clone()],
            [external[1].clone(), private[0].clone(), temp],
        )
        .unwrap()
    } else {
        Term::function(
            mk_op("SEQ"),
            [interface[1].clone(), private[1].clone()],
            [external[0].clone(), private[0].clone(), temp],
        )
        .unwrap()
    };

    let init = Term::constant(mk_op("INIT"), [interface[1].clone(), private[1].clone()]).unwrap();

    let obs = Interface::from_iter([external, interface]);
    let prvt = Interface::from_iter([private]);

    Module::sequential(obs, prvt, [init], [cons, update])
}

#[test]
fn can_instantiate_sequential_module() {
    let _module = example_counter().unwrap();
    //print!("{}", _module);
}

#[test]
fn can_instantiate_partially_observable_module() {
    let m = example_counter().unwrap();
    let wires = m.obs().clone();
    let mut obs: Vec<[Wire<&'static str>; 2]> = Vec::new();
    let mut prvt: Vec<[Wire<&'static str>; 2]> = Vec::new();
    for [ltc, nxt] in wires {
        if ltc.id() == 0 {
            prvt.push([ltc, nxt]);
        } else {
            obs.push([ltc, nxt]);
        }
    }
    let obs = Interface::from_iter(obs);
    let prvt = Interface::from_iter(prvt);

    let m = Module::partially_observable(obs, prvt, m.atoms().iter().cloned());
    //print!("{}", m);
    assert!(m.is_ok());
}

#[test]
fn cannot_instantiate_external_unobservable_wire() {
    let m = example_counter().unwrap();
    let wires = m.obs().clone();
    let obswire = wires.wire(0, 3).unwrap().clone();
    let mut obs: Vec<[Wire<&'static str>; 2]> = Vec::new();
    let mut prvt: Vec<[Wire<&'static str>; 2]> = Vec::new();
    for [ltc, nxt] in wires {
        if ltc.id() == obswire.id() {
            prvt.push([ltc, nxt]);
        } else {
            obs.push([ltc, nxt]);
        }
    }
    let obs = Interface::from_iter(obs);
    let prvt = Interface::from_iter(prvt);

    let m = Module::partially_observable(obs, prvt, m.atoms().iter().cloned());
    print!("{:?}", m);

    assert!(m.is_err());
}

#[test]
fn can_instantiate_example_peterson1() {
    let m = example_peterson1().unwrap();

    assert!(m.is_open());
    assert_eq!(m.extl().len(), 2);
    assert_eq!(m.intf().len(), 2);
}

#[test]
fn module_write_all_ctrl() {
    let x = Wire::new("real");
    let xn = Wire::new("real");
    let y = Wire::new("real");
    let yn = Wire::new("real");

    let x0 = Wire::new("real");
    let xn0 = Wire::new("real");

    let update: Vec<Term<Ops>> = [term!(mk_op("ID"), [xn.clone()], [x.clone()]).unwrap()].to_vec();

    let obs = Interface::from_iter([
        [x.clone(), xn.clone()],
        [y.clone(), yn.clone()],
        [x0, xn0.clone()],
    ]);

    let m = Module::sequential_observable(obs.clone(), vec![], update.clone());
    assert!(m.is_err_and(|msg| { msg == "unassigned control wire after init" }));

    let init: Vec<Term<Ops>> = [term!(mk_op("ID"), [xn0.clone()], [xn.clone()]).unwrap()].to_vec();
    let m = Module::sequential_observable(obs.clone(), init, update.clone());
    assert!(m.is_err_and(|msg| {
        dbg!(&msg);
        msg == "unassigned control wire after init"
    }));

    let init: Vec<Term<Ops>> = [
        term!(mk_op("ID"), [xn.clone()], [xn0.clone()]).unwrap(),
        term!(mk_op("ID"), [yn.clone()], [xn0]).unwrap(),
    ]
    .to_vec();

    let m = Module::sequential_observable(obs.clone(), init.clone(), update);
    assert!(m.is_err_and(|msg| {
        dbg!(&msg);
        msg == "unassigned control wire after update"
    }));

    let update: Vec<Term<Ops>> = [
        term!(mk_op("ID"), [xn.clone()], [x.clone()]).unwrap(),
        term!(mk_op("ID"), [yn], [y.clone()]).unwrap(),
    ]
    .to_vec();

    let m = Module::sequential_observable(obs.clone(), init, update);
    assert!(m.is_ok());
}

#[test]
fn can_compose_example_peterson1_with_empty_module() {
    let m1 = example_peterson1().unwrap();
    let m2 = Module::empty();

    let _m3 = Module::parallel([m1, m2]).unwrap();
}

#[test]
fn can_instantiate_example_tiny1_0123() {
    let x = [Wire::new("Tny"), Wire::new("Tny")];
    let y = [Wire::new("Tny"), Wire::new("Tny")];
    let m = example_tiny1(x, y, true).unwrap();
    assert!(m.is_open());
}

#[test]
fn can_instantiate_example_tiny1_2301() {
    let x = [Wire::new("Tny"), Wire::new("Tny")];
    let y = [Wire::new("Tny"), Wire::new("Tny")];
    let m = example_tiny1(y, x, true).unwrap();
    assert!(m.is_open());
}
#[test]
fn can_compose_example_tiny1() {
    let x = [Wire::new("Tny"), Wire::new("Tny")];
    let y = [Wire::new("Tny"), Wire::new("Tny")];
    let m1 = example_tiny1(x.clone(), y.clone(), false).unwrap();
    let m2 = example_tiny1(y, x, false).unwrap();

    let m3 = Module::parallel([m1, m2]);
    assert!(m3.is_ok());
}

#[test]
fn cannot_compose_example_tiny1_with_cyclic_await() {
    let x = [Wire::new("Tny"), Wire::new("Tny")];
    let y = [Wire::new("Tny"), Wire::new("Tny")];
    let m1 = example_tiny1(x.clone(), y.clone(), true).unwrap();
    let m2 = example_tiny1(y, x, true).unwrap();

    let m3 = Module::parallel([m1, m2]);
    assert!(m3.is_err());
}

#[test]
fn can_compose_example_tiny1_without_cyclic_await_and_overlapping_prvt() {
    let x = [Wire::new("Tny"), Wire::new("Tny")];
    let y = [Wire::new("Tny"), Wire::new("Tny")];
    let m1 = example_tiny1(x.clone(), y.clone(), true).unwrap();
    let m2 = example_tiny1(y, x, false).unwrap();

    let m3 = Module::parallel([m1, m2]);
    assert!(m3.is_ok());
}

#[test]
fn can_compose_three_tiny1_without_cyclic_await_and_overlapping_prvt() {
    let x = [Wire::new("Tny"), Wire::new("Tny")];
    let y = [Wire::new("Tny"), Wire::new("Tny")];
    let z = [Wire::new("Tny"), Wire::new("Tny")];
    let m1 = example_tiny1(x.clone(), y.clone(), true).unwrap();
    let m2 = example_tiny1(y.clone(), x.clone(), false).unwrap();
    let m3 = example_tiny1(y, z, false).unwrap();

    let m4 = Module::parallel([m1, m2, m3]);
    assert!(m4.is_ok());
}

#[test]
fn compose_seq() {
    // define two modules:
    //  M1: read external "y" and write it to "x"
    //  M2: read external "x" and write it to "z"
    //
    //  M1 and M2 are compatible (disjoint interface variables and acyclic waiting dependencies),
    //  and we test that they are composable
    let x = Wire::new("real");
    let xn = Wire::new("real");
    let y = Wire::new("real");
    let yn = Wire::new("real");
    let z = Wire::new("real");
    let zn = Wire::new("real");

    let assign: Vec<Term<Ops>> = [term!(mk_op("ID"), [xn.clone()], [yn.clone()]).unwrap()].to_vec();
    let obs = Interface::from_iter([[x.clone(), xn.clone()], [y.clone(), yn.clone()]]);
    let m1 = Module::combinatorial(obs.clone(), assign.clone()).unwrap();

    let assign: Vec<Term<Ops>> = [term!(mk_op("ID"), [zn.clone()], [xn.clone()]).unwrap()].to_vec();
    let obs = Interface::from_iter([[x.clone(), xn.clone()], [z.clone(), zn.clone()]]);
    let m2 = Module::combinatorial(obs.clone(), assign.clone()).unwrap();

    Module::parallel([m1, m2]).unwrap();
}

#[test]
fn compose_seq_2() {
    let (x, xn) = (Wire::new("real"), Wire::new("real"));
    let (y, yn) = (Wire::new("real"), Wire::new("real"));
    let (z, zn) = (Wire::new("real"), Wire::new("real"));
    let (y0, y0n) = (Wire::new("real"), Wire::new("real"));
    let (z0, z0n) = (Wire::new("real"), Wire::new("real"));
    let (inv, invn) = (Wire::new("real"), Wire::new("real"));

    // class Module(smt.Module):
    //     def init(self, extl) -> None:
    //         y0, z0 = extl
    //         return Int(0), nxt(y0), nxt(z0)  # = x, y, z
    //
    //     def update(self, ctrl, extl) -> None:
    //         x, y, z = ctrl
    //
    //         cond = Or(x < y, x < z)
    //         xn = Ite(cond, x + Int(1), Int(0))
    //
    //         return xn, y, z
    //
    let init: Vec<Term<Ops>> = [
        term!(mk_op("Const(0)"), [xn.clone()]).unwrap(),
        term!(mk_op("Id"), [yn.clone()], [y0n.clone()]).unwrap(),
        term!(mk_op("Id"), [zn.clone()], [z0n.clone()]).unwrap(),
    ]
    .to_vec();

    let tmps = [
        Wire::new("real"),
        Wire::new("real"),
        Wire::new("real"),
        Wire::new("real"),
        Wire::new("real"),
        Wire::new("real"),
    ];
    let update: Vec<Term<Ops>> = [
        term!(mk_op("Lt"), [tmps[0].clone()], [x.clone(), y.clone()]).unwrap(),
        term!(mk_op("Lt"), [tmps[1].clone()], [x.clone(), z.clone()]).unwrap(),
        term!(
            mk_op("Or"),
            [tmps[2].clone()],
            [tmps[0].clone(), tmps[1].clone()]
        )
        .unwrap(),
        term!(mk_op("Const(0)"), [tmps[3].clone()]).unwrap(),
        term!(mk_op("Const(1)"), [tmps[4].clone()]).unwrap(),
        term!(
            mk_op("Add"),
            [tmps[5].clone()],
            [x.clone(), tmps[4].clone()]
        )
        .unwrap(),
        term!(
            mk_op("Ite"),
            [xn.clone()],
            [tmps[2].clone(), tmps[5].clone(), tmps[3].clone()]
        )
        .unwrap(),
        term!(mk_op("Id"), [yn.clone()], [y.clone()]).unwrap(),
        term!(mk_op("Id"), [zn.clone()], [z.clone()]).unwrap(),
    ]
    .to_vec();
    let obs = Interface::from_iter([
        [x.clone(), xn.clone()],
        [y.clone(), yn.clone()],
        [z.clone(), zn.clone()],
        [y0.clone(), y0n.clone()],
        [z0.clone(), z0n.clone()],
    ]);
    let m1 = Module::sequential_observable(obs.clone(), init, update).unwrap();

    //
    // class Inv(smt.Module):
    //     def init(self, extl) -> None:
    //         x, y, z = extl
    //         return Or(nxt(x) <= nxt(y), nxt(x) <= nxt(z))
    //
    //     def update(self, inv, extl) -> None:
    //         x, y, z = extl
    //         return Or(nxt(x) <= nxt(y), nxt(x) <= nxt(z))
    let tmps = [Wire::new("real"), Wire::new("real"), Wire::new("real")];
    let assign: Vec<Term<Ops>> = [
        term!(mk_op("Le"), [tmps[0].clone()], [xn.clone(), yn.clone()]).unwrap(),
        term!(mk_op("Le"), [tmps[1].clone()], [xn.clone(), zn.clone()]).unwrap(),
        term!(
            mk_op("Or"),
            [tmps[2].clone()],
            [tmps[0].clone(), tmps[1].clone()]
        )
        .unwrap(),
        term!(mk_op("Id"), [invn.clone()], [tmps[2].clone()]).unwrap(),
    ]
    .to_vec();

    let obs = Interface::from_iter([
        [x.clone(), xn.clone()],
        [y.clone(), yn.clone()],
        [z.clone(), zn.clone()],
        [inv.clone(), invn.clone()],
    ]);
    let m2 = Module::combinatorial(obs.clone(), assign.clone()).unwrap();

    Module::parallel([m1.clone(), m2]).unwrap();

    // try to use a `sequential_observable` ctor instead of combinatorial
    let m2 = Module::sequential_observable(obs.clone(), assign.clone(), assign).unwrap();
    let _m = Module::parallel([m1, m2]).unwrap();
    println!("{:?}", _m);
}

#[test]
fn more_controlled_than_external() {
    let x = [Wire::new("A"), Wire::new("A")];
    let y = [Wire::new("B"), Wire::new("B")];
    let z = [Wire::new("C"), Wire::new("C")];

    let init = Term::constant(mk_op("A"), [y[1].clone(), z[1].clone()]).unwrap();
    let update = Term::function(
        mk_op("A"),
        [y[1].clone(), z[1].clone()],
        [y[0].clone(), z[0].clone()],
    )
    .unwrap();

    let m = Module::sequential([x], [y, z], [init], [update]);
    assert!(m.is_ok());
}

#[test]
fn module_with_invalid_read() {
    let x = [Wire::new("A"), Wire::new("A")];
    let y = [Wire::new("B"), Wire::new("B")];
    let z = [Wire::new("C"), Wire::new("C")];
    let p = Wire::new("P");

    let init = Term::function(
        mk_op("i"),
        [y[1].clone(), z[1].clone()],
        [p.clone(), x[1].clone()],
    )
    .unwrap();
    let update = Term::function(
        mk_op("u"),
        [y[1].clone(), z[1].clone()],
        [p.clone(), x[1].clone(), y[0].clone(), z[0].clone()],
    )
    .unwrap();

    let m = Module::sequential([x], [y, z], [init], [update]);
    println!("{:?}", m);
    assert!(m.is_err());
}
