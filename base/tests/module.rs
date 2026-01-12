use base::module::Module;
use base::term;
use base::term::Term;
use base::wire::Interface;
use base::wire::Wire;

#[allow(clippy::vec_init_then_push)]
fn example_counter() -> Result<Module<&'static str, &'static str>, &'static str> {
    let x0 = Wire::new(0, "real");
    let y0 = Wire::new(1, "real");
    let z0 = Wire::new(2, "real");
    let y00 = Wire::new(3, "real");
    let z00 = Wire::new(4, "real");
    let x1 = Wire::new(5, "real");
    let y1 = Wire::new(6, "real");
    let z1 = Wire::new(7, "real");
    let y01 = Wire::new(8, "real");
    let z01 = Wire::new(9, "real");

    let mut init: Vec<Term<&str, &str>> = Vec::new();

    init.push(term!("ZERO", [(10, "real")])?);

    init.push(term!("ID", [x1.clone()], [(10, "real")])?);
    init.push(term!("ABS", [(11, "bool")], [y01.clone()])?);
    init.push(term!("ID", [y1.clone()], [(11, "bool")])?);
    init.push(term!("ABS", [(12, "bool")], [z01.clone()])?);
    init.push(term!("ID", [z1.clone()], [(12, "bool")])?);

    let mut update: Vec<Term<&str, &str>> = Vec::new();

    update.push(term!("ZERO", [(10, "real")])?);
    update.push(term!("LEQ", [(13, "bool")], [x0.clone(), y0.clone()])?);
    update.push(term!("LEQ", [(14, "bool")], [x0.clone(), z0.clone()])?);
    update.push(term!("OR", [(15, "bool")], [(13, "bool"), (14, "bool")])?);

    update.push(term!("ONE", [(16, "real")])?);
    update.push(term!(
        "ADD",
        [(17, "real")],
        [x0.clone(), (16, "real").into()]
    )?);

    update.push(term!(
        "ITE",
        [x1.clone()],
        [(15, "bool"), (17, "real"), (10, "real")]
    )?);
    update.push(term!("ID", [y1.clone()], [y0.clone()])?);
    update.push(term!("ID", [z1.clone()], [z0.clone()])?);

    let obs = Interface::from_iter([[x0, x1], [y0, y1], [z0, z1], [y00, y01], [z00, z01]]);

    Module::sequential_observable(obs, init, update)
}

#[allow(clippy::vec_init_then_push)]
fn example_peterson1() -> Result<Module<&'static str, &'static str>, &'static str> {
    let stype = "{outCS, reqCS, inCS}";
    let pc1: [Wire<&str>; 2] = [(0, stype), (1, stype)].map(Into::into);
    let x1: [Wire<&str>; 2] = [(2, "bool"), (3, "bool")].map(Into::into);
    let pc2: [Wire<&str>; 2] = [(4, stype), (5, stype)].map(Into::into);
    let x2: [Wire<&str>; 2] = [(6, "bool"), (7, "bool")].map(Into::into);

    let mut init: Vec<Term<&str, &str>> = Vec::new();
    init.push(term!("CONST(outCS)", [pc1[1].clone()]).unwrap());
    init.push(term!("CONST(true)", [x1[1].clone()]).unwrap());

    let mut update: Vec<Term<&str, &str>> = Vec::new();
    let out_cs = Wire::new(8, stype);
    let cond1 = Wire::new(9, "bool");
    update.push(term!("CONST(outCS)", [out_cs.clone()]).unwrap());
    update.push(term!("EQ", [cond1.clone()], [out_cs.clone(), pc1[0].clone()]).unwrap());

    let req_cs = Wire::new(10, stype);
    let cond2 = Wire::new(15, "bool");
    update.push(term!("CONST(reqCS)", [req_cs.clone()]).unwrap());
    update.push(term!("EQ", [(11, "bool")], [req_cs.clone(), pc1[0].clone()]).unwrap());

    update.push(term!("EQ", [(12, "bool")], [out_cs.clone(), pc2[0].clone()]).unwrap());
    update.push(term!("NEQ", [(13, "bool")], [x1[0].clone(), x2[0].clone()]).unwrap());
    update.push(term!("OR", [(14, "bool")], [(12, "bool"), (13, "bool")]).unwrap());
    update.push(term!("AND", [cond2.clone()], [(14, "bool"), (11, "bool")]).unwrap());

    let in_cs = Wire::new(16, stype);
    let cond3 = Wire::new(17, "bool");
    update.push(term!("CONST(inCS)", [in_cs.clone()]).unwrap());
    update.push(term!("EQ", [cond3.clone()], [in_cs.clone(), pc1[0].clone()]).unwrap());

    let const_true = Wire::new(18, "bool");
    update.push(term!("CONST(true)", [const_true.clone()]).unwrap());

    update.push(
        term!(
            "CASE",
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
    external: (usize, usize),
    interface: (usize, usize),
    wait: bool,
    base: usize,
) -> Result<Module<&'static str, &'static str>, &'static str> {
    let external = [Wire::new(external.0, "Tny"), Wire::new(external.1, "Tny")];
    let interface = [Wire::new(interface.0, "Tny"), Wire::new(interface.1, "Tny")];
    let private = [Wire::new(base, "Tny"), Wire::new(base + 1, "Tny")];
    let temp = Wire::new(base + 2, "Tny");

    let cons = Term::constant("CONST", [temp.clone()]).unwrap();

    let update = if wait {
        Term::function(
            "AWAIT",
            [interface[1].clone(), private[1].clone()],
            [external[1].clone(), private[0].clone(), temp],
        )
        .unwrap()
    } else {
        Term::function(
            "SEQ",
            [interface[1].clone(), private[1].clone()],
            [external[0].clone(), private[0].clone(), temp],
        )
        .unwrap()
    };

    let init = Term::constant("INIT", [interface[1].clone(), private[1].clone()]).unwrap();

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

    let m = Module::new(obs, prvt, m.atoms().iter().cloned());
    //print!("{}", m);
    assert!(m.is_ok());
}

#[test]
fn cannot_instantiate_external_unobservable_wire() {
    let m = example_counter().unwrap();
    let wires = m.obs().clone();
    let mut obs: Vec<[Wire<&'static str>; 2]> = Vec::new();
    let mut prvt: Vec<[Wire<&'static str>; 2]> = Vec::new();
    for [ltc, nxt] in wires {
        if ltc.id() == 3 {
            prvt.push([ltc, nxt]);
        } else {
            obs.push([ltc, nxt]);
        }
    }
    let obs = Interface::from_iter(obs);
    let prvt = Interface::from_iter(prvt);

    let m = Module::new(obs, prvt, m.atoms().iter().cloned());
    //print!("{}", m);

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
    let x = Wire::new(0, "real");
    let xn = Wire::new(1, "real");
    let y = Wire::new(2, "real");
    let yn = Wire::new(3, "real");

    let x0 = Wire::new(4, "real");
    let xn0 = Wire::new(5, "real");

    let update: Vec<Term<&str, &str>> = [term!("ID", [xn.clone()], [x.clone()]).unwrap()].to_vec();

    let obs = Interface::from_iter([
        [x.clone(), xn.clone()],
        [y.clone(), yn.clone()],
        [x0, xn0.clone()],
    ]);

    let m = Module::sequential_observable(obs.clone(), vec![], update.clone());
    assert!(m.is_err_and(|msg| { msg == "unassigned control wire after init" }));

    let init: Vec<Term<&str, &str>> = [term!("ID", [xn0.clone()], [xn.clone()]).unwrap()].to_vec();
    let m = Module::sequential_observable(obs.clone(), init, update.clone());
    assert!(m.is_err_and(|msg| {
        dbg!(&msg);
        msg == "unassigned control wire after init"
    }));

    let init: Vec<Term<&str, &str>> = [
        term!("ID", [xn.clone()], [xn0.clone()]).unwrap(),
        term!("ID", [yn.clone()], [xn0]).unwrap(),
    ]
    .to_vec();

    let m = Module::sequential_observable(obs.clone(), init.clone(), update);
    assert!(m.is_err_and(|msg| {
        dbg!(&msg);
        msg == "unassigned control wire after update"
    }));

    let update: Vec<Term<&str, &str>> = [
        term!("ID", [xn.clone()], [x.clone()]).unwrap(),
        term!("ID", [yn], [y.clone()]).unwrap(),
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
    let m = example_tiny1((0, 1), (2, 3), true, 4).unwrap();
    assert!(m.is_open());
}

#[test]
fn can_instantiate_example_tiny1_2301() {
    let m = example_tiny1((2, 3), (0, 1), true, 4).unwrap();
    assert!(m.is_open());
}
#[test]
fn cannot_compose_example_tiny1_with_overlapping_prvt() {
    let m1 = example_tiny1((0, 1), (2, 3), false, 4).unwrap();
    let m2 = example_tiny1((2, 3), (0, 1), false, 4).unwrap();

    let m3 = Module::parallel([m1, m2]);
    assert!(m3.is_err());
}

#[test]
fn cannot_compose_example_tiny1_with_cyclic_await() {
    let m1 = example_tiny1((0, 1), (2, 3), true, 4).unwrap();
    let m2 = example_tiny1((2, 3), (0, 1), true, 7).unwrap();

    let m3 = Module::parallel([m1, m2]);
    assert!(m3.is_err());
}

#[test]
fn can_compose_example_tiny1_without_cyclic_await_and_overlapping_prvt() {
    let m1 = example_tiny1((0, 1), (2, 3), true, 4).unwrap();
    let m2 = example_tiny1((2, 3), (0, 1), false, 7).unwrap();

    let m3 = Module::parallel([m1, m2]);
    assert!(m3.is_ok());
}

#[test]
fn can_compose_three_tiny1_without_cyclic_await_and_overlapping_prvt() {
    let m1 = example_tiny1((0, 1), (2, 3), true, 4).unwrap();
    let m2 = example_tiny1((2, 3), (0, 1), false, 7).unwrap();
    let m3 = example_tiny1((2, 3), (10, 11), false, 12).unwrap();

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
    let x = Wire::new(0, "real");
    let xn = Wire::new(1, "real");
    let y = Wire::new(2, "real");
    let yn = Wire::new(3, "real");
    let z = Wire::new(4, "real");
    let zn = Wire::new(5, "real");

    let assign: Vec<Term<&str, &str>> = [term!("ID", [xn.clone()], [yn.clone()]).unwrap()].to_vec();
    let obs = Interface::from_iter([[x.clone(), xn.clone()], [y.clone(), yn.clone()]]);
    let m1 = Module::combinatorial(obs.clone(), assign.clone()).unwrap();

    let assign: Vec<Term<&str, &str>> = [term!("ID", [zn.clone()], [xn.clone()]).unwrap()].to_vec();
    let obs = Interface::from_iter([[x.clone(), xn.clone()], [z.clone(), zn.clone()]]);
    let m2 = Module::combinatorial(obs.clone(), assign.clone()).unwrap();

    Module::parallel([m1, m2]).unwrap();
}

#[test]
fn compose_seq_2() {
    let (x, xn) = (Wire::new(0, "real"), Wire::new(1, "real"));
    let (y, yn) = (Wire::new(2, "real"), Wire::new(3, "real"));
    let (z, zn) = (Wire::new(4, "real"), Wire::new(5, "real"));
    let (y0, y0n) = (Wire::new(6, "real"), Wire::new(7, "real"));
    let (z0, z0n) = (Wire::new(8, "real"), Wire::new(9, "real"));
    let (inv, invn) = (Wire::new(10, "real"), Wire::new(11, "real"));

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
    let init: Vec<Term<&str, &str>> = [
        term!("Const(0)", [xn.clone()]).unwrap(),
        term!("Id", [yn.clone()], [y0n.clone()]).unwrap(),
        term!("Id", [zn.clone()], [z0n.clone()]).unwrap(),
    ]
    .to_vec();

    let tmps = [
        Wire::new(12, "real"),
        Wire::new(13, "real"),
        Wire::new(14, "real"),
        Wire::new(15, "real"),
        Wire::new(16, "real"),
        Wire::new(17, "real"),
    ];
    let update: Vec<Term<&str, &str>> = [
        term!("Lt", [tmps[0].clone()], [x.clone(), y.clone()]).unwrap(),
        term!("Lt", [tmps[1].clone()], [x.clone(), z.clone()]).unwrap(),
        term!("Or", [tmps[2].clone()], [tmps[0].clone(), tmps[1].clone()]).unwrap(),
        term!("Const(0)", [tmps[3].clone()]).unwrap(),
        term!("Const(1)", [tmps[4].clone()]).unwrap(),
        term!("Add", [tmps[5].clone()], [x.clone(), tmps[4].clone()]).unwrap(),
        term!(
            "Ite",
            [xn.clone()],
            [tmps[2].clone(), tmps[5].clone(), tmps[3].clone()]
        )
        .unwrap(),
        term!("Id", [yn.clone()], [y.clone()]).unwrap(),
        term!("Id", [zn.clone()], [z.clone()]).unwrap(),
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
    let tmps = [
        Wire::new(18, "real"),
        Wire::new(19, "real"),
        Wire::new(20, "real"),
    ];
    let assign: Vec<Term<&str, &str>> = [
        term!("Le", [tmps[0].clone()], [xn.clone(), yn.clone()]).unwrap(),
        term!("Le", [tmps[1].clone()], [xn.clone(), zn.clone()]).unwrap(),
        term!("Or", [tmps[2].clone()], [tmps[0].clone(), tmps[1].clone()]).unwrap(),
        term!("Id", [invn.clone()], [tmps[2].clone()]).unwrap(),
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
    println!("{}", _m);
}
