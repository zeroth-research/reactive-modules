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

    Module::sequential(obs, init, update)
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

    let m = Module::partially_observable(obs, prvt, m.atoms().iter().cloned());
    //print!("{}", m);

    assert!(m.is_err());
}

#[test]
#[allow(clippy::vec_init_then_push)]
fn example_peterson1() {
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
    let m = Module::sequential(obs, init, update).unwrap();

    assert!(m.is_open());
    assert_eq!(m.extl()[0].len(), 2);
    assert_eq!(m.intf()[0].len(), 2);
}
