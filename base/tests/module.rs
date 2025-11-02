use base::atom::Atom;
use base::module::Module;
use base::term::Term;
use base::wire::Wire;
use base::wires;

#[test]
fn can_instantiate_sequential_module() {
    let x = Wire::one(0, "real");
    let y = Wire::one(1, "real");
    let z = Wire::one(2, "real");
    let y0 = Wire::one(3, "real_nneg");
    let z0 = Wire::one(4, "real_nneg");

    let x_next = Wire::one(5, "real");
    let y_next = Wire::one(6, "real");
    let z_next = Wire::one(7, "real");
    let y0_next = Wire::one(8, "real_nneg");
    let z0_next = Wire::one(9, "real_nneg");

    let ctrl: Wire<&str> = wires![&x_next, &y_next, &z_next];

    let wait = Wire::try_from_iter([&y0_next, &z0_next].into_iter().flatten().cloned()).unwrap();

    let read: Wire<&str> = wires![&x, &y, &z];
    //let read_wait: Wire<&str> = wires![&read, &wait];

    let init_term = Term::new("SEE", ctrl.clone(), wait.clone());
    let update_term = Term::new("SEE", ctrl.clone(), read.clone());

    let latched = wires![x, y, z, y0, z0];
    let next = wires![x_next, y_next, z_next, y0_next, z0_next];

    let wire = [latched, next];

    let _module = Module::sequential(wire, [init_term], [update_term]).unwrap();

    //print!("{}", _module);
}

#[test]
fn can_instantiate_partially_observable_module() {
    let x = Wire::one(0, "real");
    let y = Wire::one(1, "real");
    let z = Wire::one(2, "real");
    let y0 = Wire::one(3, "real_nneg");
    let z0 = Wire::one(4, "real_nneg");

    let x_next = Wire::one(5, "real");
    let y_next = Wire::one(6, "real");
    let z_next = Wire::one(7, "real");
    let y0_next = Wire::one(8, "real_nneg");
    let z0_next = Wire::one(9, "real_nneg");

    let ctrl: Wire<&str> = wires![&x_next, &y_next, &z_next];

    let wait = Wire::try_from_iter([&y0_next, &z0_next].into_iter().flatten().cloned()).unwrap();

    let read: Wire<&str> = wires![&x, &y, &z];
    //let read_wait: Wire<&str> = wires![&read, &wait];

    let init_term = Term::new("SEE", ctrl.clone(), wait.clone());
    let update_term = Term::new("SEE", ctrl.clone(), read.clone());

    let obs_0 = wires![x, y, y0, z0];
    let obs_1 = wires![x_next, y_next, y0_next, z0_next];
    let prvt_0 = wires![z];
    let prvt_1 = wires![z_next];
    let latched = wires![&obs_0, &prvt_0];
    let next = wires![&obs_1, &prvt_1];

    let obs = [obs_0, obs_1];
    let prvt = [prvt_0, prvt_1];
    let wires = [latched, next];

    let atom = Atom::with_module_wire(&wires, vec![init_term], vec![update_term]).unwrap();
    let _module = Module::partially_observable(obs, prvt, [atom]).unwrap();

    //print!("{}", _module);
}

#[test]
fn cannot_instantiate_external_unobservable_wire() {
    let x = Wire::one(0, "real");
    let y = Wire::one(1, "real");
    let z = Wire::one(2, "real");
    let y0 = Wire::one(3, "real_nneg");
    let z0 = Wire::one(4, "real_nneg");

    let x_next = Wire::one(5, "real");
    let y_next = Wire::one(6, "real");
    let z_next = Wire::one(7, "real");
    let y0_next = Wire::one(8, "real_nneg");
    let z0_next = Wire::one(9, "real_nneg");

    let ctrl: Wire<&str> = wires![&x_next, &y_next, &z_next];

    let wait = Wire::try_from_iter([&y0_next, &z0_next].into_iter().flatten().cloned()).unwrap();

    let read: Wire<&str> = wires![&x, &y, &z];
    //let read_wait: Wire<&str> = wires![&read, &wait];

    let init_term = Term::new("SEE", ctrl.clone(), wait.clone());
    let update_term = Term::new("SEE", ctrl.clone(), read.clone());

    let obs_0 = wires![x, y];
    let obs_1 = wires![x_next, y_next];
    let prvt_0 = wires![z, y0, z0];
    let prvt_1 = wires![z_next, y0_next, z0_next];
    let latched = wires![&obs_0, &prvt_0];
    let next = wires![&obs_1, &prvt_1];

    let obs = [obs_0, obs_1];
    let prvt = [prvt_0, prvt_1];
    let wires = [latched, next];

    let atom = Atom::with_module_wire(&wires, vec![init_term], vec![update_term]).unwrap();
    let _module = Module::partially_observable(obs, prvt, [atom]);

    assert!(_module.is_err());
}
