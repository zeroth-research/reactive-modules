use base::atom::Atom;
use base::module::Module;
use base::term::Term;
use base::wire::Wire;
use base::wires;

#[test]
fn cross_check_atom_with_module_wire_and_module_with_atoms() {
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
    let read_wait: Wire<&str> = wires![&x, &y, &z, &y0, &z0];

    let init_term = Term::new("see report", ctrl.clone(), wait.clone());
    let update_term = Term::new("see report", ctrl.clone(), read_wait.clone());

    let latched = wires![&x, &y, &z, &y0, &z0];
    let next = wires![&x_next, &y_next, &z_next, &y0_next, &z0_next];

    let wire = [latched, next];

    let atom = Atom::with_module_wire(&wire, vec![init_term], vec![update_term]).unwrap();

    let _module = Module::with_atoms(wire, vec![atom]).unwrap();
}
