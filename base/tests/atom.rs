use base::atom::Atom;
use base::module::Module;
use base::term::Term;
use base::wire::Wire;

#[test]
fn can_instantiate_atom_from_module_wire() {
    let x = Wire::one(0, "real");
    let y = Wire::one(1, "real");
    let z = Wire::one(2, "real");
    let y0 = Wire::one(3, "real_nneg");
    let z0 = Wire::one(4, "real_nneg");

    let wait = Wire::union(&y0, &z0).unwrap().twin(5).unwrap();
    let read = x.union(&y).unwrap().union(&z).unwrap();
    let ctrl = read.twin(5).unwrap();

    let init_term = Term::new(
        "w[0] = 0, w[1] = a[0], w[2] = a[1]",
        ctrl.clone(),
        wait.clone(),
    );
    let update_term = Term::new("see report", ctrl.clone(), read.union(&wait).unwrap());

    let latched = x
        .union(&y)
        .unwrap()
        .union(&z)
        .unwrap()
        .union(&y0)
        .unwrap()
        .union(&z0)
        .unwrap();
    let next = latched.twin(5).unwrap();

    let _atom =
        Atom::with_module_wire(&[latched, next], vec![init_term], vec![update_term]).unwrap();
}

#[test]
fn cross_check_atom_with_module_wire_and_module_with_atoms() {
    let x = Wire::one(0, "real");
    let y = Wire::one(1, "real");
    let z = Wire::one(2, "real");
    let y0 = Wire::one(3, "real_nneg");
    let z0 = Wire::one(4, "real_nneg");

    let wait = Wire::union(&y0, &z0).unwrap().twin(5).unwrap();
    let read = x.union(&y).unwrap().union(&z).unwrap();
    let ctrl = read.twin(5).unwrap();

    let init_term = Term::new(
        "w[0] = 0, w[1] = a[0], w[2] = a[1]",
        ctrl.clone(),
        wait.clone(),
    );
    let update_term = Term::new("see report", ctrl.clone(), read.union(&wait).unwrap());

    let latched = x
        .union(&y)
        .unwrap()
        .union(&z)
        .unwrap()
        .union(&y0)
        .unwrap()
        .union(&z0)
        .unwrap();
    let next = latched.twin(5).unwrap();
    let wire = [latched, next];

    let atom = Atom::with_module_wire(&wire, vec![init_term], vec![update_term]).unwrap();

    let _module = Module::with_atoms(wire, vec![atom]).unwrap();
}
