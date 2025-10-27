use base::atom::Atom;
use base::module::Module;
use base::term::Term;
use base::wire::Wire;

#[test]
fn can_instantiate_module() {
    let x = Wire::scalar(0, "real");
    let y = Wire::scalar(1, "real");
    let z = Wire::scalar(2, "real");
    let y0 = Wire::scalar(3, "real_nneg");
    let z0 = Wire::scalar(4, "real_nneg");

    let wait = Wire::union(&y0, &z0).unwrap();
    let ctrl = x.union(&y).unwrap().union(&z).unwrap();
    let read = ctrl.clone();

    let init_term = Term::new(
        "w[0] = 0, w[1] = a[0], w[2] = a[1]",
        ctrl.clone(),
        wait.clone(),
    );
    let update_term = Term::new("see report", ctrl.clone(), read.union(&wait).unwrap());

    let atom = Atom::new_unchecked(
        ctrl.twin(5).unwrap(),
        wait.twin(5).unwrap(),
        read,
        vec![init_term],
        vec![update_term],
    );

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

    let module = Module::new((latched, next).into(), vec![atom]);
    assert!(module.is_ok(), "Error: {:?}", module.unwrap_err());
}
