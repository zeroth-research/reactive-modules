use base::{atom::Atom, module::Module, term::Term, wire::Wire};
use nusmv::{dtype::DType, itype::IType, nusmv::parse_nusmv};
use nusmv::debug_utils::{dump_debug, compare_debug};

fn build_manual_module() -> Module<DType, IType> {
    let x = Wire::one(0, DType::Int);
    let y = Wire::one(1, DType::Int);
    let z = Wire::one(2, DType::Int);
    let y0 = Wire::one(3, DType::Int);
    let z0 = Wire::one(4, DType::Int);

    let wait = Wire::union(&y0, &z0).unwrap();
    let ctrl = x.union(&y).unwrap().union(&z).unwrap();
    let read = ctrl.clone();

    let init_term = Term::new(IType::ConstInt(0), ctrl.clone(), wait.clone());
    let update_term = Term::new(IType::ConstInt(0), ctrl.clone(), read.union(&wait).unwrap());

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

    Module::with_atoms([latched, next], vec![atom]).unwrap()
}

#[test]
fn counter_nusmv() {
    let input = r#"
        MODULE main
        IVAR
            y0 : integer;
            z0 : integer;
        VAR
            x : integer;
            y : integer;
            z : integer;
        ASSIGN
            init(x) := 0;
            init(y) := y0;
            init(z) := y0;
            next(x) := (x < y | x < z) ? x + 1 : 0;
            next(y) := y;
            next(z) := z;
    "#;

    let parsed_module = parse_nusmv(input).unwrap();
    let manual_module = build_manual_module();

    dump_debug("Parsed Module", &parsed_module);
    dump_debug("Manual Module", &manual_module);
    compare_debug("Parsed", &parsed_module, "Manual", &manual_module);
}