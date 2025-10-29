use base::{atom::Atom, module::Module, term::Term, wire::Wire};
use nusmv::{dtype::DType, itype::IType, nusmv::parse_nusmv};

// #[test]
// fn parse_boolean_model() {
//     let input = r#"
//         MODULE main
//         VAR
//             x : boolean;
//             y : boolean;
//         ASSIGN
//             init(x) := TRUE;
//             next(x) := y & !x;
//     "#;

//     let module = parse_nusmv(input);
//     assert!(module.is_ok(), "Error: {:?}", module.unwrap_err());
//     println!("Boolean module built successfully!");
//     println!("{:#?}", module);
// }

// #[test]
// fn parse_integer_model() {
//     let input = r#"
//         MODULE main
//         VAR
//             z : integer;
//         ASSIGN
//             init(z) := 0;
//             next(z) := 1;
//     "#;

//     let module = parse_nusmv(input);
//     assert!(module.is_ok(), "Error: {:?}", module.unwrap_err());
//     println!("Integer module built successfully!");
//     println!("{:#?}", module);
// }

#[test]
fn simulate_base_module_shape() {
    // To match the base test structure, we need expressions that reference
    // the y0 and z0 variables so they appear in the 'read' wires.
    // The base test has ONE atom controlling {x,y,z} that reads {x,y,z,y0,z0}.
    
    let input = r#"
        MODULE main
        VAR
            x : integer;
            y : integer;
            z : integer;
            y0 : integer;
            z0 : integer;
        ASSIGN
            init(x) := 0;
            init(y) := next(y0);
            init(z) := next(y0);
            next(x) := x + 1;
            next(y) := y;
            next(z) := z;
    "#;

    let module = parse_nusmv(input);
    assert!(module.is_ok(), "Error: {:?}", module.unwrap_err());
    println!("Parsed module: {:#?}", module);
}

#[test]
fn simulate_base_module_shape_2() {
    // This test reproduces the shape of the module constructed in
    // `base/tests/module.rs` but using `DType::Int` for all wires so it
    // can be created from the `nusmv` crate types. The goal is to ensure
    // the same wiring/twin logic (including twin offset) works for
    // nusmv-built modules.

    let x = Wire::scalar(0, DType::Int);
    let y = Wire::scalar(1, DType::Int);
    let z = Wire::scalar(2, DType::Int);
    let y0 = Wire::scalar(3, DType::Int);
    let z0 = Wire::scalar(4, DType::Int);

    let wait = y0.union(&z0).unwrap();
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

    let module = Module::new([latched, next], vec![atom]);
    assert!(module.is_ok(), "Error: {:?}", module.unwrap_err());
    println!("Module: {:#?}", module);
}