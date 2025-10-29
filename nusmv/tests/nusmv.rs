use base::{atom::Atom, module::Module, term::Term, wire::Wire};
use nusmv::{dtype::DType, itype::IType, nusmv::parse_nusmv};

#[test]
fn parse_boolean_model() {
    let input = r#"
        MODULE main
        VAR
            x : boolean;
            y : boolean;
        ASSIGN
            init(x) := TRUE;
            next(x) := y & !x;
    "#;

    let module = parse_nusmv(input);
    assert!(module.is_ok(), "Error: {:?}", module.unwrap_err());
    println!("Boolean module built successfully!");
    println!("{:#?}", module);
}

#[test]
fn parse_integer_model() {
    let input = r#"
        MODULE main
        VAR
            z : integer;
        ASSIGN
            init(z) := 0;
            next(z) := 1;
    "#;

    let module = parse_nusmv(input);
    assert!(module.is_ok(), "Error: {:?}", module.unwrap_err());
    println!("Integer module built successfully!");
    println!("{:#?}", module);
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
            next(x) := x + 1;
            next(y) := y;
            next(z) := z;
    "#;

    let module = parse_nusmv(input);
    assert!(module.is_ok(), "Error: {:?}", module.unwrap_err());
    println!("Parsed module: {:#?}", module);
}

// #[test]
// fn counter_nusmv2() {    
//     let input = r#"
//         MODULE main
//         IVAR
//             y0 : integer;
//             z0 : integer;
//         VAR
//             x : integer;
//             y : integer;
//             z : integer;
//         ASSIGN
//             init(x) := 0;
//             init(y) := y0;
//             init(z) := y0;
//             (x < y | x < z) ? next(x) := x + 1 : next(x) := 0;
//             next(y) := y;
//             next(z) := z;
//     "#;

//     let module = parse_nusmv(input);
//     assert!(module.is_ok(), "Error: {:?}", module.unwrap_err());
//     println!("Parsed module: {:#?}", module);
// }

#[test]
fn counter_core() {
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

    let module = Module::new((latched, next).into(), vec![atom]);
    assert!(module.is_ok(), "Error: {:?}", module.unwrap_err());
    println!("Module: {:#?}", module);
}