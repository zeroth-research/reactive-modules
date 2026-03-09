use base::{atom::Atom, module::Module, term::Term, wire::Wire};
use smt::dtype::DType;
use smt::itype::{ArithOp, CmpOp, IType, LogicalOp, Val};
use smt::smt::{assert_terms, declare};

pub fn create_test_module() -> Module<DType, IType> {
    let w0 = Wire::new(DType::Real);
    let w1 = Wire::new(DType::Int);
    let w2 = Wire::new(DType::Bool);
    let w3 = Wire::new(DType::Real);
    let w4 = Wire::new(DType::Int);
    let w5 = Wire::new(DType::Bool);
    let w6 = Wire::new(DType::Real);
    let w7 = Wire::new(DType::Int);
    let w8 = Wire::new(DType::Bool);
    let w9 = Wire::new(DType::Real);
    let w10 = Wire::new(DType::Int);
    let w11 = Wire::new(DType::Bool);
    // Interface wires: w0-w2 (latched), w6-w8 (next)
    let intf_ltc = vec![w0.clone(), w1.clone(), w2.clone()];
    let intf_nxt = vec![w6.clone(), w7.clone(), w8.clone()];

    // External wires: w3-w5 (latched), w9-w11 (next)
    let extl_ltc = vec![w3.clone(), w4.clone(), w5.clone()];
    let extl_nxt = vec![w9.clone(), w10.clone(), w11.clone()];

    // Private wires: none
    let prvt_ltc: Vec<Wire<DType>> = vec![];
    let prvt_nxt: Vec<Wire<DType>> = vec![];

    // All wires in the module (separate latched and next)
    let mut all_ltc = Vec::new();
    all_ltc.extend_from_slice(&extl_ltc);
    all_ltc.extend_from_slice(&intf_ltc);
    all_ltc.extend_from_slice(&prvt_ltc);
    let mut all_nxt = Vec::new();
    all_nxt.extend_from_slice(&extl_nxt);
    all_nxt.extend_from_slice(&intf_nxt);
    all_nxt.extend_from_slice(&prvt_nxt);

    // Observable wires: intf + extl
    let mut obs_ltc: Vec<Wire<DType>> = Vec::new();
    obs_ltc.extend_from_slice(&intf_ltc);
    obs_ltc.extend_from_slice(&extl_ltc);
    let mut obs_nxt: Vec<Wire<DType>> = Vec::new();
    obs_nxt.extend_from_slice(&intf_nxt);
    obs_nxt.extend_from_slice(&extl_nxt);

    let mut init = vec![];
    let mut update = vec![];

    // === INIT FLOW ===
    // w12 = 3.24, w13 = 42, w14 = true (constants)
    // w6 = w9 + w12, w7 = w10 - w13, w8 = w11 && w14 (outputs)
    let w12 = Wire::new(DType::Real);
    let w13 = Wire::new(DType::Int);
    let w14 = Wire::new(DType::Bool);
    init.push(Term::function(IType::Num(Val::Real(3.24)), [w12.clone()], vec![] as Vec<Wire<DType>>).unwrap());
    init.push(Term::function(IType::Num(Val::Int(42)), [w13.clone()], vec![] as Vec<Wire<DType>>).unwrap());
    init.push(Term::function(IType::Num(Val::Bool(true)), [w14.clone()], vec![] as Vec<Wire<DType>>).unwrap());
    init.push(Term::function(IType::Arith(ArithOp::Add), [w6.clone()], [w9.clone(), w12.clone()]).unwrap());
    init.push(Term::function(IType::Arith(ArithOp::Sub), [w7.clone()], [w10.clone(), w13.clone()]).unwrap());
    init.push(Term::function(IType::Logical(LogicalOp::And), [w8.clone()], [w11.clone(), w14.clone()]).unwrap());

    // === UPDATE FLOW ===
    // Constants and intermediates leading to w6, w7, w8
    let w15 = Wire::new(DType::Real);
    let w16 = Wire::new(DType::Real);
    let w17 = Wire::new(DType::Real);
    let w18 = Wire::new(DType::Real);
    let w19 = Wire::new(DType::Bool);
    let w20 = Wire::new(DType::Int);
    let w21 = Wire::new(DType::Bool);
    let w22 = Wire::new(DType::Bool);
    let w23 = Wire::new(DType::Bool);
    let w24 = Wire::new(DType::Bool);
    update.push(Term::function(IType::Num(Val::Real(4.20)), [w15.clone()], vec![] as Vec<Wire<DType>>).unwrap());
    update.push(Term::function(IType::Num(Val::Real(12.3)), [w16.clone()], vec![] as Vec<Wire<DType>>).unwrap());
    update.push(Term::function(IType::Arith(ArithOp::Mul), [w17.clone()], [w0.clone(), w15.clone()]).unwrap());
    update.push(Term::function(IType::Arith(ArithOp::Div), [w6.clone()], [w16.clone(), w17.clone()]).unwrap());
    update.push(Term::function(IType::Id, [w7.clone()], [w1.clone()]).unwrap());
    update.push(Term::function(IType::Num(Val::Real(50.05)), [w18.clone()], vec![] as Vec<Wire<DType>>).unwrap());
    update.push(Term::function(IType::Cmp(CmpOp::Lt), [w19.clone()], [w0.clone(), w18.clone()]).unwrap());
    update.push(Term::function(IType::Num(Val::Int(0)), [w20.clone()], vec![] as Vec<Wire<DType>>).unwrap());
    update.push(Term::function(IType::Cmp(CmpOp::Eq), [w21.clone()], [w1.clone(), w20.clone()]).unwrap());
    update.push(Term::function(IType::Logical(LogicalOp::Or), [w22.clone()], [w19.clone(), w21.clone()]).unwrap());
    update.push(Term::function(IType::Logical(LogicalOp::Not), [w23.clone()], [w2.clone()]).unwrap());
    update.push(Term::function(IType::Num(Val::Bool(false)), [w24.clone()], vec![] as Vec<Wire<DType>>).unwrap());
    update.push(Term::function(IType::Cond, [w8.clone()], [w22.clone(), w23.clone(), w24.clone()]).unwrap());

    let obs_pairs = obs_ltc.iter().zip(obs_nxt.iter()).map(|(l, n)| [l.clone(), n.clone()]);
    let prvt_pairs = prvt_ltc.iter().zip(prvt_nxt.iter()).map(|(l, n)| [l.clone(), n.clone()]);
    let atom = Atom::sequential(all_ltc.iter(), all_nxt.iter(), init, update).unwrap();
    Module::new(obs_pairs, prvt_pairs, [atom]).unwrap()
}

#[test]
fn test_declare() {
    let wires = vec![
        Wire::new(DType::Real),
        Wire::new(DType::Int),
        Wire::new(DType::Bool),
    ];
    let mut out = String::new();
    declare(&wires, &mut out).unwrap();

    // Check format: (declare-fun wN () Sort) for each wire
    let id0 = wires[0].id();
    let id1 = wires[1].id();
    let id2 = wires[2].id();
    assert!(out.contains(&format!("(declare-fun w{} () Real)", id0)));
    assert!(out.contains(&format!("(declare-fun w{} () Int)", id1)));
    assert!(out.contains(&format!("(declare-fun w{} () Bool)", id2)));
}

#[test]
fn test_assert_init() {
    let module = create_test_module();
    let atom = &module.atoms()[0];

    let mut out = String::new();
    assert_terms(atom.init().iter(), &mut out).unwrap();

    // Should contain let-bindings for intermediates and equalities for outputs
    assert!(out.contains("3.24"), "expected constant 3.24, got:\n{}", out);
    assert!(out.contains("42"), "expected constant 42, got:\n{}", out);
    assert!(out.contains("true"), "expected constant true, got:\n{}", out);
    assert!(out.contains("(assert"), "expected assert wrapper, got:\n{}", out);
}

#[test]
fn test_assert_update() {
    let module = create_test_module();
    let atom = &module.atoms()[0];

    let mut out = String::new();
    assert_terms(atom.update().iter(), &mut out).unwrap();

    // Check structural properties of update terms
    assert!(out.contains("4.2"), "expected constant 4.2, got:\n{}", out);
    assert!(out.contains("12.3"), "expected constant 12.3, got:\n{}", out);
    assert!(out.contains("50.05"), "expected constant 50.05, got:\n{}", out);
    assert!(out.contains("(assert"), "expected assert wrapper, got:\n{}", out);
}

#[test]
fn test_assert_bad_order() {
    // Term 0 reads `shared`, term 1 writes `shared` -> bad order
    let shared = Wire::new(DType::Real);
    let out_wire = Wire::new(DType::Real);
    let input = Wire::new(DType::Real);
    let terms = vec![
        Term::function(
            IType::Arith(ArithOp::Add),
            [out_wire],
            [input, shared.clone()],
        ).unwrap(),
        Term::function(
            IType::Num(Val::Real(5.0)),
            [shared],
            vec![] as Vec<Wire<DType>>,
        ).unwrap(),
    ];

    let mut out = String::new();
    let result = assert_terms(terms.iter(), &mut out);
    assert!(result.is_err(), "expected BadOrder error");
}

#[test]
fn test_assert_duplicate_write() {
    // Both terms write the same wire
    let shared = Wire::new(DType::Real);
    let terms = vec![
        Term::function(
            IType::Num(Val::Real(1.0)),
            [shared.clone()],
            vec![] as Vec<Wire<DType>>,
        ).unwrap(),
        Term::function(
            IType::Num(Val::Real(2.0)),
            [shared],
            vec![] as Vec<Wire<DType>>,
        ).unwrap(),
    ];

    let mut out = String::new();
    let result = assert_terms(terms.iter(), &mut out);
    assert!(result.is_err(), "expected DuplicateWrite error");
}
