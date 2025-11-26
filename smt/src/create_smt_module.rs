use base::{atom::Atom, module::Module, term::Term, wire::Wire};
use crate::dtype::DType;
use crate::itype::{ArithOp, CmpOp, IType, LogicalOp, Val};

pub fn create_test_module() -> Module<DType, IType> {
    // Interface wires: w0-w2 (latched), w6-w8 (next)
    let intf_ltc = Wire::from_iter(vec![
        (0, DType::Real), (1, DType::Int), (2, DType::Bool),
    ]);
    let intf_nxt = Wire::from_iter(vec![
        (6, DType::Real), (7, DType::Int), (8, DType::Bool),
    ]);

    // External wires: w3-w5 (latched), w9-w11 (next)
    let extl_ltc = Wire::from_iter(vec![
        (3, DType::Real), (4, DType::Int), (5, DType::Bool),
    ]);
    let extl_nxt = Wire::from_iter(vec![
        (9, DType::Real), (10, DType::Int), (11, DType::Bool),
    ]);

    // Private wires: none
    let prvt_ltc = Wire::none();
    let prvt_nxt = Wire::none();

    // Controlled wires: intf + prvt (separate latched and next)
    let ctrl_ltc = intf_ltc.clone().extend(&prvt_ltc);
    let ctrl_nxt = intf_nxt.clone().extend(&prvt_nxt);

    // Observable wires: intf + extl (separate latched and next)
    let obs_ltc = intf_ltc.clone().extend(&extl_ltc);
    let obs_nxt = intf_nxt.clone().extend(&extl_nxt);

    // All wires in the module (separate latched and next)
    let all_ltc = extl_ltc.clone().extend(&intf_ltc).extend(&prvt_ltc);
    let all_nxt = extl_nxt.clone().extend(&intf_nxt).extend(&prvt_nxt);

    let mut init = vec![];
    let mut update = vec![];

    // === INIT FLOW ===
    init.push(Term::new(IType::Const(Val::Real(3.14)), Wire::one(12, DType::Real), Wire::none()));
    init.push(Term::new(IType::Const(Val::Int(42)), Wire::one(13, DType::Int), Wire::none()));
    init.push(Term::new(IType::Const(Val::Bool(true)), Wire::one(14, DType::Bool), Wire::none()));

    init.push(Term::new(IType::Arith(ArithOp::Add), Wire::one(6, DType::Real), Wire::from_iter(vec![(9, DType::Real), (12, DType::Real)])));
    init.push(Term::new(IType::Arith(ArithOp::Sub), Wire::one(7, DType::Int), Wire::from_iter(vec![(10, DType::Int), (13, DType::Int)])));
    init.push(Term::new(IType::Logical(LogicalOp::And), Wire::one(8, DType::Bool), Wire::from_iter(vec![(11, DType::Bool), (14, DType::Bool)])));

    // === UPDATE FLOW ===
    update.push(Term::new(IType::Const(Val::Real(4.20)), Wire::one(15, DType::Real), Wire::none()));
    update.push(Term::new(IType::Const(Val::Real(12.3)), Wire::one(16, DType::Real), Wire::none()));
    update.push(Term::new(IType::Arith(ArithOp::Mul), Wire::one(17, DType::Real), Wire::from_iter(vec![(0, DType::Real), (15, DType::Real)])));
    update.push(Term::new(IType::Arith(ArithOp::Div), Wire::one(6, DType::Real), Wire::from_iter(vec![(16, DType::Real), (17, DType::Real)])));
    

    update.push(Term::new(IType::Id, Wire::one(7, DType::Real), Wire::one(1, DType::Real)));

    update.push(Term::new(IType::Const(Val::Real(50.05)), Wire::one(18, DType::Real), Wire::none()));
    update.push(Term::new(IType::Cmp(CmpOp::Lt), Wire::one(19, DType::Bool), Wire::from_iter(vec![(0, DType::Real), (18, DType::Real)])));
    update.push(Term::new(IType::Const(Val::Int(0)), Wire::one(20, DType::Int), Wire::none()));
    update.push(Term::new(IType::Cmp(CmpOp::Eq), Wire::one(21, DType::Bool), Wire::from_iter(vec![(1, DType::Int), (20, DType::Int)])));
    update.push(Term::new(IType::Logical(LogicalOp::Or), Wire::one(22, DType::Bool), Wire::from_iter(vec![(19, DType::Bool), (21, DType::Bool)])));
    update.push(Term::new(IType::Logical(LogicalOp::Not), Wire::one(23, DType::Bool), Wire::one(2, DType::Bool)));
    update.push(Term::new(IType::Const(Val::Bool(false)), Wire::one(24, DType::Bool), Wire::none()));
    update.push(Term::new(IType::Cond, Wire::one(8, DType::Bool), Wire::from_iter(vec![(22, DType::Bool), (23, DType::Bool), (24, DType::Bool)])));

    let atom = Atom::new_unchecked(intf_nxt.clone(), extl_nxt.clone(), intf_ltc.clone(), init, update);
    
    Module::new_unchecked(
        [extl_ltc, extl_nxt],                // extl - external wires [latched, next]
        [intf_ltc, intf_nxt],                // intf - interface wires [latched, next]
        [prvt_ltc, prvt_nxt],                // prvt - private wires [latched, next]
        [obs_ltc, obs_nxt],                   // obs  - observable wires [latched, next]
        [ctrl_ltc, ctrl_nxt],                // ctrl - controlled wires [latched, next]
        [all_ltc, all_nxt],                  // wire - all wires [latched, next]
        vec![atom],
    )
}
