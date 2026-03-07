use base::{atom::Atom, module::Module, term::Term, wire::Wire};
use smt::dtype::DType;
use smt::itype::{ArithOp, CmpOp, IType, LogicalOp, Val};

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

    // // External wires: w3-w5 (latched), w9-w11 (next)
    let extl_ltc = vec![w3.clone(), w4.clone(), w5.clone()];
    let extl_nxt = vec![w9.clone(), w10.clone(), w11.clone()];

    // // Private wires: none
    let prvt_ltc: Vec<Wire<DType>> = vec![];
    let prvt_nxt: Vec<Wire<DType>> = vec![];

    // // Controlled wires: intf + prvt (separate latched and next)
    let mut ctrl_ltc = Vec::new();
    ctrl_ltc.extend_from_slice(&intf_ltc);
    ctrl_ltc.extend_from_slice(&prvt_ltc);
    let mut ctrl_nxt = Vec::new();
    ctrl_nxt.extend_from_slice(&intf_nxt);
    ctrl_nxt.extend_from_slice(&prvt_nxt);

    // // Observable wires: intf + extl (separate latched and next)
    let mut obs_ltc: Vec<Wire<DType>> = Vec::new();
    obs_ltc.extend_from_slice(&intf_ltc);
    obs_ltc.extend_from_slice(&extl_ltc);
    let mut obs_nxt: Vec<Wire<DType>> = Vec::new();
    obs_nxt.extend_from_slice(&intf_nxt);
    obs_nxt.extend_from_slice(&extl_nxt);

    // // All wires in the module (separate latched and next)
    let mut all_ltc = Vec::new();
    all_ltc.extend_from_slice(&extl_ltc);
    all_ltc.extend_from_slice(&intf_ltc);
    all_ltc.extend_from_slice(&prvt_ltc);
    let mut all_nxt = Vec::new();
    all_nxt.extend_from_slice(&extl_nxt);
    all_nxt.extend_from_slice(&intf_nxt);
    all_nxt.extend_from_slice(&prvt_nxt);

    let mut init = vec![];
    let mut update = vec![];

    let w12 = Wire::new(DType::Real);
    // === INIT FLOW ===
    init.push(
        Term::function(
            IType::Num(Val::Real(3.24)),
            [w12.clone()],
            vec![] as Vec<Wire<DType>>,
        )
        .unwrap(),
    );
    let w13 = Wire::new(DType::Int);
    init.push(
        Term::function(
            IType::Num(Val::Int(42)),
            [w13.clone()],
            vec![] as Vec<Wire<DType>>,
        )
        .unwrap(),
    );
    let w14 = Wire::new(DType::Bool);
    init.push(
        Term::function(
            IType::Num(Val::Bool(true)),
            [w14.clone()],
            vec![] as Vec<Wire<DType>>,
        )
        .unwrap(),
    );

    init.push(
        Term::function(
            IType::Arith(ArithOp::Add),
            [w6.clone()],
            [w9.clone(), w12.clone()],
        )
        .unwrap(),
    );
    init.push(
        Term::function(
            IType::Arith(ArithOp::Sub),
            [w7.clone()],
            [w10.clone(), w13.clone()],
        )
        .unwrap(),
    );
    init.push(
        Term::function(
            IType::Logical(LogicalOp::And),
            [w8.clone()],
            [w11.clone(), w14.clone()],
        )
        .unwrap(),
    );

    let w15 = Wire::new(DType::Real);
    // === UPDATE FLOW ===
    update.push(
        Term::function(
            IType::Num(Val::Real(4.20)),
            [w15.clone()],
            vec![] as Vec<Wire<DType>>,
        )
        .unwrap(),
    );
    let w16 = Wire::new(DType::Real);
    update.push(
        Term::function(
            IType::Num(Val::Real(12.3)),
            [w16.clone()],
            vec![] as Vec<Wire<DType>>,
        )
        .unwrap(),
    );
    let w17 = Wire::new(DType::Real);
    update.push(
        Term::function(
            IType::Arith(ArithOp::Mul),
            [w17.clone()],
            [w0.clone(), w15.clone()],
        )
        .unwrap(),
    );
    update.push(
        Term::function(
            IType::Arith(ArithOp::Div),
            [w6.clone()],
            [w16.clone(), w17.clone()],
        )
        .unwrap(),
    );

    update.push(Term::function(IType::Id, [w7.clone()], [w1.clone()]).unwrap());

    let w18 = Wire::new(DType::Real);
    update.push(
        Term::function(
            IType::Num(Val::Real(50.05)),
            [w18.clone()],
            vec![] as Vec<Wire<DType>>,
        )
        .unwrap(),
    );
    let w19 = Wire::new(DType::Bool);
    update.push(
        Term::function(
            IType::Cmp(CmpOp::Lt),
            [w19.clone()],
            [w0.clone(), w18.clone()],
        )
        .unwrap(),
    );
    let w20 = Wire::new(DType::Int);
    update.push(
        Term::function(
            IType::Num(Val::Int(0)),
            [w20.clone()],
            vec![] as Vec<Wire<DType>>,
        )
        .unwrap(),
    );
    let w21 = Wire::new(DType::Bool);
    update.push(
        Term::function(
            IType::Cmp(CmpOp::Eq),
            [w21.clone()],
            [w1.clone(), w20.clone()],
        )
        .unwrap(),
    );
    let w22 = Wire::new(DType::Bool);
    update.push(
        Term::function(
            IType::Logical(LogicalOp::Or),
            [w22.clone()],
            [w19.clone(), w21.clone()],
        )
        .unwrap(),
    );
    let w23 = Wire::new(DType::Bool);
    update
        .push(Term::function(IType::Logical(LogicalOp::Not), [w23.clone()], [w2.clone()]).unwrap());
    let w24 = Wire::new(DType::Bool);
    update.push(
        Term::function(
            IType::Num(Val::Bool(false)),
            [w24.clone()],
            vec![] as Vec<Wire<DType>>,
        )
        .unwrap(),
    );
    update.push(
        Term::function(
            IType::Cond,
            [w8.clone()],
            [w22.clone(), w23.clone(), w24.clone()],
        )
        .unwrap(),
    );

    let obs_pairs = obs_ltc
        .iter()
        .zip(obs_nxt.iter())
        .map(|(ltc, nxt)| [ltc.clone(), nxt.clone()]);

    let prvt_pairs = prvt_ltc
        .iter()
        .zip(prvt_nxt.iter())
        .map(|(ltc, nxt)| [ltc.clone(), nxt.clone()]);

    let atom = Atom::sequential(all_ltc.iter(), all_nxt.iter(), init, update).unwrap();

    Module::new(obs_pairs, prvt_pairs, [atom]).unwrap()
}

#[test]
fn create_module_1() {
    let _module = create_test_module();
}
