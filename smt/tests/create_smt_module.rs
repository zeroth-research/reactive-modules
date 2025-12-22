use base::{atom::Atom, module::Module, term::Term, wire::Wire};
use smt::dtype::DType;
use smt::itype::{ArithOp, CmpOp, IType, LogicalOp, Val};

pub fn create_test_module() -> Module<DType, IType> {
    // Interface wires: w0-w2 (latched), w6-w8 (next)
    let intf_ltc = vec![
        Wire::new(0, DType::Real),
        Wire::new(1, DType::Int),
        Wire::new(2, DType::Bool),
    ];
    let intf_nxt = vec![
        Wire::new(6, DType::Real),
        Wire::new(7, DType::Int),
        Wire::new(8, DType::Bool),
    ];

    // // External wires: w3-w5 (latched), w9-w11 (next)
    let extl_ltc = vec![
        Wire::new(3, DType::Real),
        Wire::new(4, DType::Int),
        Wire::new(5, DType::Bool),
    ];
    let extl_nxt = vec![
        Wire::new(9, DType::Real),
        Wire::new(10, DType::Int),
        Wire::new(11, DType::Bool),
    ];

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

    // === INIT FLOW ===
    init.push(
        Term::function(
            IType::Num(Val::Real(3.24)),
            [Wire::new(12, DType::Real)],
            vec![] as Vec<Wire<DType>>,
        )
        .unwrap(),
    );
    init.push(
        Term::function(
            IType::Num(Val::Int(42)),
            [Wire::new(13, DType::Int)],
            vec![] as Vec<Wire<DType>>,
        )
        .unwrap(),
    );
    init.push(
        Term::function(
            IType::Num(Val::Bool(true)),
            [Wire::new(14, DType::Bool)],
            vec![] as Vec<Wire<DType>>,
        )
        .unwrap(),
    );

    init.push(
        Term::function(
            IType::Arith(ArithOp::Add),
            [Wire::new(6, DType::Real)],
            [Wire::new(9, DType::Real), Wire::new(12, DType::Real)],
        )
        .unwrap(),
    );
    init.push(
        Term::function(
            IType::Arith(ArithOp::Sub),
            [Wire::new(7, DType::Int)],
            [Wire::new(10, DType::Int), Wire::new(13, DType::Int)],
        )
        .unwrap(),
    );
    init.push(
        Term::function(
            IType::Logical(LogicalOp::And),
            [Wire::new(8, DType::Bool)],
            [Wire::new(11, DType::Bool), Wire::new(14, DType::Bool)],
        )
        .unwrap(),
    );

    // === UPDATE FLOW ===
    update.push(
        Term::function(
            IType::Num(Val::Real(4.20)),
            [Wire::new(15, DType::Real)],
            vec![] as Vec<Wire<DType>>,
        )
        .unwrap(),
    );
    update.push(
        Term::function(
            IType::Num(Val::Real(12.3)),
            [Wire::new(16, DType::Real)],
            vec![] as Vec<Wire<DType>>,
        )
        .unwrap(),
    );
    update.push(
        Term::function(
            IType::Arith(ArithOp::Mul),
            [Wire::new(17, DType::Real)],
            [Wire::new(0, DType::Real), Wire::new(15, DType::Real)],
        )
        .unwrap(),
    );
    update.push(
        Term::function(
            IType::Arith(ArithOp::Div),
            [Wire::new(6, DType::Real)],
            [Wire::new(16, DType::Real), Wire::new(17, DType::Real)],
        )
        .unwrap(),
    );

    update.push(
        Term::function(
            IType::Id,
            [Wire::new(7, DType::Int)],
            [Wire::new(1, DType::Int)],
        )
        .unwrap(),
    );

    update.push(
        Term::function(
            IType::Num(Val::Real(50.05)),
            [Wire::new(18, DType::Real)],
            vec![] as Vec<Wire<DType>>,
        )
        .unwrap(),
    );
    update.push(
        Term::function(
            IType::Cmp(CmpOp::Lt),
            [Wire::new(19, DType::Bool)],
            [Wire::new(0, DType::Real), Wire::new(18, DType::Real)],
        )
        .unwrap(),
    );
    update.push(
        Term::function(
            IType::Num(Val::Int(0)),
            [Wire::new(20, DType::Int)],
            vec![] as Vec<Wire<DType>>,
        )
        .unwrap(),
    );
    update.push(
        Term::function(
            IType::Cmp(CmpOp::Eq),
            [Wire::new(21, DType::Bool)],
            [Wire::new(1, DType::Int), Wire::new(20, DType::Int)],
        )
        .unwrap(),
    );
    update.push(
        Term::function(
            IType::Logical(LogicalOp::Or),
            [Wire::new(22, DType::Bool)],
            [Wire::new(19, DType::Bool), Wire::new(21, DType::Bool)],
        )
        .unwrap(),
    );
    update.push(
        Term::function(
            IType::Logical(LogicalOp::Not),
            [Wire::new(23, DType::Bool)],
            [Wire::new(2, DType::Bool)],
        )
        .unwrap(),
    );
    update.push(
        Term::function(
            IType::Num(Val::Bool(false)),
            [Wire::new(24, DType::Bool)],
            vec![] as Vec<Wire<DType>>,
        )
        .unwrap(),
    );
    update.push(
        Term::function(
            IType::Cond,
            [Wire::new(8, DType::Bool)],
            [
                Wire::new(22, DType::Bool),
                Wire::new(23, DType::Bool),
                Wire::new(24, DType::Bool),
            ],
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

    Module::partially_observable(obs_pairs, prvt_pairs, [atom]).unwrap()
}

#[test]
fn create_module_1() {
    let _module = create_test_module();
}
