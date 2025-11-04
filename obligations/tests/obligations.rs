use base::term::{Term, TermWire};
use base::wire::Wire;
use obligations::BuchiObligations;
use smv::{dtype::DType, itype::IType};

#[test]
fn test_obligations() {
    let obligations = BuchiObligations {
        // x ≤ y ∨ x ≤ z
        // x = wire 0
        // y = wire 1
        // z = wire 2
        invariant: vec![
            Term::new(
                IType::Le,
                Wire::one(3, DType::Bool),
                Wire::many(0, DType::Int, 2),
            ),
            Term::new(
                IType::Le,
                Wire::one(4, DType::Bool),
                Wire::one(0, DType::Int)
                    .concat(&Wire::one(2, DType::Int)),
            ),
            Term::new(
                IType::Or,
                Wire::one(5, DType::Bool),
                Wire::many(3, DType::Bool, 2),
            ),
        ],
        // ReLU(y − x) + ReLU(z − x)
        // x = wire 0
        // y = wire 1
        // z = wire 2
        variant: vec![
            Term::new(
                IType::Sub,
                // TODO: argument order is important here
                Wire::one(3, DType::Int),
                Wire::many(0, DType::Int, 2),
            ),
            Term::new(
                IType::Sub,
                // TODO: argument order is important here
                Wire::one(4, DType::Int),
                Wire::one(0, DType::Int)
                    .concat(&Wire::one(2, DType::Int)),
            ),
            // ReLU(x) = Cond(x < 0, 0, x)
            Term::new(IType::ConstInt(0), Wire::one(5, DType::Int), Wire::none()),
            // x6 = x3 < x5
            Term::new(
                IType::Lt,
                Wire::one(6, DType::Bool),
                Wire::one(3, DType::Int)
                    .concat(&Wire::one(5, DType::Int)),
            ),
            // x7 = x4 < x5
            Term::new(
                IType::Lt,
                Wire::one(7, DType::Bool),
                Wire::one(4, DType::Int)
                    .concat(&Wire::one(5, DType::Int)),
            ),
            // x8 = Cond(x6, x5, x3)
            Term::new(
                IType::Cond,
                Wire::one(8, DType::Int),
                Wire::one(6, DType::Bool)
                    .concat(&Wire::one(5, DType::Int))
                    .concat(&Wire::one(3, DType::Int)),
            ),
            // x9 = Cond(x7, x5, x4)
            Term::new(
                IType::Cond,
                Wire::one(9, DType::Int),
                Wire::one(7, DType::Bool)
                    .concat(&Wire::one(5, DType::Int))
                    .concat(&Wire::one(4, DType::Int)),
            ),
            // x10 = x8 + x9
            Term::new(
                IType::Add,
                Wire::one(10, DType::Int),
                Wire::many(8, DType::Int, 2),
            ),
        ],
        // x = y ∨ x = z
        // x = wire 0
        // y = wire 1
        // z = wire 2
        buchi: vec![
            Term::new(
                IType::Eq,
                Wire::one(3, DType::Bool),
                Wire::many(0, DType::Int, 2),
            ),
            Term::new(
                IType::Eq,
                Wire::one(4, DType::Bool),
                Wire::one(0, DType::Int)
                    .concat(&Wire::one(2, DType::Int)),
            ),
            Term::new(
                IType::Or,
                Wire::one(5, DType::Bool),
                Wire::many(3, DType::Bool, 2),
            ),
        ],
    };

    assert_eq!(obligations.invariant.read(), Wire::many(0, DType::Int, 3));
    assert_eq!(obligations.invariant.write(), Wire::one(5, DType::Bool));

    assert_eq!(obligations.buchi.read(), Wire::many(0, DType::Int, 3));
    assert_eq!(obligations.buchi.write(), Wire::one(5, DType::Bool));

    assert_eq!(obligations.variant.read(), Wire::many(0, DType::Int, 3));
    assert_eq!(obligations.variant.write(), Wire::one(10, DType::Int));
}
