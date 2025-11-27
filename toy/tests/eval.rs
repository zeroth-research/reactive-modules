use toy::interpreter::eval;
use toy::itype::{ArithOp, IType};
use toy::val::Val;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const() {
        let mut res: Val = Val::new();
        assert!(eval(IType::Const(Val::Bool(true)), &[], &mut [&mut res]) == Ok(()));
        assert!(res == Val::Bool(true));

        assert!(eval(IType::Const(Val::Int(-13)), &[], &mut [&mut res]) == Ok(()));
        assert!(res == Val::Int(-13));
    }

    #[test]
    fn test_id() {
        let mut res: Val = Val::new();
        assert!(eval(IType::Id, &[&Val::Int(42)], &mut [&mut res]) == Ok(()));
        assert!(res == Val::Int(42));
    }

    #[test]
    #[should_panic]
    fn test_id_fail() {
        let mut res: Val = Val::new();
        assert!(eval(IType::Id, &[], &mut [&mut res]) == Ok(()));
        assert!(res == Val::Int(42));
    }

    #[test]
    #[should_panic]
    fn test_id_fail2() {
        let mut res: Val = Val::new();
        assert!(eval(IType::Id, &[&Val::Int(32), &Val::None], &mut [&mut res]) == Ok(()));
        assert!(res == Val::Int(42));
    }

    #[test]
    fn test_add() {
        let mut res: Val = Val::new();
        assert!(
            eval(
                IType::Arith(ArithOp::Add),
                &[&Val::Int(42), &Val::Int(43)],
                &mut [&mut res]
            ) == Ok(())
        );
        assert!(res == Val::Int(85));

        assert!(
            eval(
                IType::Arith(ArithOp::Add),
                &[&Val::Real(42.0), &Val::Real(43.0)],
                &mut [&mut res]
            ) == Ok(())
        );
        assert!(res == Val::Real(85.0));

        // invalid arguments
        assert!(
            eval(
                IType::Arith(ArithOp::Add),
                &[&Val::Int(42), &Val::Real(43.0)],
                &mut [&mut res]
            )
            .is_err()
        );
    }
}
