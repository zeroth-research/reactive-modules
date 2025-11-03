use toy::instruction::{ArithOp, Instruction};
use toy::interpreter::eval;
use toy::val::Val;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const() {
        let mut res: Val = Val::new();
        assert!(
            eval(
                Instruction::Const(Val::Bool(true)),
                &vec![],
                &mut vec![&mut res]
            ) == Ok(())
        );
        assert!(res == Val::Bool(true));

        assert!(
            eval(
                Instruction::Const(Val::Int(-13)),
                &vec![],
                &mut vec![&mut res]
            ) == Ok(())
        );
        assert!(res == Val::Int(-13));
    }

    #[test]
    fn test_id() {
        let mut res: Val = Val::new();
        assert!(eval(Instruction::Id, &vec![&Val::Int(42)], &mut vec![&mut res]) == Ok(()));
        assert!(res == Val::Int(42));
    }

    #[test]
    #[should_panic]
    fn test_id_fail() {
        let mut res: Val = Val::new();
        assert!(eval(Instruction::Id, &vec![], &mut vec![&mut res]) == Ok(()));
        assert!(res == Val::Int(42));
    }

    #[test]
    #[should_panic]
    fn test_id_fail2() {
        let mut res: Val = Val::new();
        assert!(
            eval(
                Instruction::Id,
                &vec![&Val::Int(32), &Val::None],
                &mut vec![&mut res]
            ) == Ok(())
        );
        assert!(res == Val::Int(42));
    }

    #[test]
    fn test_add() {
        let mut res: Val = Val::new();
        assert!(
            eval(
                Instruction::Arith(ArithOp::Add),
                &vec![&Val::Int(42), &Val::Int(43)],
                &mut vec![&mut res]
            ) == Ok(())
        );
        assert!(res == Val::Int(85));

        assert!(
            eval(
                Instruction::Arith(ArithOp::Add),
                &vec![&Val::Real(42.0), &Val::Real(43.0)],
                &mut vec![&mut res]
            ) == Ok(())
        );
        assert!(res == Val::Real(85.0));

        // invalid arguments
        assert!(
            eval(
                Instruction::Arith(ArithOp::Add),
                &vec![&Val::Int(42), &Val::Real(43.0)],
                &mut vec![&mut res]
            )
            .is_err()
        );
    }
}
