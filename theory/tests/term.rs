use theory::*;

struct Term<T: Theory> {
    itype: T::IT,
    read: Vec<T::DT>,
    write: Vec<T::DT>,
}

impl<T: Theory> Term<T> {
    fn mk_bin_op<O: TheoryOperation<T>, T1: TheoryType<T>, T2: TheoryType<T>, R: TheoryType<T>>(
        op: O,
        read: (T1, T2),
        write: R,
    ) -> Term<T>
    where
        O: Into<T::IT>,              // the operation is an operation of IType
        O: Operation2To1<T1, T2, R>, // it is a binary operation
    {
        Term {
            itype: op.into(),
            read: vec![read.0.into(), read.1.into()],
            write: vec![write.into()],
        }
    }

    fn mk_unary_op<O: Operation, T1: TheoryType<T>, R: TheoryType<T>>(
        op: O,
        read: T1,
        write: R,
    ) -> Term<T>
    where
        O: Into<T::IT>,          // the operation is an operation of IType
        O: Operation1To1<T1, R>, // it is a binary operation
    {
        Term {
            itype: op.into(),
            read: vec![read.into()],
            write: vec![write.into()],
        }
    }
}

macro_rules! mk_term {
    ($op:expr, ($r1:expr, $r2:expr), $w:expr) => {
        Term::mk_bin_op($op, ($r1, $r2), $w)
    };
    ($op:expr, $r:expr, $w:expr) => {
        Term::mk_unary_op($op, $r, $w)
    };
}

#[test]
fn test_typecheck1() {
    use theory::nat::*;
    let _: Term<Theory> = mk_term!(Add(), (Nat(), Nat()), Nat());
    let _: Term<Theory> = mk_term!(Id(), Nat(), Nat());
}

#[test]
fn test_typecheck2() {
    {
        use theory::nat::Theory as NatTheory;
        use theory::nat::*;
        let _: Term<NatTheory> = mk_term!(Add(), (Nat(), Nat()), Nat());
    }

    {
        let _: Term<int::Theory> = mk_term!(int::Id(), int::Int(), int::Int());
    }
}
