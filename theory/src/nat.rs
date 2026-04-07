use crate::mk_theory;

mk_theory!(
    nat, Types(Nat),
    Add(Nat, Nat) => Nat,
    Mul(Nat, Nat) => Nat,
    Id(Nat) => Nat
);
