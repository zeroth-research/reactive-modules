use crate::mk_theory_mod;

mk_theory_mod!(
    nat, Types(Nat),
    Add(Nat, Nat) => Nat,
    Mul(Nat, Nat) => Nat,
    Id(Nat) => Nat
);
