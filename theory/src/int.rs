use crate::mk_theory_mod;

mk_theory_mod!(
    int, Types(Int),
    Add(Int, Int) => Int,
    Mul(Int, Int) => Int,
    Id(Int) => Int
);
