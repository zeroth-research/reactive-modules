use crate::mk_theory;

mk_theory!(
    int, Types(Int),
    Add(Int, Int) => Int,
    Mul(Int, Int) => Int,
    Id(Int) => Int
);
