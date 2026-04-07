use crate::*;

mk_theory!(
    bools, Types(Bool),
    And(Bool, Bool) => Bool,
    Or(Bool, Bool) => Bool,
    Xor(Bool, Bool) => Bool,
    Not(Bool) => Bool,
    Id(Bool) => Bool
);
